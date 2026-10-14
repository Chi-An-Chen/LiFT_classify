"""Full-image FAS data, conservative lighting changes, and read-only data audits."""

from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import csv
import hashlib
import io
import math
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, InterpolationMode
from torchvision.transforms import functional as TF


CLASSES = {"fake": 0, "live": 1}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PREPROCESSING = {
    "version": 1, "resize": "longest_side_bilinear_antialias", "exif_transpose": True,
    "padding_rgb": [124, 116, 104], "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
AUGMENTATION = {
    "version": 1, "jitter_probability": 0.5, "brightness": 0.15,
    "contrast": 0.15, "saturation": 0.10, "hue": 0.0,
    "lighting_probability": 0.3, "gamma_range": [0.9, 1.1],
    "channel_gain_range": [0.95, 1.05], "directional_amplitude": 0.15,
}


def preprocessing_spec(size: int) -> dict:
    return dict(PREPROCESSING, image_size=size)


class FASTransform:
    def __init__(self, spec: dict, training: bool = False):
        if spec != preprocessing_spec(spec.get("image_size")):
            raise ValueError("Unsupported checkpoint preprocessing recipe")
        self.size = spec["image_size"]
        if not isinstance(self.size, int) or isinstance(self.size, bool) or self.size < 1:
            raise ValueError("image_size must be a positive integer")
        self.training = training
        self.jitter = ColorJitter(AUGMENTATION["brightness"], AUGMENTATION["contrast"],
                                 AUGMENTATION["saturation"], AUGMENTATION["hue"])

    def lighting(self, pixels: torch.Tensor, kind: int) -> torch.Tensor:
        if kind == 0:
            low, high = AUGMENTATION["gamma_range"]
            pixels = pixels.pow(float(torch.empty(()).uniform_(low, high)))
        elif kind == 1:
            low, high = AUGMENTATION["channel_gain_range"]
            pixels = pixels * torch.empty(3, 1, 1).uniform_(low, high)
        elif kind == 2:
            height, width = pixels.shape[-2:]
            y = torch.linspace(-1, 1, height)[:, None]
            x = torch.linspace(-1, 1, width)[None, :]
            angle = float(torch.rand(())) * 2 * math.pi
            amplitude = float(torch.empty(()).uniform_(-AUGMENTATION["directional_amplitude"],
                                                       AUGMENTATION["directional_amplitude"]))
            field = 1 + amplitude * (x * math.cos(angle) + y * math.sin(angle)) / math.sqrt(2)
            pixels = pixels * field
        else:
            raise ValueError("Unknown lighting operation")
        return pixels.clamp(0, 1)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        scale = self.size / max(width, height)
        new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
        image = TF.resize(image, [new_height, new_width], InterpolationMode.BILINEAR, antialias=True)
        pixels = TF.to_tensor(image)
        if self.training:
            if torch.rand(()) < AUGMENTATION["jitter_probability"]:
                pixels = self.jitter(pixels)
            if torch.rand(()) < AUGMENTATION["lighting_probability"]:
                pixels = self.lighting(pixels, int(torch.randint(3, ())))
        canvas = torch.tensor(PREPROCESSING["padding_rgb"], dtype=torch.float32)[:, None, None]
        canvas = (canvas / 255).expand(3, self.size, self.size).clone()
        top, left = (self.size - new_height) // 2, (self.size - new_width) // 2
        canvas[:, top:top + new_height, left:left + new_width] = pixels
        return TF.normalize(canvas, PREPROCESSING["mean"], PREPROCESSING["std"])


def discover_records(root: Path, split: str) -> list[dict]:
    root = Path(root).resolve()
    records = []
    for name, label in CLASSES.items():
        directory = root / split / name
        files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f"No images in {directory}")
        for path in files:
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Image points outside data root: {path}")
            records.append(dict(split=split, label=label, class_name=name,
                                relative_path=path.relative_to(root).as_posix()))
    return records


def audit_dataset(root: Path) -> tuple[list[dict], dict]:
    """Decode and hash every image. Never remove or rewrite dataset files."""
    root = Path(root)
    records = discover_records(root, "train") + discover_records(root, "test")
    manifest = root / "manifest.csv"
    if manifest.exists():
        with manifest.open(newline="", encoding="utf-8-sig") as stream:
            supplied = list(csv.DictReader(stream))
        expected = {(r["split"], r["class_name"], r["relative_path"]) for r in records}
        actual = {(r["split"], r["label"], r["relative_path"]) for r in supplied}
        if actual != expected or len(supplied) != len(actual):
            raise ValueError("manifest.csv must match all images, splits and labels without repeated rows")

    def inspect(record):
        path = root / record["relative_path"]
        try:
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                width, height = image.size
                mode = image.mode
        except Exception as error:
            raise ValueError(f"Cannot decode image {path}: {error}") from error
        return dict(record, sha256=hashlib.sha256(data).hexdigest(), width=width, height=height, mode=mode)

    checked = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, record in enumerate(pool.map(inspect, records), 1):
            checked.append(record)
            if index % 5000 == 0:
                print(f"Data audit: {index}/{len(records)} images", flush=True)
    groups = defaultdict(list)
    for record in checked:
        groups[record["sha256"]].append(record)
    conflicts = [group for group in groups.values() if len({r["label"] for r in group}) > 1]
    if conflicts:
        raise ValueError(f"Identical image bytes have conflicting labels: {conflicts[0]}")
    cross = [group for group in groups.values() if len({r["split"] for r in group}) > 1]
    digest = hashlib.sha256()
    for record in checked:
        digest.update(f"{record['relative_path']}\t{record['label']}\t{record['sha256']}\n".encode())
    report = dict(
        fingerprint=digest.hexdigest(), images=len(checked), unique_hashes=len(groups),
        counts=dict(Counter(f"{r['split']}/{r['class_name']}" for r in checked)),
        cross_split_duplicate_groups=len(cross),
        cross_split_image_counts=dict(Counter(f"{r['split']}/{r['class_name']}" for g in cross for r in g)),
        cross_split_duplicates=[[r["relative_path"] for r in g] for g in cross],
        within_train_duplicate_groups=sum(sum(r["split"] == "train" for r in g) > 1 for g in groups.values()),
        protocol="Original train/test retained; test is validation for model selection; duplicates retained.",
        limitation="Byte-level audit only; subject/video/session independence and near-duplicates are unknown.",
    )
    return checked, report


class FASDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], transform: FASTransform):
        self.root, self.records, self.transform = Path(root), records, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.root / record["relative_path"]
        try:
            with Image.open(path) as image:
                pixels = self.transform(image)
        except Exception as error:
            raise ValueError(f"Cannot preprocess image {path}: {error}") from error
        return pixels, record["label"], record["relative_path"]
