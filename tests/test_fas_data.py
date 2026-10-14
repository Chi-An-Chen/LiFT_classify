import csv
import pytest
import torch

np = pytest.importorskip("numpy")
pytest.importorskip("torchvision")
pytest.importorskip("sklearn")
from PIL import Image

from lift_classify.fas_data import (CLASSES, PREPROCESSING, FASDataset, FASTransform,
                                    audit_dataset, discover_records, preprocessing_spec)


@pytest.fixture
def fas_root(tmp_path):
    root = tmp_path / "data"
    rng = np.random.default_rng(123)
    for split in ("train", "test"):
        for name in CLASSES:
            directory = root / split / name
            directory.mkdir(parents=True)
            for i in range(3):
                Image.fromarray(rng.integers(0, 256, (24, 16, 3), dtype=np.uint8)).save(directory / f"{i}.png")
    return root


def denormalize(x):
    return x * torch.tensor(PREPROCESSING["std"])[:, None, None] + torch.tensor(PREPROCESSING["mean"])[:, None, None]


def test_full_frame_letterbox_and_class_order(fas_root):
    transform = FASTransform(preprocessing_spec(24))
    array = np.zeros((24, 12, 3), dtype=np.uint8)
    array[0:3] = [255, 0, 0]
    array[-3:] = [0, 255, 0]
    result = denormalize(transform(Image.fromarray(array)))
    torch.testing.assert_close(result[:, :, 0], torch.tensor([124, 116, 104])[:, None].float().expand(3,24)/255)
    torch.testing.assert_close(result[:, 0, 12], torch.tensor([1., 0., 0.]), atol=1e-6, rtol=0)
    torch.testing.assert_close(result[:, -1, 12], torch.tensor([0., 1., 0.]), atol=1e-6, rtol=0)
    assert result.shape == (3, 24, 24)
    records = discover_records(fas_root, "train")
    assert {r["class_name"]: r["label"] for r in records} == {"fake":0,"live":1}
    dataset = FASDataset(fas_root, records, transform)
    assert dataset[0][0].shape == (3,24,24)


def test_exif_orientation_and_rgb_conversion():
    image = Image.new("RGB", (12, 24), (255,0,0))
    image.getexif()[274] = 6
    result = denormalize(FASTransform(preprocessing_spec(24))(image))
    torch.testing.assert_close(result[:, 12, 0], torch.tensor([1.,0.,0.]), atol=1e-6, rtol=0)
    assert FASTransform(preprocessing_spec(16))(Image.new("L", (10,10))).shape == (3,16,16)


def test_augmentation_is_seeded_bounded_and_leaves_padding_unchanged():
    transform = FASTransform(preprocessing_spec(24), training=True)
    image = Image.new("RGB", (12,24), (150,120,80))
    torch.manual_seed(42)
    first = transform(image)
    torch.manual_seed(42)
    torch.testing.assert_close(first, transform(image), atol=0, rtol=0)
    variants = [denormalize(transform(image)) for _ in range(20)]
    assert any(not torch.equal(variants[0], value) for value in variants[1:])
    for value in variants:
        assert value.min() >= -1e-6 and value.max() <= 1+1e-6
        torch.testing.assert_close(value[:,:,0], torch.tensor([124,116,104])[:,None].float().expand(3,24)/255)
    for kind in (0,1,2):
        result = transform.lighting(torch.full((3,16,12), .5), kind)
        assert result.shape == (3,16,12) and result.min() > .4 and result.max() < .6
    evaluation = FASTransform(preprocessing_spec(24))
    torch.testing.assert_close(evaluation(image), evaluation(image), atol=0, rtol=0)


def test_audit_keeps_duplicates_and_manifest_is_checked(fas_root):
    source = fas_root / "train/live/0.png"
    (fas_root / "test/live/0.png").write_bytes(source.read_bytes())
    records, report = audit_dataset(fas_root)
    assert len(records) == 12 and report["cross_split_duplicate_groups"] == 1
    assert report["cross_split_image_counts"] == {"train/live":1,"test/live":1}
    assert len(discover_records(fas_root,"train")) == 6
    with (fas_root / "manifest.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["split","label","relative_path"])
        writer.writeheader()
        writer.writerows({"split":r["split"],"label":r["class_name"],"relative_path":r["relative_path"]} for r in records)
    assert audit_dataset(fas_root)[1]["fingerprint"] == report["fingerprint"]
    with (fas_root/"manifest.csv").open("a") as f:
        f.write("test,live,test/live/missing.png\n")
    with pytest.raises(ValueError,match="manifest"):
        audit_dataset(fas_root)


def test_corrupt_images_fail_with_path_and_conflicting_labels_fail(fas_root):
    bad=fas_root / "test/live/0.png"
    bad.write_bytes(b"broken")
    with pytest.raises(ValueError,match="test/live/0.png"):
        audit_dataset(fas_root)
    bad.write_bytes((fas_root/"train/fake/0.png").read_bytes())
    with pytest.raises(ValueError,match="conflicting labels"):
        audit_dataset(fas_root)


def test_transform_rejects_unknown_checkpoint_recipe():
    spec=preprocessing_spec(16)
    spec["resize"]="center_crop"
    with pytest.raises(ValueError,match="Unsupported"):
        FASTransform(spec)
