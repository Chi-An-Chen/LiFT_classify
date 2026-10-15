"""Predict fake/live probabilities for an image or image directory on CUDA, MPS or CPU."""

import argparse
from pathlib import Path

from PIL import Image
import torch

from lift_classify.fas_data import EXTENSIONS, FASTransform
from lift_classify.fas_runtime import load_model, resolve_device, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--num-loops", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new CSV path")
    if args.num_loops is not None and args.num_loops < 1:
        parser.error("num-loops must be positive")
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    transform = FASTransform(checkpoint["preprocessing"])
    input_is_directory = args.input.is_dir()
    files = sorted(p for p in args.input.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS) if input_is_directory else [args.input]
    if not files:
        parser.error("No input images found")
    records = []
    with torch.inference_mode():
        for path in files:
            with Image.open(path) as image:
                pixels = transform(image).unsqueeze(0).to(device)
            logits = model(pixels, num_loops=args.num_loops).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite prediction for {path}")
            probability = logits.softmax(-1)[0, 1].item()
            reported_path = path.relative_to(args.input).as_posix() if input_is_directory else path.name
            records.append(dict(path=reported_path, p_live=probability, p_fake=1-probability,
                                prediction="live" if probability >= 0.5 else "fake",
                                threshold=0.5, num_loops=args.num_loops or model.config.num_loops))
    write_csv(args.output, records)
    print(f"Predicted {len(records)} images on {device} (FP32): {args.output}")


if __name__ == "__main__":
    main()
