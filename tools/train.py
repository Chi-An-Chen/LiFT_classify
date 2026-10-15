"""Train LiFT on the original FAS train/test (validation) directories."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lift_classify.fas_runtime import read_checkpoint
from lift_classify.fas_training import run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.config is None and args.resume is not None:
        config = read_checkpoint(args.resume)["config"]
    else:
        path = args.config or Path(__file__).resolve().parents[1] / "configs/fas_lift_tiny.json"
        config = json.loads(path.read_text())
    for field in ("epochs", "batch_size", "num_workers"):
        value = getattr(args, field)
        if value is not None:
            config["training"][field] = value
    run_training(config, args.data_root, args.output_dir, args.device, args.resume)


if __name__ == "__main__":
    main()
