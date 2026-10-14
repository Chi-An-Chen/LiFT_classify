"""Evaluate one saved classifier at fixed threshold 0.5, optionally across loop counts."""

import argparse
from pathlib import Path

from lift_classify.fas_data import FASDataset, FASTransform, discover_records
from lift_classify.fas_runtime import evaluate, load_model, resolve_device, write_csv, write_json
from lift_classify.fas_training import make_loader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--loops", type=int, nargs="+")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or (args.loops and min(args.loops) < 1):
        parser.error("batch-size/loops must be positive and num-workers nonnegative")
    if args.output_dir.resolve().is_relative_to(args.data_root.resolve()):
        parser.error("output-dir must be outside the source dataset")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output-dir must be empty to preserve existing results")
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    records = discover_records(args.data_root, "test")
    dataset = FASDataset(args.data_root, records, FASTransform(checkpoint["preprocessing"]))
    loader = make_loader(dataset, dict(batch_size=args.batch_size, num_workers=args.num_workers, seed=42), device)
    results = {}
    for loops in args.loops or [model.config.num_loops]:
        metrics, predictions = evaluate(model, loader, device, loops)
        results[str(loops)] = metrics
        write_csv(args.output_dir / f"validation_predictions_k{loops}.csv", predictions)
    report = dict(checkpoint=str(args.checkpoint.resolve()), data_root=str(args.data_root.resolve()),
                  split="test", role="validation", precision="fp32", device=str(device),
                  metrics_by_num_loops=results, checkpoint_training_protocol=checkpoint["protocol"],
                  checkpoint_cross_split_duplicate_groups=checkpoint["cross_split_duplicate_groups"],
                  overlap_audit="Not recomputed for this evaluation dataset; see training data_audit.json")
    write_json(args.output_dir / "validation_report.json", report)
    print(results)


if __name__ == "__main__":
    main()
