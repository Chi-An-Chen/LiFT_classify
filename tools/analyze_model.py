"""Analytic parameter/FLOP comparisons and optional LiFT-only CPU timing.

Run from the repository: python -m tools.analyze_model --loops 1 2 4 8
"""

import argparse
from dataclasses import asdict, replace
import json
import platform

import torch
from torch.utils.benchmark import Timer

from lift_classify import LiFTClassifier, LiFTConfig, paper_reference_config, small_config, tiny_config
from lift_classify.config import spatial_pair, validate_integer


def estimate_costs(config: LiFTConfig, num_loops: int | None = None,
                   image_size: int | tuple[int, int] | None = None) -> list[dict]:
    """Per-example multiply-add=2 estimates; elementwise operations are excluded.

    ViT baselines use pre-LayerNorm, biased projections, absolute positions, and GAP.
    Looped ViT adds non-affine boundary RMSNorm and reinjection but no depth condition.
    The fourth row isolates sharing with the same LiFT block design.
    """
    loops = config.num_loops if num_loops is None else num_loops
    validate_integer(loops, "num_loops")
    height, width = config.image_size if image_size is None else spatial_pair(image_size, "image_size")
    ph, pw = config.patch_size
    if height % ph or width % pw:
        raise ValueError("image_size must be divisible by patch_size")
    tokens = (height // ph) * (width // pw)
    dim = config.hidden_dim
    mlp_dim = int(dim * config.mlp_ratio)
    head_dim = dim // config.num_heads
    patch_dim = config.in_channels * ph * pw
    unique = config.prelude_depth + config.core_depth
    executed = config.prelude_depth + loops * config.core_depth
    patch_params = dim * patch_dim + dim
    head_params = dim * config.num_classes + config.num_classes
    vit_block_params = 4 * dim**2 + 2 * dim * mlp_dim + 9 * dim + mlp_dim
    lift_block_params = 10 * dim**2 + 2 * dim * mlp_dim + 11 * dim + mlp_dim + 2 * head_dim
    vit_common_params = patch_params + tokens * dim + 2 * dim + head_params
    depth_params = dim**2 + 258 * dim
    lift_common_params = patch_params + dim + depth_params + 2 * dim**2 + 2 * dim + head_params
    vit_block_flops = tokens * (8 * dim**2 + 4 * dim * mlp_dim + 4 * tokens * dim)
    lift_block_flops = vit_block_flops + 12 * dim**2
    io_flops = 2 * tokens * patch_dim * dim + 2 * dim * config.num_classes
    depth_flops = 2 * (256 * dim + dim**2)
    readout_modulation_flops = 4 * dim**2

    def row(name, blocks, params, flops):
        return dict(model=name, num_loops=loops, tokens=tokens, unique_blocks=blocks,
                    executed_blocks=executed, parameters=params, forward_flops=flops)

    return [
        row("Dense ViT", executed, executed * vit_block_params + vit_common_params,
            executed * vit_block_flops + io_flops),
        row("Looped ViT", unique, unique * vit_block_params + vit_common_params,
            executed * vit_block_flops + io_flops),
        row("LiFT classification", unique, unique * lift_block_params + lift_common_params,
            executed * lift_block_flops + io_flops + loops * depth_flops + readout_modulation_flops),
        row("Dense / LiFT blocks", executed,
            executed * lift_block_params + lift_common_params - depth_params,
            executed * lift_block_flops + io_flops + readout_modulation_flops),
    ]


def benchmark_cpu(model: LiFTClassifier, loops: list[int], batch_size: int = 1,
                  num_threads: int = 1, min_run_time: float = 0.2) -> dict:
    validate_integer(batch_size, "batch_size")
    validate_integer(num_threads, "num_threads")
    if min_run_time <= 0:
        raise ValueError("min_run_time must be positive")
    for count in loops:
        validate_integer(count, "num_loops")
    model = model.cpu().float().eval()
    images = torch.randn(batch_size, model.config.in_channels, *model.config.image_size)
    previous_threads = torch.get_num_threads()
    measurements = []
    try:
        torch.set_num_threads(num_threads)
        with torch.inference_mode():
            for count in loops:
                for _ in range(3):
                    model(images, num_loops=count)
                result = Timer(
                    stmt="model(images, num_loops=count)",
                    globals={"model": model, "images": images, "count": count},
                    num_threads=num_threads,
                ).blocked_autorange(min_run_time=min_run_time)
                measurements.append(dict(num_loops=count, median_ms=result.median * 1000,
                                         iqr_ms=result.iqr * 1000))
    finally:
        torch.set_num_threads(previous_threads)
    return dict(platform=platform.platform(), machine=platform.machine(),
                processor=platform.processor(), torch_version=str(torch.__version__),
                device="cpu", dtype="float32", num_threads=num_threads,
                batch_size=batch_size, config=asdict(model.config), measurements=measurements)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=["tiny", "small", "paper-B", "paper-L", "paper-XL"], default="tiny")
    parser.add_argument("--rgb", action="store_true", help="Adapt a paper preset to RGB 224 / patch 16")
    parser.add_argument("--loops", nargs="+", type=int, default=[1, 2, 4, 8])
    for field in ("image_size", "patch_size", "in_channels", "num_classes", "hidden_dim",
                  "num_heads", "prelude_depth", "core_depth"):
        parser.add_argument("--" + field.replace("_", "-"), type=int)
    for field in ("mlp_ratio", "dropout", "attention_dropout"):
        parser.add_argument("--" + field.replace("_", "-"), type=float)
    parser.add_argument("--latency", action="store_true", help="Time LiFT only on CPU; allocates the model")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=0.2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.preset.startswith("paper-"):
        config = paper_reference_config(args.preset.removeprefix("paper-"))
    else:
        config = tiny_config() if args.preset == "tiny" else small_config()
    overrides = {field: getattr(args, field) for field in LiFTConfig.__dataclass_fields__
                 if hasattr(args, field) and getattr(args, field) is not None}
    if args.rgb:
        overrides = {"image_size": 224, "patch_size": 16, "in_channels": 3} | overrides
    config = replace(config, **overrides)
    rows = [row for count in args.loops for row in estimate_costs(config, count)]
    report = dict(config=asdict(config), estimates=rows,
                  accounting="Per image, final readout only, multiply-add=2; excludes elementwise operations.")
    if args.latency:
        report["cpu_latency"] = benchmark_cpu(LiFTClassifier(config), args.loops, args.batch_size,
                                              args.threads, args.min_run_time)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print(report["accounting"])
    print(f"{'Model':<23} {'K':>3} {'Unique':>7} {'Executed':>9} {'Params(M)':>11} {'GFLOPs':>10}")
    for row in rows:
        print(f"{row['model']:<23} {row['num_loops']:>3} {row['unique_blocks']:>7} "
              f"{row['executed_blocks']:>9} {row['parameters']/1e6:>11.4f} {row['forward_flops']/1e9:>10.4f}")
    if args.latency:
        print(json.dumps(report["cpu_latency"], indent=2))


if __name__ == "__main__":
    main()
