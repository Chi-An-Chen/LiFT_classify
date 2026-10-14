from dataclasses import replace
import json
import subprocess
import sys

import pytest
import torch

from lift_classify import LiFTClassifier, LiFTConfig, paper_reference_config, small_config, tiny_config
from tools.analyze_model import benchmark_cpu, estimate_costs


@pytest.mark.parametrize("settings", [
    {"num_heads": 0}, {"hidden_dim": 31}, {"hidden_dim": 30, "num_heads": 3},
    {"num_loops": 0}, {"prelude_depth": -1}, {"core_depth": 0},
    {"image_size": 225}, {"patch_size": (0, 16)}, {"patch_size": (16,)},
    {"num_classes": 0}, {"hidden_dim": True}, {"mlp_ratio": float("nan")},
    {"mlp_ratio": float("inf")}, {"mlp_ratio": 0.0}, {"mlp_ratio": 0.0001},
    {"dropout": 1.0}, {"dropout": -0.1}, {"attention_dropout": float("nan")},
])
def test_invalid_config(settings):
    with pytest.raises(ValueError):
        LiFTConfig(**settings)


def test_lightweight_presets():
    assert tiny_config().hidden_dim == 192
    assert tiny_config(num_classes=7, num_loops=4).num_classes == 7
    assert small_config().hidden_dim == 384 and small_config().num_heads == 6
    assert small_config(hidden_dim=192, num_heads=3).hidden_dim == 192


@pytest.mark.parametrize("scale,width,heads,dense,choices", [
    ("B", 768, 12, 12, [1, 2, 4]),
    ("L", 1024, 16, 24, [1, 2, 4, 5, 10]),
    ("XL", 1152, 16, 28, [1, 2, 3, 4, 6, 8, 12]),
])
def test_paper_presets_match_tables(scale, width, heads, dense, choices):
    for core in choices:
        config = paper_reference_config(scale, core_depth=core)
        assert (config.hidden_dim, config.num_heads) == (width, heads)
        assert config.prelude_depth == 4
        assert config.prelude_depth + config.num_loops * config.core_depth == dense
        assert config.image_size == (32, 32) and config.patch_size == (2, 2)
        assert config.in_channels == 4
    assert paper_reference_config(scale).num_loops == 4


def test_invalid_paper_presets():
    with pytest.raises(ValueError, match="scale"):
        paper_reference_config("S")
    with pytest.raises(ValueError, match="core_depth"):
        paper_reference_config("B", core_depth=3)


@pytest.mark.parametrize("ratio", [2.5, 4.0])
@pytest.mark.parametrize("loops", [1, 2, 4, 8])
def test_estimated_parameters_match_real_model(config, ratio, loops):
    config = replace(config, mlp_ratio=ratio)
    model = LiFTClassifier(config)
    row = estimate_costs(config, num_loops=loops)[2]
    assert row["parameters"] == sum(p.numel() for p in model.parameters())
    assert row["unique_blocks"] == config.prelude_depth + config.core_depth
    assert row["executed_blocks"] == config.prelude_depth + loops * config.core_depth
    assert row["forward_flops"] > 0


def test_tiny_costs_and_same_block_comparison():
    config = tiny_config()
    previous_flops = 0
    for loops in (1, 2, 4, 8):
        rows = estimate_costs(config, loops)
        lift = rows[2]
        assert lift["parameters"] == 3_167_592
        assert lift["forward_flops"] > previous_flops
        previous_flops = lift["forward_flops"]
        assert rows[0]["executed_blocks"] == rows[1]["executed_blocks"] == lift["executed_blocks"]
        if loops >= 2:
            assert lift["parameters"] < rows[3]["parameters"]
    assert estimate_costs(config, 2)[2]["forward_flops"] == 1_278_802_944


def test_paper_size_construction_without_allocating_large_weights():
    with torch.device("meta"):
        for scale in ("B", "L", "XL"):
            config = paper_reference_config(scale)
            model = LiFTClassifier(config)
            assert sum(p.numel() for p in model.parameters()) == estimate_costs(config)[2]["parameters"]


def test_analysis_cli():
    result = subprocess.run([sys.executable, "-m", "tools.analyze_model", "--loops", "1", "4", "--json"],
                            check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert len(report["estimates"]) == 8
    assert report["estimates"][2]["parameters"] == report["estimates"][6]["parameters"]


def test_cpu_benchmark_records_context_and_restores_threads(config):
    previous = torch.get_num_threads()
    report = benchmark_cpu(LiFTClassifier(config), [1, 2], min_run_time=0.01)
    assert torch.get_num_threads() == previous
    assert report["device"] == "cpu" and report["batch_size"] == 1
    assert [row["num_loops"] for row in report["measurements"]] == [1, 2]
    assert all(row["median_ms"] > 0 and row["iqr_ms"] >= 0 for row in report["measurements"])
