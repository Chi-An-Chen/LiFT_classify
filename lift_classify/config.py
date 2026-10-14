"""Validated capacities and the paper's architecture-reference presets."""

from dataclasses import dataclass
import math
from numbers import Real


def validate_integer(value: int, name: str, minimum: int = 1) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def spatial_pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    pair = (value, value) if isinstance(value, int) else value
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError(f"{name} must be an integer or a (height, width) pair")
    for item in pair:
        validate_integer(item, name)
    return tuple(pair)


@dataclass(frozen=True)
class LiFTConfig:
    image_size: int | tuple[int, int] = 224
    in_channels: int = 3
    patch_size: int | tuple[int, int] = 16
    num_classes: int = 1000
    hidden_dim: int = 192
    num_heads: int = 3
    mlp_ratio: float = 4.0
    prelude_depth: int = 2
    core_depth: int = 2
    num_loops: int = 2
    dropout: float = 0.0
    attention_dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_size", spatial_pair(self.image_size, "image_size"))
        object.__setattr__(self, "patch_size", spatial_pair(self.patch_size, "patch_size"))
        for name in ("in_channels", "num_classes", "hidden_dim", "num_heads", "core_depth", "num_loops"):
            validate_integer(getattr(self, name), name)
        validate_integer(self.prelude_depth, "prelude_depth", minimum=0)
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if (self.hidden_dim // self.num_heads) % 4:
            raise ValueError("head dimension must be divisible by 4 for axial 2D RoPE")
        if any(size % patch for size, patch in zip(self.image_size, self.patch_size)):
            raise ValueError("image_size must be divisible by patch_size")
        if (not isinstance(self.mlp_ratio, Real) or isinstance(self.mlp_ratio, bool)
                or not math.isfinite(self.mlp_ratio) or self.mlp_ratio <= 0
                or not math.isfinite(self.hidden_dim * self.mlp_ratio)
                or int(self.hidden_dim * self.mlp_ratio) < 1):
            raise ValueError("mlp_ratio must give a positive finite MLP width")
        for name in ("dropout", "attention_dropout"):
            value = getattr(self, name)
            if not isinstance(value, Real) or isinstance(value, bool) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")


def tiny_config(**overrides) -> LiFTConfig:
    return LiFTConfig(**overrides)


def small_config(**overrides) -> LiFTConfig:
    return LiFTConfig(**({"hidden_dim": 384, "num_heads": 6} | overrides))


def paper_reference_config(scale: str, core_depth: int | None = None, **overrides) -> LiFTConfig:
    """Paper geometry and capacity, with a classification readout (not a generator).

    RGB adaptations should override image_size=224, in_channels=3, patch_size=16.
    """
    dimensions = {"B": (768, 12, 12, 2), "L": (1024, 16, 24, 5), "XL": (1152, 16, 28, 6)}
    core_choices = {"B": (1, 2, 4), "L": (1, 2, 4, 5, 10), "XL": (1, 2, 3, 4, 6, 8, 12)}
    if scale not in dimensions:
        raise ValueError("scale must be 'B', 'L', or 'XL'")
    width, heads, dense_depth, default_core = dimensions[scale]
    core_depth = default_core if core_depth is None else core_depth
    validate_integer(core_depth, "core_depth")
    if core_depth not in core_choices[scale]:
        raise ValueError(f"paper {scale} core_depth must be one of {core_choices[scale]}")
    settings = dict(image_size=32, in_channels=4, patch_size=2, hidden_dim=width,
                    num_heads=heads, prelude_depth=4, core_depth=core_depth,
                    num_loops=(dense_depth - 4) // core_depth)
    return LiFTConfig(**(settings | overrides))
