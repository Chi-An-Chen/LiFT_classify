"""Continuous depth embeddings and opt-in paper-style random coordinates."""

import math

import torch
from torch import Tensor, nn

from .config import validate_integer


class DepthEmbedding(nn.Module):
    frequency_dim = 256

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(self.frequency_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        for layer in (self.mlp[0], self.mlp[2]):
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.zeros_(layer.bias)

    def forward(self, coordinates: Tensor) -> Tensor:
        if coordinates.ndim != 1:
            raise ValueError("DepthEmbedding expects one coordinate per example, shape [B]")
        half_dim = self.frequency_dim // 2
        frequencies = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=coordinates.device, dtype=torch.float32)
            / half_dim
        )
        angles = coordinates.float()[:, None] * frequencies
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


def uniform_depth_coordinates(num_loops: int, *, device=None) -> Tensor:
    """Return [K] FP32 coordinates, including 1 but not the prelude coordinate 0."""
    validate_integer(num_loops, "num_loops")
    return torch.arange(1, num_loops + 1, device=device, dtype=torch.float32) / num_loops


def sample_depth_coordinates(batch_size: int, num_loops: int, *, device=None,
                             generator: torch.Generator | None = None) -> Tensor:
    """D.3 spacing correction, sampled on CPU in FP64 and returned as [B,K] FP32.

    The optional generator must be a CPU generator, including for MPS/CUDA output.
    """
    validate_integer(batch_size, "batch_size")
    validate_integer(num_loops, "num_loops")
    epsilon = 1e-6
    if num_loops * epsilon >= 1:
        raise ValueError("num_loops must be < 1,000,000 for the paper's spacing correction")
    interior = torch.rand(batch_size, num_loops - 1, device="cpu", dtype=torch.float64, generator=generator)
    interior = interior.sort(dim=-1).values
    offsets = torch.arange(1, num_loops, device="cpu", dtype=torch.float64) * epsilon
    interior = (1 - num_loops * epsilon) * interior + offsets
    return torch.cat((interior, torch.ones(batch_size, 1, device="cpu", dtype=torch.float64)), dim=-1).to(
        device=device, dtype=torch.float32
    )


def prepare_depth_coordinates(coordinates: Tensor | None, batch_size: int,
                              num_loops: int, device: torch.device) -> Tensor:
    if coordinates is None:
        return uniform_depth_coordinates(num_loops, device=device).expand(batch_size, -1)
    if not isinstance(coordinates, Tensor) or not coordinates.is_floating_point():
        raise ValueError("depth_coordinates must be a floating-point tensor")
    coordinates = coordinates.to(device=device, dtype=torch.float32)
    if coordinates.shape == (num_loops,):
        coordinates = coordinates.expand(batch_size, -1)
    if coordinates.shape != (batch_size, num_loops):
        raise ValueError("depth_coordinates must have shape [K] or [B,K]")
    if (not torch.isfinite(coordinates).all() or (coordinates <= 0).any()
            or (coordinates > 1).any() or (coordinates[:, -1] != 1).any()
            or (coordinates[:, 1:] <= coordinates[:, :-1]).any()):
        raise ValueError("depth_coordinates must be finite, strictly increasing in (0,1], and end at 1")
    return coordinates
