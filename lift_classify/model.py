"""LiFT recurrence adapted to classification; labels never enter the model."""

from typing import TypedDict

import torch
from torch import Tensor, nn

from .blocks import TransformerStack
from .conditioning import DepthEmbedding, prepare_depth_coordinates
from .config import LiFTConfig, validate_integer
from .layers import RMSNorm, initialize_linear, modulate


class LiFTOutput(TypedDict):
    logits: Tensor
    hidden_states: tuple[Tensor, ...] | None
    intermediate_logits: tuple[Tensor, ...] | None
    depth_coordinates: Tensor


class ClassificationReadout(nn.Module):
    """Prediction-only coda: modulated token norm, GAP, then a shared classifier."""

    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.norm = RMSNorm(hidden_dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        # A nonzero classifier lets CE reach the backbone on the first backward.
        initialize_linear(self.classifier)

    def forward(self, hidden_state: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.modulation(condition).chunk(2, -1)
        features = modulate(self.norm(hidden_state), shift, scale).mean(dim=1)
        return self.classifier(features)


class LiFTClassifier(nn.Module):
    def __init__(self, config: LiFTConfig | None = None):
        super().__init__()
        self.config = config if config is not None else LiFTConfig()
        config = self.config
        self.patch_embedding = nn.Conv2d(config.in_channels, config.hidden_dim,
                                         kernel_size=config.patch_size, stride=config.patch_size)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        nn.init.zeros_(self.patch_embedding.bias)
        self.base_condition = nn.Parameter(torch.empty(config.hidden_dim))
        nn.init.normal_(self.base_condition, std=0.02)
        self.prelude = TransformerStack(config, config.prelude_depth)
        self.core = TransformerStack(config, config.core_depth)
        self.boundary_norm = RMSNorm(config.hidden_dim)
        self.depth_embedding = DepthEmbedding(config.hidden_dim)
        self.readout = ClassificationReadout(config.hidden_dim, config.num_classes)

    def classify_features(self, hidden_state: Tensor) -> Tensor:
        """Apply the same readout to a returned state, preserving its autograd graph."""
        if hidden_state.ndim != 3 or hidden_state.shape[-1] != self.config.hidden_dim:
            raise ValueError("hidden_state must have shape [B,N,hidden_dim]")
        condition = self.base_condition.expand(hidden_state.shape[0], -1)
        return self.readout(hidden_state, condition)

    def forward(self, images: Tensor, num_loops: int | None = None, *,
                return_intermediates: bool = False, return_intermediate_logits: bool = False,
                depth_coordinates: Tensor | None = None) -> Tensor | LiFTOutput:
        if images.ndim != 4 or images.shape[1] != self.config.in_channels:
            raise ValueError(f"images must have shape [B,{self.config.in_channels},H,W]")
        if not images.is_floating_point():
            raise ValueError("images must be floating-point tensors")
        batch, _, height, width = images.shape
        patch_h, patch_w = self.config.patch_size
        if batch < 1 or height < 1 or width < 1 or height % patch_h or width % patch_w:
            raise ValueError("images need a positive batch and spatial sizes divisible by patch_size")
        loops = self.config.num_loops if num_loops is None else num_loops
        validate_integer(loops, "num_loops")
        coordinates = prepare_depth_coordinates(depth_coordinates, batch, loops, images.device)
        patches = self.patch_embedding(images)
        grid_shape = (patches.shape[2], patches.shape[3])
        tokens = patches.flatten(2).transpose(1, 2)
        condition = self.base_condition.expand(batch, -1)
        h0 = self.prelude(tokens, condition, grid_shape)
        hidden = h0
        states = [h0] if return_intermediates else None
        logits_by_depth = [self.readout(h0, condition)] if return_intermediate_logits else None
        for step in range(loops):
            core_condition = condition + self.depth_embedding(coordinates[:, step])
            hidden = self.core(self.boundary_norm(hidden) + h0, core_condition, grid_shape)
            if states is not None:
                states.append(hidden)
            if logits_by_depth is not None:
                logits_by_depth.append(self.readout(hidden, condition))
        logits = logits_by_depth[-1] if logits_by_depth is not None else self.readout(hidden, condition)
        if not return_intermediates and not return_intermediate_logits:
            return logits
        return LiFTOutput(logits=logits, hidden_states=tuple(states) if states is not None else None,
                          intermediate_logits=tuple(logits_by_depth) if logits_by_depth is not None else None,
                          depth_coordinates=coordinates)
