"""Native PyTorch attention with FP32 RMS statistics and axial 2D RoPE."""

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, affine: bool = False, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x: Tensor) -> Tensor:
        # FP32 statistics avoid squaring overflow in half-precision states.
        values = x.float()
        normalized = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        if self.weight is not None:
            normalized = normalized * self.weight.float()
        return normalized.to(x.dtype)


class RotaryEmbedding2D(nn.Module):
    """Assumed axial convention: H then W, adjacent channel pairs, integer grid."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 4:
            raise ValueError("head_dim must be positive and divisible by 4")
        self.head_dim = head_dim
        self.theta = theta

    def forward(self, x: Tensor, grid_shape: tuple[int, int]) -> Tensor:
        height, width = grid_shape
        if height <= 0 or width <= 0 or x.shape[-2:] != (height * width, self.head_dim):
            raise ValueError("RoPE tokens/head dimension must match the spatial grid")
        quarter_dim = self.head_dim // 4
        frequencies = torch.exp(
            -math.log(self.theta) * torch.arange(quarter_dim, device=x.device, dtype=torch.float32)
            / quarter_dim
        )
        y = torch.arange(height, device=x.device, dtype=torch.float32).repeat_interleave(width)
        x_position = torch.arange(width, device=x.device, dtype=torch.float32).repeat(height)
        angles = torch.cat((y[:, None] * frequencies, x_position[:, None] * frequencies), dim=-1)
        pairs = x.float().reshape(*x.shape[:-1], self.head_dim // 2, 2)
        even, odd = pairs.unbind(dim=-1)
        cos, sin = angles.cos(), angles.sin()
        rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
        return rotated.flatten(-2).to(x.dtype)


def initialize_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0,
                 attention_dropout: float = 0.0):
        super().__init__()
        if num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by a positive num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.q_norm = RMSNorm(self.head_dim, affine=True)
        self.k_norm = RMSNorm(self.head_dim, affine=True)
        self.rope = RotaryEmbedding2D(self.head_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.attention_dropout = attention_dropout
        self.apply(initialize_linear)

    def forward(self, x: Tensor, grid_shape: tuple[int, int]) -> Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = self.rope(self.q_norm(q), grid_shape)
        k = self.rope(self.k_norm(k), grid_shape)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj_dropout(self.proj(attended))


class MLP(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.activation = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.drop2 = nn.Dropout(dropout)
        self.apply(initialize_linear)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop2(self.fc2(self.drop1(self.activation(self.fc1(x)))))
