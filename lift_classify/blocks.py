"""SpeedrunDiT-style blocks without value residuals or token routing."""

from torch import Tensor, nn

from .config import LiFTConfig
from .layers import Attention, MLP, RMSNorm, modulate


class LiFTBlock(nn.Module):
    def __init__(self, config: LiFTConfig):
        super().__init__()
        dim = config.hidden_dim
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, config.num_heads, config.dropout, config.attention_dropout)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, int(dim * config.mlp_ratio), config.dropout)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: Tensor, condition: Tensor, grid_shape: tuple[int, int]) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(condition).chunk(6, -1)
        x = x + gate_a[:, None, :] * self.attn(modulate(self.norm1(x), shift_a, scale_a), grid_shape)
        x = x + gate_m[:, None, :] * self.mlp(modulate(self.norm2(x), shift_m, scale_m))
        return x


class TransformerStack(nn.Module):
    """A stack owns unique blocks; reusing the stack shares all of their parameters."""

    def __init__(self, config: LiFTConfig, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList(LiFTBlock(config) for _ in range(depth))

    def forward(self, x: Tensor, condition: Tensor, grid_shape: tuple[int, int]) -> Tensor:
        for block in self.blocks:
            x = block(x, condition, grid_shape)
        return x
