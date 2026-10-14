import torch
import torch.nn.functional as F

from lift_classify.blocks import LiFTBlock
from conftest import activate_modulation


def test_zero_initialization_is_identity(config):
    block = LiFTBlock(config)
    x = torch.randn(2, 6, config.hidden_dim)
    torch.testing.assert_close(block(x, torch.randn(2, config.hidden_dim), (2, 3)), x, rtol=0, atol=0)
    assert torch.count_nonzero(block.modulation[-1].weight) == 0
    assert torch.count_nonzero(block.modulation[-1].bias) == 0
    assert block.norm1.weight is None and block.norm2.weight is None
    assert torch.count_nonzero(block.attn.qkv.bias) == 0


def test_block_formula_uses_updated_attention_residual(config):
    block = LiFTBlock(config).eval()
    activate_modulation(block)
    x = torch.randn(2, 6, config.hidden_dim)
    condition = torch.randn(2, config.hidden_dim)
    parts = F.linear(F.silu(condition), block.modulation[-1].weight, block.modulation[-1].bias)
    shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = [v[:, None] for v in parts.chunk(6, -1)]
    norm = lambda values: values / torch.sqrt(values.square().mean(-1, keepdim=True) + 1e-6)
    updated = x + gate_a * block.attn(norm(x) * (1 + scale_a) + shift_a, (2, 3))
    expected = updated + gate_m * block.mlp(norm(updated) * (1 + scale_m) + shift_m)
    torch.testing.assert_close(block(x, condition, (2, 3)), expected)
    wrong_order = updated + gate_m * block.mlp(norm(x) * (1 + scale_m) + shift_m)
    assert not torch.allclose(expected, wrong_order)


def test_condition_reaches_attention_and_mlp(config):
    block = LiFTBlock(config)
    activate_modulation(block)
    x = torch.randn(2, 6, config.hidden_dim, requires_grad=True)
    condition = torch.randn(2, config.hidden_dim, requires_grad=True)
    block(x, condition, (2, 3)).square().mean().backward()
    assert condition.grad.abs().sum() > 0
    assert block.attn.qkv.weight.grad.abs().sum() > 0
    assert block.mlp.fc1.weight.grad.abs().sum() > 0
