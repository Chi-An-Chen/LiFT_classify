import math

import pytest
import torch
import torch.nn.functional as F

from lift_classify.layers import Attention, RMSNorm, RotaryEmbedding2D


def reference_rope(x, grid_shape):
    height, width = grid_shape
    dim = x.shape[-1]
    result = x.clone()
    for row in range(height):
        for column in range(width):
            token = row * width + column
            for axis, coordinate in enumerate((row, column)):
                for pair in range(dim // 4):
                    channel = axis * dim // 2 + 2 * pair
                    angle = coordinate / (10000 ** (pair / (dim // 4)))
                    a, b = x[..., token, channel], x[..., token, channel + 1]
                    result[..., token, channel] = a * math.cos(angle) - b * math.sin(angle)
                    result[..., token, channel + 1] = a * math.sin(angle) + b * math.cos(angle)
    return result


@pytest.mark.parametrize("affine", [False, True])
@pytest.mark.parametrize("scale", [0.0, 1e-10, 1.0, 1e4])
def test_rmsnorm_stability_and_formula(affine, scale):
    norm = RMSNorm(8, affine=affine)
    if affine:
        norm.weight.data.copy_(torch.linspace(0.5, 1.5, 8))
    x = (torch.randn(2, 3, 8) * scale).requires_grad_()
    expected = x / torch.sqrt(x.square().mean(-1, keepdim=True) + 1e-6)
    if affine:
        expected = expected * norm.weight
    actual = norm(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert torch.isfinite(actual).all() and torch.isfinite(x.grad).all()


def test_half_statistics_do_not_overflow():
    x = torch.full((2, 8), 1000.0, dtype=torch.float16, requires_grad=True)
    result = RMSNorm(8)(x)
    torch.testing.assert_close(result, torch.ones_like(x))
    result.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("grid", [(2, 3), (3, 2), (1, 6)])
def test_axial_rope_matches_reference_and_preserves_norm(grid):
    x = torch.randn(2, 3, 6, 16)
    result = RotaryEmbedding2D(16)(x, grid)
    torch.testing.assert_close(result, reference_rope(x, grid))
    torch.testing.assert_close(result.square().sum(-1), x.square().sum(-1))
    torch.testing.assert_close(result[..., 0, :], x[..., 0, :])
    if grid[1] > 1:
        torch.testing.assert_close(result[..., 1, :8], x[..., 1, :8])


def test_rope_rejects_invalid_dimensions_and_grid():
    with pytest.raises(ValueError, match="divisible by 4"):
        RotaryEmbedding2D(6)
    with pytest.raises(ValueError, match="grid"):
        RotaryEmbedding2D(8)(torch.randn(1, 2, 5, 8), (2, 3))


def test_attention_matches_manual_noncausal_softmax():
    attn = Attention(16, 2).eval()
    attn.q_norm.weight.data.copy_(torch.linspace(0.4, 1.7, 8))
    attn.k_norm.weight.data.copy_(torch.linspace(1.4, 0.6, 8))
    x = torch.randn(2, 6, 16)
    qkv = F.linear(x, attn.qkv.weight, attn.qkv.bias).view(2, 6, 3, 2, 8)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q = q / torch.sqrt(q.square().mean(-1, keepdim=True) + 1e-6) * attn.q_norm.weight
    k = k / torch.sqrt(k.square().mean(-1, keepdim=True) + 1e-6) * attn.k_norm.weight
    q, k = reference_rope(q, (2, 3)), reference_rope(k, (2, 3))
    weights = (q @ k.transpose(-2, -1) / math.sqrt(8)).softmax(-1)
    expected = (weights @ v).transpose(1, 2).reshape(2, 6, 16)
    expected = F.linear(expected, attn.proj.weight, attn.proj.bias)
    torch.testing.assert_close(attn(x, (2, 3)), expected, rtol=1e-5, atol=1e-6)
    assert attn.q_norm.weight.shape == (8,)


def test_attention_dropout_is_disabled_in_eval():
    attn = Attention(16, 2, dropout=0.5, attention_dropout=0.5)
    x = torch.randn(2, 6, 16)
    attn.eval()
    torch.testing.assert_close(attn(x, (2, 3)), attn(x, (2, 3)), rtol=0, atol=0)
    attn.train()
    assert not torch.equal(attn(x, (2, 3)), attn(x, (2, 3)))
