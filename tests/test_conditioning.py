import math

import pytest
import torch

from lift_classify import sample_depth_coordinates, uniform_depth_coordinates
from lift_classify.conditioning import DepthEmbedding, prepare_depth_coordinates


def test_continuous_depth_embedding_formula_and_gradient():
    embedding = DepthEmbedding(16)
    depths = torch.tensor([0.1234, 0.1235, 0.75], requires_grad=True)
    frequencies = 10000.0 ** (-torch.arange(128, dtype=torch.float32) / 128)
    phases = depths[:, None] * frequencies
    expected = embedding.mlp(torch.cat((phases.cos(), phases.sin()), dim=-1))
    result = embedding(depths)
    torch.testing.assert_close(result, expected)
    assert not torch.equal(result[0], result[1])
    result.sum().backward()
    assert depths.grad.abs().sum() > 0
    assert all(p.grad is not None for p in embedding.parameters())


@pytest.mark.parametrize("loops", [1, 2, 4, 8])
def test_uniform_and_random_grids(loops):
    uniform = uniform_depth_coordinates(loops)
    torch.testing.assert_close(uniform, torch.tensor([i / loops for i in range(1, loops + 1)]))
    generator = torch.Generator().manual_seed(17)
    random = sample_depth_coordinates(3, loops, generator=generator)
    generator.manual_seed(17)
    raw = torch.rand(3, loops - 1, dtype=torch.float64, generator=generator).sort(-1).values
    expected = (1 - loops * 1e-6) * raw + torch.arange(1, loops, dtype=torch.float64) * 1e-6
    expected = torch.cat((expected, torch.ones(3, 1, dtype=torch.float64)), -1).float()
    torch.testing.assert_close(random, expected, rtol=0, atol=0)
    assert random.dtype == torch.float32 and random.shape == (3, loops)
    assert (random > 0).all() and (random[:, -1] == 1).all()
    assert (random[:, 1:] > random[:, :-1]).all()
    if loops > 1:
        assert not torch.equal(random[0], random[1])


def test_prepare_shared_and_per_example_depths():
    shared = prepare_depth_coordinates(torch.tensor([0.2, 1.0]), 3, 2, torch.device("cpu"))
    assert shared.shape == (3, 2)
    supplied = torch.tensor([[0.2, 1], [0.8, 1], [0.9, 1]])
    torch.testing.assert_close(prepare_depth_coordinates(supplied, 3, 2, torch.device("cpu")), supplied)


@pytest.mark.parametrize("coordinates", [
    torch.tensor([0.0, 1.0]), torch.tensor([-0.1, 1.0]), torch.tensor([1.1, 1.0]),
    torch.tensor([1.0, 1.0]), torch.tensor([0.9, 0.8]), torch.tensor([0.1, 0.9]),
    torch.tensor([math.nan, 1.0]), torch.tensor([math.inf, 1.0]),
    torch.tensor([[0.2, 1.0]]), torch.ones(2, 3), torch.tensor([0, 1]), [0.5, 1.0],
])
def test_invalid_coordinates(coordinates):
    with pytest.raises(ValueError, match="depth_coordinates"):
        prepare_depth_coordinates(coordinates, 2, 2, torch.device("cpu"))


@pytest.mark.parametrize("loops", [0, -1, 2.5, True])
def test_invalid_loop_counts(loops):
    with pytest.raises(ValueError):
        uniform_depth_coordinates(loops)
    with pytest.raises(ValueError):
        sample_depth_coordinates(2, loops)


def test_random_grid_limit():
    with pytest.raises(ValueError, match="spacing correction"):
        sample_depth_coordinates(1, 1_000_000)


def test_random_helper_samples_on_cpu_even_with_a_different_default_device():
    with torch.device("meta"):
        coordinates = sample_depth_coordinates(2, 4)
    assert coordinates.device.type == "cpu"
    assert torch.isfinite(coordinates).all()
