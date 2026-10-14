import pytest
import torch

from lift_classify import LiFTConfig


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


@pytest.fixture
def config():
    return LiFTConfig(image_size=16, patch_size=4, hidden_dim=32, num_heads=4, num_classes=5)


def activate_modulation(model):
    """Exercise actual branches without waiting for zero gates to learn."""
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "modulation"):
                torch.nn.init.normal_(module.modulation[-1].weight, std=0.1)
                torch.nn.init.normal_(module.modulation[-1].bias, std=0.1)
