"""A configurable, trainable classification adaptation of the LiFT architecture."""

from .conditioning import sample_depth_coordinates, uniform_depth_coordinates
from .config import LiFTConfig, paper_reference_config, small_config, tiny_config
from .model import LiFTClassifier, LiFTOutput

__all__ = [
    "LiFTClassifier", "LiFTConfig", "LiFTOutput", "paper_reference_config",
    "sample_depth_coordinates", "small_config", "tiny_config", "uniform_depth_coordinates",
]
