"""Metric depth estimator adapter (e.g. Depth Anything metric / ZoeDepth-style checkpoints)."""

from ..sources import Frame
from .base import BaseDepthEstimator, DepthOutput


class MetricDepthEstimator(BaseDepthEstimator):
    """Wraps a pretrained metric-depth checkpoint as a `BaseDepthEstimator`."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device

    def estimate(self, frame: Frame) -> DepthOutput:
        raise NotImplementedError
