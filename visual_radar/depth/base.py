"""Depth output shape and the pluggable monocular depth estimator contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from ..sources import Frame


@dataclass
class DepthOutput:
    depth_map: torch.Tensor
    confidence_map: Optional[torch.Tensor] = None  # rarely populated in v1


class BaseDepthEstimator(ABC):
    """Adapter over a pretrained metric-depth model (black box, not trained here)."""

    @abstractmethod
    def estimate(self, frame: Frame) -> DepthOutput:
        """Run depth estimation on a single frame's image."""
        ...
