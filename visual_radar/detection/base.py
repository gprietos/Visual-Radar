"""Detection output shapes and the pluggable object detector contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ..sources import Frame


@dataclass
class Detection:
    bbox_xyxy: torch.Tensor
    confidence: float
    class_id: int
    class_name: str
    mask: Optional[torch.Tensor] = None


@dataclass
class DetectionOutput:
    detections: List[Detection]
    image_shape: tuple


class BaseDetector(ABC):
    """Adapter over a pretrained detection model (black box, not trained here)."""

    @abstractmethod
    def detect(self, frame: Frame) -> DetectionOutput:
        """Run detection on a single frame's image."""
        ...

    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]:
        """Class-id -> class-name taxonomy sourced from the underlying model.

        Downstream code must read classes from here, never from a hardcoded list.
        """
        ...
