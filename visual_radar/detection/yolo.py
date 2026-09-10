"""YOLO (ultralytics) detector adapter."""

from typing import Dict

from ..sources import Frame
from .base import BaseDetector, DetectionOutput


class YoloDetector(BaseDetector):
    """Wraps an `ultralytics.YOLO` pretrained model as a `BaseDetector`."""

    def __init__(self, weights_path: str, confidence_threshold: float = 0.25, device: str = "cuda"):
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.device = device

    def detect(self, frame: Frame) -> DetectionOutput:
        raise NotImplementedError

    @property
    def class_names(self) -> Dict[int, str]:
        raise NotImplementedError
