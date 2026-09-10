"""Radar scene shapes and the radar rendering contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class RadarPoint:
    azimuth_rad: float
    range_m: float
    class_id: int
    class_name: str
    detection_confidence: float
    depth_confidence: Optional[float] = None
    track_id: Optional[int] = None  # reserved for v2+ tracking, unused in v1


@dataclass
class RadarScene:
    points: List[RadarPoint]
    fov_deg: float
    heading_rad: float = 0.0


class RadarRenderer(ABC):
    @abstractmethod
    def render(self, scene: RadarScene, config: dict) -> np.ndarray:
        """`config` is the `radar_view:` section of the loaded YAML config."""
        ...
