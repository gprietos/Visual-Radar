"""Per-frame input shape.

Frame ingestion itself (recorded 360 video, phone capture, eventually
live) is external to this project. This module just holds the `Frame`
dataclass every building block consumes; an iterable of frames is typed
inline as `Iterable[Frame]` wherever needed, with no dedicated alias or
ingestor class.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Frame:
    image: torch.Tensor
    yaw: float
    pitch: float
    roll: float = 0.0
    timestamp: Optional[float] = None
