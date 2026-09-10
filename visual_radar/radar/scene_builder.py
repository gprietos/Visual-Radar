"""Detection + depth + pose -> RadarScene."""

from ..depth.base import DepthOutput
from ..detection.base import DetectionOutput
from ..sources import Frame
from .base import RadarScene


def build_radar_scene(
    detection_output: DetectionOutput, depth_output: DepthOutput, frame: Frame, intrinsics: dict
) -> RadarScene:
    """Build a fresh RadarScene for this frame.

    `intrinsics` is the `camera:` section of the loaded config. Must always
    return a new `RadarScene(points=[], ...)` when there are no detections
    rather than reusing a prior frame's scene — the original demo's
    `cartesian_detections` staleness bug must not be reintroduced.
    """
    raise NotImplementedError
