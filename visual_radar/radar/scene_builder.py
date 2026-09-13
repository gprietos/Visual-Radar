"""Detection + depth + pose -> RadarScene."""

from typing import List

import numpy as np

from ..detection.base import Detection
from ..geometry.projection import detection_to_world_point, to_ground_polar
from ..sources import Frame
from .base import RadarPoint, RadarScene


def build_radar_scene(
    detection: Detection, depth_map: np.ndarray, frame: Frame, intrinsics: dict, classes: List[str]
) -> RadarScene:
    """Build a fresh RadarScene for this frame.

    `intrinsics` is the `camera:` section of the loaded config. `classes`
    is the detector's own class-id -> name taxonomy. Always returns a new
    `RadarScene` (an empty `points` list when `detection` has no boxes)
    rather than reusing a prior frame's scene — the original demo's
    `cartesian_detections` staleness bug must not be reintroduced.
    """
    points = []
    for i in range(detection.boxes.shape[0]):
        world_point = detection_to_world_point(detection.boxes[i], depth_map, frame, intrinsics)
        azimuth_rad, range_m = to_ground_polar(world_point)
        class_id = int(detection.labels[i])

        points.append(
            RadarPoint(
                azimuth_rad=azimuth_rad,
                range_m=range_m,
                class_id=class_id,
                class_name=classes[class_id],
                detection_confidence=float(detection.confs[i]),
                depth_confidence=None,
            )
        )

    return RadarScene(points=points, fov_deg=intrinsics["fov_h_deg"], heading_rad=frame.yaw)
