"""Pixel + pose + depth -> world azimuth/range.

The non-obvious core of the project:
1. Pixel -> camera-space ray (pinhole model).
2. Rotate into world space via R(yaw, pitch, roll) (see `rotations.py`).
3. Correct raw z-depth to true range along the ray.
4. Project to the ground plane: azimuth = atan2(x_world, y_world), range_2d = hypot(x_world, y_world).

`intrinsics` throughout is the `camera:` section of the loaded config
(dict with keys `width`, `height`, `fov_h_deg`, `fov_v_deg`,
`principal_point`), not a typed class.
"""

import numpy as np
import torch

from ..depth.base import DepthOutput
from ..detection.base import Detection
from ..sources import Frame
from .rotations import rotation_matrix


def focal_length_from_fov(width: int, fov_h_deg: float) -> float:
    """fx = width / (2*tan(FOV_h/2)) for uncalibrated intrinsics."""
    raise NotImplementedError


def pixel_to_camera_ray(u: float, v: float, intrinsics: dict) -> np.ndarray:
    """Pinhole model: x_cam=(u-cx)/fx, y_cam=(v-cy)/fy, z_cam=1."""
    raise NotImplementedError


def camera_ray_to_world(ray_cam: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """d_world = R(yaw, pitch, roll) . d_cam."""
    raise NotImplementedError


def depth_to_range(z_depth: float, ray_cam: np.ndarray) -> float:
    """range = z_depth * sqrt(x_cam^2 + y_cam^2 + 1); raw depth only equals range at image center."""
    raise NotImplementedError


def to_ground_polar(world_point: np.ndarray) -> tuple:
    """(azimuth_rad, range_2d_m) = (atan2(x_world, y_world), hypot(x_world, y_world))."""
    raise NotImplementedError


def sample_depth_for_bbox(depth_map: torch.Tensor, bbox_xyxy: torch.Tensor) -> float:
    """Median depth over the bbox region, not a single center pixel.

    Fixes the original demo's bug of sampling depth at one center pixel.
    """
    raise NotImplementedError


def detection_to_world_point(
    detection: Detection, depth_output: DepthOutput, frame: Frame, intrinsics: dict
) -> np.ndarray:
    """Compose pixel->ray->world->range-corrected into one world-space point per detection."""
    raise NotImplementedError
