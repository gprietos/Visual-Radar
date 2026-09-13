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

from ..sources import Frame
from .rotations import rotation_matrix


def focal_length_from_fov(width: int, fov_h_deg: float) -> float:
    """fx = width / (2*tan(FOV_h/2)) for uncalibrated intrinsics."""
    return width / (2.0 * np.tan(np.radians(fov_h_deg) / 2.0))


def pixel_to_camera_ray(u: float, v: float, intrinsics: dict) -> np.ndarray:
    """Pinhole model: x_cam=(u-cx)/fx, y_cam=(v-cy)/fy, z_cam=1.

    Not normalized — `depth_to_range` relies on this exact scale. `fy`
    falls back to `fx` (square pixels) when `fov_v_deg` is unset; the
    principal point falls back to the image center when unset.
    """
    width = intrinsics["width"]
    height = intrinsics["height"]

    fx = focal_length_from_fov(width, intrinsics["fov_h_deg"])
    fov_v_deg = intrinsics.get("fov_v_deg")
    fy = focal_length_from_fov(height, fov_v_deg) if fov_v_deg else fx

    principal_point = intrinsics.get("principal_point")
    cx, cy = principal_point if principal_point else (width / 2.0, height / 2.0)

    return np.array([(u - cx) / fx, (v - cy) / fy, 1.0])


def camera_ray_to_world(ray_cam: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    """d_world = R(yaw, pitch, roll) . d_cam.

    Direction only, same scale as `ray_cam` — not normalized, not scaled
    by range.
    """
    return rotation_matrix(yaw, pitch, roll) @ ray_cam


def depth_to_range(z_depth: float, ray_cam: np.ndarray) -> float:
    """range = z_depth * sqrt(x_cam^2 + y_cam^2 + 1); raw depth only equals range at image center."""
    return z_depth * float(np.linalg.norm(ray_cam))


def to_ground_polar(world_point: np.ndarray) -> tuple:
    """(azimuth_rad, range_2d_m) = (atan2(x_world, y_world), hypot(x_world, y_world)).

    Ignores z_world (height) — ground-projected range, not full 3D range.
    """
    x_world, y_world = world_point[0], world_point[1]
    return float(np.arctan2(x_world, y_world)), float(np.hypot(x_world, y_world))


def sample_depth_for_bbox(depth_map: np.ndarray, bbox_xyxy: torch.Tensor) -> float:
    """Median depth over the bbox region, not a single center pixel.

    Fixes the original demo's bug of sampling depth at one center pixel.
    `bbox_xyxy` is clamped to `depth_map`'s bounds (minimum 1x1 region).
    """
    height, width = depth_map.shape[-2:]
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)

    x1_i = min(max(int(x1), 0), width - 1)
    y1_i = min(max(int(y1), 0), height - 1)
    x2_i = min(max(int(np.ceil(x2)), x1_i + 1), width)
    y2_i = min(max(int(np.ceil(y2)), y1_i + 1), height)

    region = depth_map[y1_i:y2_i, x1_i:x2_i]
    return float(np.median(region))


def detection_to_world_point(bbox_xyxy: torch.Tensor, depth_map: np.ndarray, frame: Frame, intrinsics: dict) -> np.ndarray:
    """Compose pixel->ray->world->range-corrected into one world-space point for a single box."""
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)

    ray_cam = pixel_to_camera_ray(cx, cy, intrinsics)
    z_depth = sample_depth_for_bbox(depth_map, bbox_xyxy)
    range_m = depth_to_range(z_depth, ray_cam)

    world_dir = camera_ray_to_world(ray_cam, frame.yaw, frame.pitch, frame.roll)
    world_dir = world_dir / np.linalg.norm(world_dir)

    return range_m * world_dir
