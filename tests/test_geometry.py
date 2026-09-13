"""Geometry pipeline correctness checks. Framework-agnostic, matches test_smoke.py."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from visual_radar.geometry.projection import (
    depth_to_range,
    detection_to_world_point,
    focal_length_from_fov,
    pixel_to_camera_ray,
    sample_depth_for_bbox,
    to_ground_polar,
)
from visual_radar.geometry.rotations import rotation_matrix
from visual_radar.sources import Frame

INTRINSICS = {
    "width": 1920,
    "height": 1080,
    "fov_h_deg": 90.0,
    "fov_v_deg": None,
    "principal_point": None,
}


def assert_close(a: float, b: float, tol: float = 1e-6, msg: str = "") -> None:
    assert abs(a - b) < tol, f"{msg}: {a} != {b}"


def test_rotation_matrix_identity() -> None:
    R = rotation_matrix(0.0, 0.0, 0.0)
    world = R @ np.array([0.0, 0.0, 1.0])
    assert_close(world[0], 0.0, msg="identity x")
    assert_close(world[1], 1.0, msg="identity y")
    assert_close(world[2], 0.0, msg="identity z")


def test_rotation_matrix_is_proper() -> None:
    R = rotation_matrix(0.3, -0.2, 0.5)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert_close(float(np.linalg.det(R)), 1.0, msg="det(R)")


def test_rotation_matrix_yaw_90() -> None:
    R = rotation_matrix(math.pi / 2, 0.0, 0.0)
    world = R @ np.array([0.0, 0.0, 1.0])
    assert_close(world[0], 1.0, msg="yaw90 x")
    assert_close(world[1], 0.0, msg="yaw90 y")
    assert_close(world[2], 0.0, msg="yaw90 z")


def test_rotation_matrix_pitch_90() -> None:
    R = rotation_matrix(0.0, math.pi / 2, 0.0)
    world = R @ np.array([0.0, 0.0, 1.0])
    assert_close(world[0], 0.0, msg="pitch90 x")
    assert_close(world[1], 0.0, msg="pitch90 y")
    assert_close(world[2], 1.0, msg="pitch90 z")


def test_rotation_matrix_unsupported_convention() -> None:
    try:
        rotation_matrix(0.0, 0.0, 0.0, convention="android")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unsupported convention")


def test_focal_length_from_fov() -> None:
    fx = focal_length_from_fov(1920, 90.0)
    assert_close(fx, 960.0, tol=1e-3, msg="fx")


def test_pixel_to_camera_ray_center() -> None:
    ray = pixel_to_camera_ray(960.0, 540.0, INTRINSICS)
    assert_close(ray[0], 0.0, msg="center ray x")
    assert_close(ray[1], 0.0, msg="center ray y")
    assert_close(ray[2], 1.0, msg="center ray z")


def test_pixel_to_camera_ray_off_center() -> None:
    ray = pixel_to_camera_ray(1920.0, 540.0, INTRINSICS)
    assert ray[0] > 0, "pixel to the right of center should give positive x_cam"
    assert_close(ray[1], 0.0, msg="off-center ray y")
    assert_close(ray[2], 1.0, msg="off-center ray z")


def test_depth_to_range_center() -> None:
    r = depth_to_range(5.0, np.array([0.0, 0.0, 1.0]))
    assert_close(r, 5.0, msg="center-pixel range")


def test_depth_to_range_off_center() -> None:
    r = depth_to_range(5.0, np.array([1.0, 0.0, 1.0]))
    assert_close(r, 5.0 * math.sqrt(2.0), msg="off-center range")


def test_to_ground_polar_forward() -> None:
    azimuth, range_2d = to_ground_polar(np.array([0.0, 10.0, 3.0]))
    assert_close(azimuth, 0.0, msg="forward azimuth")
    assert_close(range_2d, 10.0, msg="forward range_2d")


def test_to_ground_polar_right() -> None:
    azimuth, range_2d = to_ground_polar(np.array([10.0, 0.0, 0.0]))
    assert_close(azimuth, math.pi / 2, msg="right azimuth")
    assert_close(range_2d, 10.0, msg="right range_2d")


def test_sample_depth_for_bbox_median() -> None:
    depth_map = np.zeros((10, 10))
    depth_map[2:5, 2:5] = 7.0
    bbox = torch.tensor([2.0, 2.0, 5.0, 5.0])
    assert_close(sample_depth_for_bbox(depth_map, bbox), 7.0, msg="bbox median")


def test_sample_depth_for_bbox_clamps_out_of_bounds() -> None:
    depth_map = np.full((10, 10), 3.0)
    bbox = torch.tensor([-50.0, -50.0, 200.0, 200.0])
    assert_close(sample_depth_for_bbox(depth_map, bbox), 3.0, msg="clamped bbox median")


def test_detection_to_world_point_forward() -> None:
    frame = Frame(image=torch.zeros(3, 10, 10), yaw=0.0, pitch=0.0)
    bbox = torch.tensor([860.0, 440.0, 1060.0, 640.0])
    depth_map = np.full((1080, 1920), 20.0)

    world_point = detection_to_world_point(bbox, depth_map, frame, INTRINSICS)
    assert_close(world_point[0], 0.0, tol=1e-3, msg="forward world x")
    assert_close(world_point[1], 20.0, tol=1e-3, msg="forward world y")


def test_detection_to_world_point_yaw_90() -> None:
    frame = Frame(image=torch.zeros(3, 10, 10), yaw=math.pi / 2, pitch=0.0)
    bbox = torch.tensor([860.0, 440.0, 1060.0, 640.0])
    depth_map = np.full((1080, 1920), 20.0)

    world_point = detection_to_world_point(bbox, depth_map, frame, INTRINSICS)
    assert_close(world_point[0], 20.0, tol=1e-3, msg="yaw90 world x")
    assert_close(world_point[1], 0.0, tol=1e-3, msg="yaw90 world y")


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
