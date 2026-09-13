"""Radar scene-building and rendering checks. Framework-agnostic, matches test_geometry.py."""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from visual_radar.detection.base import Detection
from visual_radar.radar.opencv_renderer import OpenCVRadarRenderer
from visual_radar.radar.palette import default_class_colors
from visual_radar.radar.scene_builder import build_radar_scene
from visual_radar.sources import Frame

INTRINSICS = {
    "width": 1920,
    "height": 1080,
    "fov_h_deg": 90.0,
    "fov_v_deg": None,
    "principal_point": None,
}

RADAR_VIEW_CONFIG = {
    "canvas_size": 200,
    "max_range_m": 100.0,
    "ring_interval_m": 20.0,
    "point_radius_px": 4,
    "show_legend": True,
    "show_labels": False,
    "background_image_path": None,
    "size_by": "fixed",
}

CLASSES = ["person", "car", "dog"]


def assert_close(a: float, b: float, tol: float = 1e-6, msg: str = "") -> None:
    assert abs(a - b) < tol, f"{msg}: {a} != {b}"


def make_detection(class_id: int = 0) -> Detection:
    return Detection(
        boxes=torch.tensor([[860.0, 440.0, 1060.0, 640.0]]),
        conf=torch.tensor([0.9]),
        label=torch.tensor([class_id], dtype=torch.int64),
    )


def make_empty_detection() -> Detection:
    return Detection(
        boxes=torch.zeros((0, 4)),
        conf=torch.zeros((0,)),
        label=torch.zeros((0,), dtype=torch.int64),
    )


def test_default_class_colors_order_independent() -> None:
    solo = default_class_colors([3])
    grouped = default_class_colors([0, 1, 2, 3])
    assert solo[3] == grouped[3], "class 3's color must not depend on what else is in the call"


def test_default_class_colors_distinct() -> None:
    colors = default_class_colors(range(10))
    assert len(set(colors.values())) == 10, "expected 10 distinct colors for 10 distinct class ids"


def test_build_radar_scene_empty_on_no_detections() -> None:
    frame = Frame(image=torch.zeros(3, 10, 10), yaw=0.5, pitch=0.0)
    detection = make_empty_detection()
    depth_map = np.zeros((1080, 1920))

    scene = build_radar_scene(detection, depth_map, frame, INTRINSICS, CLASSES)
    assert scene.points == []
    assert_close(scene.heading_rad, 0.5, msg="heading_rad")
    assert_close(scene.fov_deg, 90.0, msg="fov_deg")


def test_build_radar_scene_basic() -> None:
    frame = Frame(image=torch.zeros(3, 10, 10), yaw=0.0, pitch=0.0)
    detection = make_detection(class_id=1)
    depth_map = np.full((1080, 1920), 20.0)

    scene = build_radar_scene(detection, depth_map, frame, INTRINSICS, CLASSES)
    assert len(scene.points) == 1
    point = scene.points[0]

    assert_close(point.azimuth_rad, 0.0, tol=1e-3, msg="azimuth_rad")
    assert_close(point.range_m, 20.0, tol=1e-3, msg="range_m")
    assert point.class_id == 1
    assert point.class_name == "car"
    assert_close(point.detection_confidence, 0.9, msg="detection_confidence")
    assert point.depth_confidence is None


def test_build_radar_scene_multiple_detections() -> None:
    frame = Frame(image=torch.zeros(3, 10, 10), yaw=0.0, pitch=0.0)
    detection = Detection(
        boxes=torch.tensor([[860.0, 440.0, 1060.0, 640.0], [0.0, 440.0, 200.0, 640.0]]),
        conf=torch.tensor([0.9, 0.5]),
        label=torch.tensor([0, 2], dtype=torch.int64),
    )
    depth_map = np.full((1080, 1920), 20.0)

    scene = build_radar_scene(detection, depth_map, frame, INTRINSICS, CLASSES)
    assert len(scene.points) == 2
    assert scene.points[0].class_name == "person"
    assert scene.points[1].class_name == "dog"


def test_renderer_output_shape() -> None:
    frame = Frame(image=torch.zeros(3, 10, 10), yaw=0.0, pitch=0.0)
    detection = make_detection()
    depth_map = np.full((1080, 1920), 20.0)
    scene = build_radar_scene(detection, depth_map, frame, INTRINSICS, CLASSES)

    renderer = OpenCVRadarRenderer()
    canvas = renderer.render(scene, RADAR_VIEW_CONFIG)
    assert canvas.shape == (200, 200, 3)
    assert canvas.dtype == np.uint8


def test_renderer_ego_centric_invariant() -> None:
    renderer = OpenCVRadarRenderer()
    renderer._build_background(RADAR_VIEW_CONFIG)

    for heading_rad in (0.0, 1.2, -2.3):
        display_azimuth_rad = heading_rad - heading_rad  # a point dead-ahead: azimuth_rad == heading_rad
        px, py = renderer._polar_to_pixel(display_azimuth_rad, 10.0)
        assert_close(px, renderer._canvas_size / 2.0, tol=1.0, msg="ego-centric x")
        assert py < renderer._canvas_size / 2.0, "dead-ahead point should render above center (toward 'up')"


def test_renderer_show_labels_not_implemented() -> None:
    frame = Frame(image=torch.zeros(3, 10, 10), yaw=0.0, pitch=0.0)
    detection = make_empty_detection()
    depth_map = np.zeros((1080, 1920))
    scene = build_radar_scene(detection, depth_map, frame, INTRINSICS, CLASSES)

    config = dict(RADAR_VIEW_CONFIG, show_labels=True)
    renderer = OpenCVRadarRenderer()
    try:
        renderer.render(scene, config)
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError for show_labels=True")


def test_renderer_size_by_not_implemented() -> None:
    frame = Frame(image=torch.zeros(3, 10, 10), yaw=0.0, pitch=0.0)
    detection = make_empty_detection()
    depth_map = np.zeros((1080, 1920))
    scene = build_radar_scene(detection, depth_map, frame, INTRINSICS, CLASSES)

    config = dict(RADAR_VIEW_CONFIG, size_by="depth_confidence")
    renderer = OpenCVRadarRenderer()
    try:
        renderer.render(scene, config)
    except NotImplementedError:
        return
    raise AssertionError("expected NotImplementedError for size_by='depth_confidence'")


def test_renderer_handles_growing_class_set() -> None:
    frame = Frame(image=torch.zeros(3, 10, 10), yaw=0.0, pitch=0.0)
    depth_map = np.full((1080, 1920), 20.0)
    renderer = OpenCVRadarRenderer()

    detection_1 = make_detection(class_id=0)
    scene_1 = build_radar_scene(detection_1, depth_map, frame, INTRINSICS, CLASSES)
    renderer.render(scene_1, RADAR_VIEW_CONFIG)
    assert set(renderer._legend_entries.keys()) == {0}

    detection_2 = Detection(
        boxes=torch.tensor([[860.0, 440.0, 1060.0, 640.0], [0.0, 440.0, 200.0, 640.0]]),
        conf=torch.tensor([0.9, 0.5]),
        label=torch.tensor([0, 1], dtype=torch.int64),
    )
    scene_2 = build_radar_scene(detection_2, depth_map, frame, INTRINSICS, CLASSES)
    renderer.render(scene_2, RADAR_VIEW_CONFIG)
    assert set(renderer._legend_entries.keys()) == {0, 1}
    assert set(renderer._sprites.keys()) == {0, 1}


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
