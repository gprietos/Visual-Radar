"""Camera-view rendering checks. Framework-agnostic, matches test_geometry.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from visual_radar.camera_view.renderer import CameraViewRenderer
from visual_radar.detection.base import Detection
from visual_radar.radar.palette import default_class_colors
from visual_radar.sources import Frame

CLASSES = ["person", "car"]


def make_detection(x1=20.0, y1=20.0, x2=60.0, y2=60.0, class_id=0, confidence=0.9) -> Detection:
    return Detection(
        boxes=torch.tensor([[x1, y1, x2, y2]]),
        conf=torch.tensor([confidence]),
        label=torch.tensor([class_id], dtype=torch.int64),
    )


def make_empty_detection() -> Detection:
    return Detection(
        boxes=torch.zeros((0, 4)),
        conf=torch.zeros((0,)),
        label=torch.zeros((0,), dtype=torch.int64),
    )


def test_render_output_shape_and_dtype() -> None:
    frame = Frame(image=torch.zeros(3, 100, 120), yaw=0.0, pitch=0.0)
    detection = make_empty_detection()

    renderer = CameraViewRenderer()
    canvas = renderer.render(frame, detection, CLASSES)
    assert canvas.shape == (100, 120, 3)
    assert canvas.dtype == np.uint8


def test_render_no_detections_leaves_background_untouched() -> None:
    frame = Frame(image=torch.full((3, 50, 50), 0.5), yaw=0.0, pitch=0.0)
    detection = make_empty_detection()

    renderer = CameraViewRenderer()
    canvas = renderer.render(frame, detection, CLASSES)
    expected_value = int(0.5 * 255)
    assert (canvas == expected_value).all()


def test_render_draws_box_in_class_color() -> None:
    frame = Frame(image=torch.zeros(3, 100, 100), yaw=0.0, pitch=0.0)
    detection = make_detection(x1=20.0, y1=20.0, x2=60.0, y2=60.0, class_id=0)

    renderer = CameraViewRenderer()
    canvas = renderer.render(frame, detection, CLASSES)

    r, g, b = default_class_colors([0])[0]
    top_edge_pixel = canvas[20, 40]
    assert tuple(int(c) for c in top_edge_pixel) == (b, g, r)


def test_render_respects_explicit_class_colors() -> None:
    frame = Frame(image=torch.zeros(3, 100, 100), yaw=0.0, pitch=0.0)
    detection = make_detection(x1=20.0, y1=20.0, x2=60.0, y2=60.0, class_id=0)

    renderer = CameraViewRenderer()
    canvas = renderer.render(frame, detection, CLASSES, class_colors={0: (10, 20, 30)})

    top_edge_pixel = canvas[20, 40]
    assert tuple(int(c) for c in top_edge_pixel) == (30, 20, 10)


def test_render_clips_out_of_bounds_bbox_without_crashing() -> None:
    frame = Frame(image=torch.zeros(3, 50, 50), yaw=0.0, pitch=0.0)
    detection = make_detection(x1=-30.0, y1=-30.0, x2=200.0, y2=200.0, class_id=1)

    renderer = CameraViewRenderer()
    canvas = renderer.render(frame, detection, CLASSES)
    assert canvas.shape == (50, 50, 3)


def test_render_handles_degenerate_bbox_without_crashing() -> None:
    frame = Frame(image=torch.zeros(3, 50, 50), yaw=0.0, pitch=0.0)
    detection = make_detection(x1=200.0, y1=200.0, x2=210.0, y2=210.0, class_id=0)

    renderer = CameraViewRenderer()
    canvas = renderer.render(frame, detection, CLASSES)
    assert canvas.shape == (50, 50, 3)


def test_render_multiple_detections() -> None:
    frame = Frame(image=torch.zeros(3, 100, 100), yaw=0.0, pitch=0.0)
    detection = Detection(
        boxes=torch.tensor([[10.0, 10.0, 40.0, 40.0], [50.0, 50.0, 90.0, 90.0]]),
        conf=torch.tensor([0.9, 0.7]),
        label=torch.tensor([0, 1], dtype=torch.int64),
    )

    renderer = CameraViewRenderer()
    canvas = renderer.render(frame, detection, CLASSES)
    assert canvas.shape == (100, 100, 3)


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
