"""End-to-end wiring example: detector + depth + both renderers, inline loop.

No pipeline class — building blocks are combined directly here, the same
way DESIGN.md's original demo did, just decomposed and fixed. This script
does not implement frame ingestion: it expects a caller-supplied
`Iterable[Frame]` — that's the intentional seam where an external ingestion
project (recorded video, phone capture, live feed) plugs in.
"""

import argparse
from typing import Iterable

from visual_radar.camera_view.renderer import CameraViewRenderer
from visual_radar.config import load_config
from visual_radar.depth.metric_depth import MetricDepthEstimator
from visual_radar.detection.yolo import YoloDetector
from visual_radar.radar.opencv_renderer import OpenCVRadarRenderer
from visual_radar.radar.scene_builder import build_radar_scene
from visual_radar.sources import Frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Visual Radar building blocks over an external frame iterable.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load config
    config = load_config(args.config)

    detector = YoloDetector(
        weights_path=config["detection"]["weights_path"],
        confidence_threshold=config["detection"]["confidence_threshold"],
        device=config["detection"]["device"],
    )
    depth_estimator = MetricDepthEstimator(
        checkpoint_path=config["depth"]["checkpoint_path"],
        device=config["depth"]["device"],
    )
    radar_renderer = OpenCVRadarRenderer()
    camera_view_renderer = CameraViewRenderer()

    intrinsics = config["camera"]
    radar_view_config = config["radar_view"]

    # TODO: replace with a real Iterable[Frame] supplied by an ingestion project.
    frame_source: Iterable[Frame] = []

    for frame in frame_source:
        detection_output = detector.detect(frame)
        depth_output = depth_estimator.estimate(frame)
        scene = build_radar_scene(detection_output, depth_output, frame, intrinsics)

        camera_view = camera_view_renderer.render(frame, detection_output, detector.class_names)
        radar_view = radar_renderer.render(scene, radar_view_config)

        # TODO: write/show camera_view and radar_view as two separate outputs
        # (e.g. under config["output"]["output_dir"]).


if __name__ == "__main__":
    main()
