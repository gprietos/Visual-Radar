"""End-to-end wiring example: detector + depth + both renderers, inline loop.

No pipeline class — building blocks are combined directly here, the same
way DESIGN.md's original demo did, just decomposed and fixed. This script
does not implement frame ingestion: it expects a caller-supplied
`Iterable[Frame]` — that's the intentional seam where an external ingestion
project (recorded video, phone capture, live feed) plugs in.

The detector and depth estimator are also external to this project (models
from a separate project, imported and instantiated here directly) — this
project only consumes their output shapes: a `visual_radar.detection.base.
Detection` (boxes/conf/label tensors) from the detector's `forward(frame)`,
a plain `np.ndarray` depth map from the depth estimator's `forward(frame)`,
and a `classes: List[str]` taxonomy off the detector instance.
"""

import argparse
from pathlib import Path
from typing import Iterable

import cv2

from visual_radar.camera_view.renderer import CameraViewRenderer
from visual_radar.config import load_config
from visual_radar.radar.opencv_renderer import OpenCVRadarRenderer
from visual_radar.radar.scene_builder import build_radar_scene
from visual_radar.sources import Frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Visual Radar building blocks over an external frame iterable.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a YAML config file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = load_config(args.config)

    # detection_model/depth_model come from a separate project and are
    # imported/instantiated here directly, e.g.:
    #   from some_other_project.detection import SomeDetector
    #   from some_other_project.depth import SomeDepthEstimator
    #   detection_model = SomeDetector(weights_path=config["detection"]["weights_path"], device=config["detection"]["device"])
    #   depth_model = SomeDepthEstimator(checkpoint_path=config["depth"]["checkpoint_path"], device=config["depth"]["device"])
    detection_model = ...
    depth_model = ...

    radar_renderer = OpenCVRadarRenderer()
    camera_view_renderer = CameraViewRenderer()

    intrinsics = config["camera"]
    radar_view_config = config["radar_view"]

    output_dir = Path(config["output"]["output_dir"])
    (output_dir / "camera_view").mkdir(parents=True, exist_ok=True)
    (output_dir / "radar_view").mkdir(parents=True, exist_ok=True)

    # TODO: replace with a real Iterable[Frame] supplied by an ingestion project.
    frame_source: Iterable[Frame] = []

    for i, frame in enumerate(frame_source):
        detections = detection_model(frame)
        depth_map = depth_model(frame)
        scene = build_radar_scene(detections, depth_map, frame, intrinsics, detection_model.classes)

        camera_view = camera_view_renderer.render(frame, detections, detection_model.classes)
        radar_view = radar_renderer.render(scene, radar_view_config)

        cv2.imwrite(str(output_dir / "camera_view" / f"{i:06d}.png"), camera_view)
        cv2.imwrite(str(output_dir / "radar_view" / f"{i:06d}.png"), radar_view)


if __name__ == "__main__":
    main()
