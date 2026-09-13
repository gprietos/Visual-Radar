"""OpenCV-based radar renderer, the v1 concrete `RadarRenderer`.

`RadarPoint.azimuth_rad` is an absolute azimuth (the frame's yaw is already
baked in by `geometry.projection.detection_to_world_point`). Whether
`render()` subtracts `scene.heading_rad` before converting to a pixel
position is controlled by `radar_view.ego_centric` (default `False`):
`True` keeps camera-forward pointing "up" on the canvas regardless of
absolute heading (ego-centric); `False` leaves azimuth absolute, so "up" is
a fixed world reference and the camera's facing rotates on-canvas as
heading changes (world-referenced).

`show_labels` is a reserved no-op in v1 (per-point text is v2+, tied to
`RadarPoint.track_id`); `size_by` only supports `"fixed"`. Both raise
`NotImplementedError` if misused, rather than silently doing nothing.
"""

import matplotlib

matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .base import RadarPoint, RadarRenderer, RadarScene
from .palette import default_class_colors


class OpenCVRadarRenderer(RadarRenderer):
    """Builds the static background once via matplotlib, then does a matplotlib-free per-frame hot path."""

    def __init__(self):
        self._background: np.ndarray | None = None
        self._polar_to_pixel_transform = None
        self._canvas_size: int | None = None
        self._manual_center_px: tuple | None = None
        self._manual_pixels_per_meter: float | None = None
        self._sprites: dict = {}
        self._sprite_radius_px: int | None = None
        self._colors: dict = {}
        self._legend_entries: dict = {}

    def render(self, scene: RadarScene, config: dict) -> np.ndarray:
        """Copy the cached background and alpha-composite each point's sprite. No matplotlib here."""
        if config.get("show_labels"):
            raise NotImplementedError(
                "radar_view.show_labels is reserved for v2+ per-point text (e.g. track id); not implemented in v1"
            )
        size_by = config.get("size_by", "fixed")
        if size_by != "fixed":
            raise NotImplementedError(f"radar_view.size_by={size_by!r} is not implemented in v1; only 'fixed' is")

        if self._background is None:
            if config.get("background_image_path"):
                self._load_background_image(config)
            else:
                self._build_background(config)

        offset_rad = scene.heading_rad if config.get("ego_centric", False) else 0.0

        canvas = self._background.copy()
        self._draw_fov_wedge(canvas, scene.heading_rad - offset_rad, scene.fov_deg, config)

        for point in scene.points:
            display_azimuth_rad = point.azimuth_rad - offset_rad
            self._composite_point(canvas, point, display_azimuth_rad, config)
            self._legend_entries[point.class_id] = (point.class_name, self._colors[point.class_id])

        if config.get("show_legend") and self._legend_entries:
            self._draw_legend(canvas)

        return canvas

    def _build_background(self, config: dict) -> None:
        """Build range rings and tick labels via matplotlib exactly once.

        Extracts the polar->pixel transform from `ax.transData`, caches the
        rendered RGBA buffer via `np.array(fig.canvas.renderer.buffer_rgba())`,
        then discards the figure. The FOV wedge and legend are NOT baked in
        here: FOV only arrives per-`render()` call (via `scene.fov_deg`), and
        the legend grows as new classes are seen — both are drawn fresh each
        frame with plain cv2 instead.
        """
        canvas_size = config["canvas_size"]
        max_range_m = config["max_range_m"]
        ring_interval_m = config["ring_interval_m"]

        dpi = 100
        fig = plt.figure(figsize=(canvas_size / dpi, canvas_size / dpi), dpi=dpi)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="polar")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, max_range_m)

        ring_values = np.arange(ring_interval_m, max_range_m + ring_interval_m, ring_interval_m)
        ax.set_yticks(ring_values)
        ax.set_yticklabels([f"{ring_m:g}m" for ring_m in ring_values], fontsize=8)
        ax.set_xticklabels([])
        ax.grid(True, color="gray", alpha=0.5)

        fig.canvas.draw()
        buffer_rgba = np.array(fig.canvas.renderer.buffer_rgba())
        background_bgr = cv2.cvtColor(buffer_rgba, cv2.COLOR_RGBA2BGR)
        if background_bgr.shape[0] != canvas_size or background_bgr.shape[1] != canvas_size:
            background_bgr = cv2.resize(background_bgr, (canvas_size, canvas_size))

        self._background = background_bgr
        self._polar_to_pixel_transform = ax.transData
        self._canvas_size = canvas_size

        plt.close(fig)

    def _load_background_image(self, config: dict) -> None:
        """Load `config['background_image_path']` directly; center_px/pixels_per_meter are set manually."""
        canvas_size = config["canvas_size"]
        image = cv2.imread(config["background_image_path"])
        if image is None:
            raise FileNotFoundError(f"Could not load radar background image: {config['background_image_path']!r}")
        if image.shape[0] != canvas_size or image.shape[1] != canvas_size:
            image = cv2.resize(image, (canvas_size, canvas_size))

        self._background = image
        self._canvas_size = canvas_size
        self._polar_to_pixel_transform = None
        self._manual_center_px = (canvas_size / 2.0, canvas_size / 2.0)
        self._manual_pixels_per_meter = (canvas_size / 2.0) / config["max_range_m"]

    def _polar_to_pixel(self, azimuth_rad: float, range_m: float) -> tuple:
        if self._polar_to_pixel_transform is not None:
            x_px, y_px_from_bottom = self._polar_to_pixel_transform.transform((azimuth_rad, range_m))
            return float(x_px), float(self._canvas_size - y_px_from_bottom)

        cx, cy = self._manual_center_px
        ppm = self._manual_pixels_per_meter
        x_px = cx + range_m * ppm * np.sin(azimuth_rad)
        y_px = cy - range_m * ppm * np.cos(azimuth_rad)
        return float(x_px), float(y_px)

    def _draw_fov_wedge(self, canvas: np.ndarray, center_azimuth_rad: float, fov_deg: float, config: dict) -> None:
        """Shade the camera's FOV sector, centered at `center_azimuth_rad` ("up" when ego-centric)."""
        center_x, center_y = self._polar_to_pixel(center_azimuth_rad, 0.0)
        center = (int(round(center_x)), int(round(center_y)))

        edge_x, edge_y = self._polar_to_pixel(center_azimuth_rad, config["max_range_m"])
        axis_radius_px = int(round(np.hypot(edge_x - center[0], edge_y - center[1])))

        center_azimuth_deg = np.degrees(center_azimuth_rad)
        half_fov_deg = fov_deg / 2.0
        overlay = canvas.copy()
        cv2.ellipse(
            overlay,
            center,
            (axis_radius_px, axis_radius_px),
            0,
            -90 + center_azimuth_deg - half_fov_deg,
            -90 + center_azimuth_deg + half_fov_deg,
            (200, 200, 200),
            thickness=-1,
        )
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, dst=canvas)

    def _ensure_sprite(self, class_id: int, config: dict) -> None:
        radius_px = config["point_radius_px"]
        if self._sprite_radius_px != radius_px:
            self._sprites.clear()
            self._sprite_radius_px = radius_px

        if class_id in self._sprites:
            return

        if class_id not in self._colors:
            self._colors.update(default_class_colors([class_id]))
        r, g, b = self._colors[class_id]

        diameter = 2 * radius_px + 1
        sprite = np.zeros((diameter, diameter, 4), dtype=np.uint8)
        cv2.circle(sprite, (radius_px, radius_px), radius_px, (b, g, r, 255), thickness=-1)
        self._sprites[class_id] = sprite

    def _composite_point(self, canvas: np.ndarray, point: RadarPoint, display_azimuth_rad: float, config: dict) -> None:
        """Alpha-composite a pre-rendered per-class sprite onto canvas at the point's pixel position."""
        self._ensure_sprite(point.class_id, config)
        sprite = self._sprites[point.class_id]
        radius_px = self._sprite_radius_px

        px, py = self._polar_to_pixel(display_azimuth_rad, point.range_m)
        cx, cy = int(round(px)), int(round(py))

        x1, y1 = cx - radius_px, cy - radius_px
        x2, y2 = cx + radius_px + 1, cy + radius_px + 1
        sx1, sy1, sx2, sy2 = 0, 0, sprite.shape[1], sprite.shape[0]

        canvas_h, canvas_w = canvas.shape[:2]
        if x1 < 0:
            sx1, x1 = -x1, 0
        if y1 < 0:
            sy1, y1 = -y1, 0
        if x2 > canvas_w:
            sx2, x2 = sx2 - (x2 - canvas_w), canvas_w
        if y2 > canvas_h:
            sy2, y2 = sy2 - (y2 - canvas_h), canvas_h
        if x1 >= x2 or y1 >= y2:
            return

        sprite_region = sprite[sy1:sy2, sx1:sx2]
        canvas_region = canvas[y1:y2, x1:x2]
        alpha = sprite_region[:, :, 3:4].astype(np.float32) / 255.0
        blended = sprite_region[:, :, :3].astype(np.float32) * alpha + canvas_region.astype(np.float32) * (1 - alpha)
        canvas_region[:] = blended.astype(np.uint8)

    def _draw_legend(self, canvas: np.ndarray) -> None:
        x0, y0, line_height = 10, 10, 18
        for i, (class_id, (class_name, color)) in enumerate(sorted(self._legend_entries.items())):
            y = y0 + i * line_height
            r, g, b = color
            cv2.rectangle(canvas, (x0, y), (x0 + 12, y + 12), (b, g, r), thickness=-1)
            cv2.putText(
                canvas, class_name, (x0 + 18, y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA
            )
