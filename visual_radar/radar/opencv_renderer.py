"""OpenCV-based radar renderer, the v1 concrete `RadarRenderer`."""

import numpy as np

from .base import RadarPoint, RadarRenderer, RadarScene


class OpenCVRadarRenderer(RadarRenderer):
    """Builds the static background once via matplotlib, then does a matplotlib-free per-frame hot path."""

    def __init__(self):
        self._background: np.ndarray | None = None
        self._polar_to_pixel_transform = None
        self._sprites: dict = {}

    def render(self, scene: RadarScene, config: dict) -> np.ndarray:
        """Copy the cached background and alpha-composite each point's sprite. No matplotlib here."""
        raise NotImplementedError

    def _build_background(self, config: dict) -> None:
        """Build range rings, sector wedge, tick labels, legend via matplotlib exactly once.

        Extracts the polar->pixel transform from `ax.transData`, caches the
        rendered RGBA buffer via `np.array(fig.canvas.renderer.buffer_rgba())`,
        then discards the figure.
        """
        raise NotImplementedError

    def _load_background_image(self, config: dict) -> None:
        """Load `config['background_image_path']` directly; center_px/pixels_per_meter must be set manually."""
        raise NotImplementedError

    def _polar_to_pixel(self, azimuth_rad: float, range_m: float) -> tuple:
        raise NotImplementedError

    def _composite_point(self, canvas: np.ndarray, point: RadarPoint, config: dict) -> None:
        """Alpha-composite a pre-rendered per-class sprite onto canvas at the point's pixel position."""
        raise NotImplementedError
