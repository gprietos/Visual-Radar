"""Camera-view annotation renderer.

Renders annotated detection boxes over the camera frame. Independent of the
radar view — never composited into one frame with it.
"""

from typing import Dict, Optional

import numpy as np

from ..detection.base import DetectionOutput
from ..sources import Frame


class CameraViewRenderer:
    def render(
        self,
        frame: Frame,
        detection_output: DetectionOutput,
        class_names: Dict[int, str],
        class_colors: Optional[Dict[int, tuple]] = None,
    ) -> np.ndarray:
        """Draw annotated detection boxes over `frame.image` and return it as an array."""
        raise NotImplementedError
