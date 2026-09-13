"""Camera-view annotation renderer.

Renders annotated detection boxes over the camera frame. Independent of the
radar view — never composited into one frame with it.
"""

from typing import Dict, List, Optional

import cv2
import numpy as np

from ..detection.base import Detection
from ..radar.palette import default_class_colors
from ..sources import Frame


class CameraViewRenderer:
    def render(
        self,
        frame: Frame,
        detection: Detection,
        classes: List[str],
        class_colors: Optional[Dict[int, tuple]] = None,
    ) -> np.ndarray:
        """Draw annotated detection boxes over `frame.image` and return it as an array."""
        canvas = (frame.image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        canvas = np.ascontiguousarray(canvas)

        if class_colors is None:
            class_ids = {int(label) for label in detection.labels}
            class_colors = default_class_colors(class_ids)

        height, width = canvas.shape[:2]
        for i in range(detection.boxes.shape[0]):
            x1, y1, x2, y2 = detection.boxes[i].tolist()
            x1, y1 = max(0, int(round(x1))), max(0, int(round(y1)))
            x2, y2 = min(width, int(round(x2))), min(height, int(round(y2)))
            if x1 >= x2 or y1 >= y2:
                continue

            class_id = int(detection.labels[i])
            r, g, b = class_colors[class_id]
            color_bgr = (b, g, r)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, thickness=2)

            label = f"{classes[class_id]} {float(detection.confs[i]):.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y1 = max(0, y1 - text_h - baseline - 4)
            cv2.rectangle(canvas, (x1, label_y1), (x1 + text_w + 4, y1), color_bgr, thickness=-1)
            cv2.putText(
                canvas,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas
