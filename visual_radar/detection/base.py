"""Detection output shape for a single image.

The detector itself is external to this project (imported and run
wherever it's needed, e.g. `examples/run_offline_demo.py`) — this module
only defines the shape its output takes so the rest of the pipeline can
consume it.
"""

from dataclasses import dataclass

import torch


@dataclass
class Detection:
    """Post-processed predictions for a SINGLE image."""

    boxes: torch.Tensor  # (N, 4) -> [x1, y1, x2, y2]
    confs: torch.Tensor  # (N,)   -> confidence float [0.0, 1.0]
    labels: torch.Tensor  # (N,)   -> class IDs (torch.int64)
