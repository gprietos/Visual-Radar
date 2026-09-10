"""Per-class color derivation.

Colors are derived from whatever classes the plugged-in detector actually
reports, never a hardcoded fixed-length palette.
"""

from typing import Dict, Iterable


def default_class_colors(class_ids: Iterable[int]) -> Dict[int, tuple]:
    """Deterministically derive an RGB color per class_id (e.g. from a colormap)."""
    raise NotImplementedError
