"""Per-class color derivation.

Colors are derived from whatever classes the plugged-in detector actually
reports, never a hardcoded fixed-length palette.
"""

import colorsys
from typing import Dict, Iterable

_GOLDEN_RATIO_CONJUGATE = 0.6180339887
_SATURATION = 0.65
_VALUE = 0.95


def default_class_colors(class_ids: Iterable[int]) -> Dict[int, tuple]:
    """Deterministically derive an RGB (0-255) color per class_id.

    Each id's color is a pure function of that id alone (golden-ratio hue
    step), so it never depends on which other ids are present in a given
    call — a class's color stays stable across frames even as the set of
    currently-detected classes changes.
    """
    colors = {}
    for class_id in set(class_ids):
        hue = (class_id * _GOLDEN_RATIO_CONJUGATE) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, _SATURATION, _VALUE)
        colors[class_id] = (round(r * 255), round(g * 255), round(b * 255))
    return colors
