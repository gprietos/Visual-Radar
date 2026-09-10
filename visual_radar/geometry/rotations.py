"""World-space rotation from camera yaw/pitch/roll.

The exact axis/rotation-order convention of whatever produced yaw/pitch/roll
must be confirmed per frame source (e.g. Android/iOS sensor conventions
differ from aerospace ZYX Tait-Bryan order). Getting this wrong doesn't
crash, it silently produces wrong azimuths.
"""

import numpy as np


def rotation_matrix(yaw: float, pitch: float, roll: float, convention: str = "aerospace_zyx") -> np.ndarray:
    """Build R(yaw, pitch, roll) for the given source's rotation convention.

    `convention` must match the actual sensor/source that produced the
    angles; do not assume the default is correct for a new source.
    """
    raise NotImplementedError
