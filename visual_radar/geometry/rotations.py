"""World-space rotation from camera yaw/pitch/roll.

The exact axis/rotation-order convention of whatever produced yaw/pitch/roll
must be confirmed per frame source (e.g. Android/iOS sensor conventions
differ from aerospace ZYX Tait-Bryan order). Getting this wrong doesn't
crash, it silently produces wrong azimuths.

`"aerospace_zyx"` (the only convention implemented in v1) fixes:
- Units: yaw/pitch/roll in radians.
- World frame: X = right/east, Y = forward/north (heading zero), Z = up.
- Camera ray frame (pre-rotation): x_cam = right, y_cam = down,
  z_cam = forward (optical axis) — matches `projection.pixel_to_camera_ray`.
- Yaw about world Z, compass-positive: yaw=0 -> forward = world +Y;
  positive yaw turns forward toward +X (right).
- Pitch about world X: positive = camera tilts up.
- Roll about world Y: positive = clockwise about the forward axis, seen
  from behind the camera (right side dips down).

`rotation_matrix()` returns the full camera-ray-to-world matrix (axis
remap folded in), so callers apply it directly to a camera-space ray:
`world_dir = rotation_matrix(yaw, pitch, roll) @ ray_cam`.
"""

import numpy as np

# Remaps camera-ray axes (right, down, forward) to world-aligned body axes
# (right, forward, up) at zero rotation.
_CAMERA_TO_BODY = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)


def _rotation_z_compass(yaw: float) -> np.ndarray:
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cos_yaw, sin_yaw, 0.0],
            [-sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _rotation_x(pitch: float) -> np.ndarray:
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_p, -sin_p],
            [0.0, sin_p, cos_p],
        ]
    )


def _rotation_y(roll: float) -> np.ndarray:
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    return np.array(
        [
            [cos_r, 0.0, sin_r],
            [0.0, 1.0, 0.0],
            [-sin_r, 0.0, cos_r],
        ]
    )


def rotation_matrix(yaw: float, pitch: float, roll: float, convention: str = "aerospace_zyx") -> np.ndarray:
    """Build the camera-ray-to-world rotation matrix R(yaw, pitch, roll).

    `convention` must match the actual sensor/source that produced the
    angles; do not assume the default is correct for a new source. Only
    `"aerospace_zyx"` is implemented in v1.
    """
    if convention != "aerospace_zyx":
        raise ValueError(f"Unsupported rotation convention: {convention!r}")

    return _rotation_z_compass(yaw) @ _rotation_x(pitch) @ _rotation_y(roll) @ _CAMERA_TO_BODY
