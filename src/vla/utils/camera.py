"""Camera view padding utilities."""

from __future__ import annotations

import numpy as np


def pad_camera_views(
    views: list[np.ndarray],
    num_cameras: int,
    *,
    default: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Ensure exactly ``num_cameras`` views by truncating or duplicating.

    If fewer views are available than requested, the last view is
    duplicated to fill the gap.  When ``views`` is empty, ``default``
    is used as the fill value (raises if neither is available).

    Args:
        views: Available camera view arrays (any shape).
        num_cameras: Desired number of views.
        default: Fallback array used when ``views`` is empty.

    Returns:
        List of exactly ``num_cameras`` arrays.
    """
    if len(views) >= num_cameras:
        return views[:num_cameras]
    if not views and default is None:
        raise RuntimeError("No camera views available and no default provided")
    padded = list(views)
    fill = padded[-1] if padded else default
    while len(padded) < num_cameras:
        padded.append(fill.copy())
    return padded[:num_cameras]
