"""Tensor utility helpers."""

from __future__ import annotations

import torch


def to_float01(img: torch.Tensor) -> torch.Tensor:
    """Convert a uint8 or float image tensor to float in [0, 1].

    Args:
        img: Image tensor of any dtype.

    Returns:
        Float tensor with values in ``[0, 1]``.
    """
    if img.dtype == torch.uint8:
        return img.float() / 255.0
    return img.float()
