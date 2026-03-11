"""Tensor utility helpers."""

from __future__ import annotations

import numpy as np
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


def action_to_numpy(action: torch.Tensor | np.ndarray | list) -> np.ndarray:
    """Convert a policy action output to a flat float32 numpy array.

    Handles torch tensors (detach + cpu), numpy arrays, and plain lists.

    Args:
        action: Action from a policy — tensor, array, or list.

    Returns:
        Flat ``float32`` numpy array.
    """
    if isinstance(action, torch.Tensor):
        return action.detach().cpu().numpy().flatten().astype(np.float32)
    return np.asarray(action, dtype=np.float32).flatten()
