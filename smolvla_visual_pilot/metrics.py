"""L2 distance metrics for action comparison."""

from __future__ import annotations

import torch


def compute_l2_distances(
    predicted: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Per-timestep absolute L2 distance.

    Parameters
    ----------
    predicted : torch.Tensor
        ``(T, action_dim)`` predicted actions.
    reference : torch.Tensor
        ``(T, action_dim)`` reference (ground-truth or clean-baseline) actions.

    Returns
    -------
    torch.Tensor
        ``(T,)`` L2 distances.
    """
    T = min(predicted.shape[0], reference.shape[0])
    return torch.norm(predicted[:T] - reference[:T], dim=-1)


def compute_relative_l2_distances(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-timestep relative L2 distance (normalised by reference norm).

    Parameters
    ----------
    predicted : torch.Tensor
        ``(T, action_dim)`` predicted actions.
    reference : torch.Tensor
        ``(T, action_dim)`` reference actions.
    eps : float
        Epsilon to avoid division by zero.

    Returns
    -------
    torch.Tensor
        ``(T,)`` relative L2 distances.
    """
    T = min(predicted.shape[0], reference.shape[0])
    abs_l2 = torch.norm(predicted[:T] - reference[:T], dim=-1)
    ref_norm = torch.norm(reference[:T], dim=-1).clamp(min=eps)
    return abs_l2 / ref_norm
