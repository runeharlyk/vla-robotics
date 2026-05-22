"""L2 distance metrics for action comparison."""

from __future__ import annotations

import torch


def _align_time_axis(
    predicted: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim two trajectories to the same timestep horizon."""
    T = min(predicted.shape[0], reference.shape[0])
    return predicted[:T], reference[:T]


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
    pred, ref = _align_time_axis(predicted, reference)
    return torch.norm(pred - ref, dim=-1)


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
    pred, ref = _align_time_axis(predicted, reference)
    abs_l2 = torch.norm(pred - ref, dim=-1)
    ref_norm = torch.norm(ref, dim=-1).clamp(min=eps)
    return abs_l2 / ref_norm


def compute_per_dimension_absolute_errors(
    predicted: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Per-timestep absolute error for each action dimension.

    Returns
    -------
    torch.Tensor
        ``(T, action_dim)`` absolute errors.
    """
    pred, ref = _align_time_axis(predicted, reference)
    return torch.abs(pred - ref)


def compute_per_dimension_relative_errors(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-timestep relative absolute error for each action dimension.

    Returns
    -------
    torch.Tensor
        ``(T, action_dim)`` relative absolute errors.
    """
    pred, ref = _align_time_axis(predicted, reference)
    abs_err = torch.abs(pred - ref)
    denom = torch.abs(ref).clamp(min=eps)
    return abs_err / denom


def compute_quality_degradation(
    noisy_predicted: torch.Tensor,
    clean_predicted: torch.Tensor,
    ground_truth: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Measure degradation in task quality due to noise.

    Uses ground truth as anchor and compares:
    - clean model error vs GT
    - noisy model error vs GT

    Returns
    -------
    dict[str, torch.Tensor]
        Keys:
        - ``clean_error_l2``: ``(T,)``
        - ``noisy_error_l2``: ``(T,)``
        - ``degradation_delta_l2``: ``(T,)`` noisy - clean
        - ``degradation_ratio_l2``: ``(T,)`` noisy / clean
    """
    T = min(
        noisy_predicted.shape[0],
        clean_predicted.shape[0],
        ground_truth.shape[0],
    )
    noisy = noisy_predicted[:T]
    clean = clean_predicted[:T]
    gt = ground_truth[:T]

    clean_error_l2 = torch.norm(clean - gt, dim=-1)
    noisy_error_l2 = torch.norm(noisy - gt, dim=-1)

    return {
        "clean_error_l2": clean_error_l2,
        "noisy_error_l2": noisy_error_l2,
        "degradation_delta_l2": noisy_error_l2 - clean_error_l2,
        "degradation_ratio_l2": noisy_error_l2 / clean_error_l2.clamp(min=eps),
    }