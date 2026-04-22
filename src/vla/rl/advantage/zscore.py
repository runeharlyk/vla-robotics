"""Per-task z-score advantage normalisation for SRPO training.

Extracts the z-score normalisation logic so it can be tested independently
of the full training loop.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from .result import AdvantageResult


def normalize_advantages_per_task(
    g_values: list[float],
    task_ids: list[str],
    eps: float = 1e-8,
    skip_threshold: float = 1e-6,
) -> AdvantageResult:
    """Z-score normalise rewards per task, skipping uniform-reward tasks.

    Args:
        g_values: Per-trajectory reward values (aligned with *task_ids*).
        task_ids: Per-trajectory task identifier strings.
        eps: Minimum standard deviation clamp to avoid division by zero.
        skip_threshold: Tasks whose reward std falls below this are skipped
            (all their advantages stay at 0.0).

    Returns:
        An :class:`AdvantageResult` containing the normalised advantages,
        the list of skipped task ids, and the per-task reward means.
    """
    n = len(g_values)
    advantages = [0.0] * n
    by_task: dict[str, list[int]] = defaultdict(list)
    for i, tid in enumerate(task_ids):
        by_task[tid].append(i)

    per_task_g_mean: dict[str, float] = {}
    skipped_tasks: list[str] = []

    for tid, indices in by_task.items():
        task_g = torch.tensor([g_values[i] for i in indices], dtype=torch.float32)
        g_mean = task_g.mean()
        g_std = task_g.std().clamp(min=eps)
        per_task_g_mean[tid] = g_mean.item()
        if g_std < skip_threshold:
            skipped_tasks.append(tid)
            continue
        task_adv = ((task_g - g_mean) / g_std).tolist()
        for j, idx in enumerate(indices):
            advantages[idx] = task_adv[j]

    return AdvantageResult(
        advantages=advantages,
        skipped_tasks=skipped_tasks,
        per_task_g_mean=per_task_g_mean,
    )
