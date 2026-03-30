"""Per-task advantage normalisation for SRPO training.

Extracts the z-score normalisation logic so it can be tested independently
of the full training loop.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from vla.constants import UpdateMethod


@dataclass
class AdvantageResult:
    """Output of per-task advantage normalisation."""

    advantages: list[float]
    skipped_tasks: list[str]
    per_task_g_mean: dict[str, float]


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


def leave_one_out_advantages_per_task(
    g_values: list[float],
    task_ids: list[str],
    update_method: UpdateMethod,
    skip_threshold: float = 1e-6,
) -> AdvantageResult:
    n = len(g_values)
    advantages = [0.0] * n
    by_task: dict[str, list[int]] = defaultdict(list)
    for i, tid in enumerate(task_ids):
        by_task[tid].append(i)

    per_task_g_mean: dict[str, float] = {}
    skipped_tasks: list[str] = []

    for tid, indices in by_task.items():
        task_g = torch.tensor([g_values[i] for i in indices], dtype=torch.float32)
        per_task_g_mean[tid] = task_g.mean().item()

        m = task_g.numel()
        if m <= 1:
            skipped_tasks.append(tid)
            continue

        if task_g.std(unbiased=False).item() < skip_threshold:
            skipped_tasks.append(tid)
            continue

        # Leave-one-out baseline: b_k = (1/(K-1)) * Σ_{j≠k} R_j
        # Advantage: A_k = R_k - b_k
        # Per RIPT-VLA (Section 3.2) and SimpleVLA-RL's GRPO formulation.
        baselines = (task_g.sum() - task_g) / (m - 1)
        raw_adv = task_g - baselines

        if update_method in (UpdateMethod.AWR, UpdateMethod.FPO):
            # Z-score normalisation: Â_k = (A_k - μ_A) / σ_A
            # Per GRPO (SimpleVLA-RL Eq. 5) and RIPT-VLA, advantages must be
            # normalised to zero mean and unit variance for methods like AWR and FPO.
            adv_std = raw_adv.std().clamp(min=1e-8)
            task_adv = ((raw_adv - raw_adv.mean()) / adv_std).tolist()
        else:
            # Per RIPT-VLA (Section 3.2) and CombinedVLA-RL, RLOO advantages
            # for PPO are NOT Z-score normalised (unlike GRPO). This keeps advantages
            # bounded in [-1, +1] and prevents gradient explosions.
            task_adv = raw_adv.tolist()

        for j, idx in enumerate(indices):
            advantages[idx] = task_adv[j]

    return AdvantageResult(
        advantages=advantages,
        skipped_tasks=skipped_tasks,
        per_task_g_mean=per_task_g_mean,
    )
