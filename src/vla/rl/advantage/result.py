"""Result dataclass for per-task advantage normalisation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AdvantageResult:
    """Output of per-task advantage normalisation."""

    advantages: list[float]
    skipped_tasks: list[str]
    per_task_g_mean: dict[str, float]
