"""Adaptive task-level curriculum for perturbation training."""

from __future__ import annotations

from dataclasses import dataclass

from vla.rl.config import CurriculumConfig, TaskSpec


@dataclass
class CurriculumDecision:
    level: int
    active_task_count: int
    success_rate: float
    action: str
    high_streak: int
    low_streak: int


class AdaptiveTaskCurriculum:
    """Gate task activation by curriculum level and rollout success rate."""

    def __init__(self, config: CurriculumConfig, task_specs: list[TaskSpec]) -> None:
        self.config = config
        levels = [spec.curriculum_level for spec in task_specs if spec.curriculum_level > 0]
        if not levels:
            levels = [0]
        self.min_level = min(levels)
        configured_start = max(config.start_level, self.min_level)
        self.level = configured_start
        self.max_level = config.max_level if config.max_level is not None else max(levels)
        self.max_level = max(self.max_level, self.level)
        self.high_streak = 0
        self.low_streak = 0
        self.last_action = "init"

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def active_specs(self, task_specs: list[TaskSpec]) -> list[TaskSpec]:
        if not self.enabled:
            return task_specs
        active = [spec for spec in task_specs if spec.curriculum_level <= self.level]
        return active or task_specs

    def update(self, successes: dict[str, int], task_specs: list[TaskSpec], trajs_per_task: int) -> CurriculumDecision:
        active = self.active_specs(task_specs)
        total = sum(successes.get(spec.task_id, 0) for spec in active)
        denom = max(len(active) * trajs_per_task, 1)
        success_rate = total / denom
        action = "hold"

        if success_rate >= self.config.target_max:
            self.high_streak += 1
            self.low_streak = 0
        elif success_rate < self.config.target_min:
            self.low_streak += 1
            self.high_streak = 0
        else:
            self.high_streak = 0
            self.low_streak = 0

        patience = max(self.config.patience, 1)
        if self.high_streak >= patience and self.level < self.max_level:
            self.level += 1
            self.high_streak = 0
            self.low_streak = 0
            action = "increase"
        elif self.config.allow_regression and self.low_streak >= patience and self.level > self.min_level:
            self.level -= 1
            self.high_streak = 0
            self.low_streak = 0
            action = "decrease"

        self.last_action = action
        return CurriculumDecision(
            level=self.level,
            active_task_count=len(self.active_specs(task_specs)),
            success_rate=success_rate,
            action=action,
            high_streak=self.high_streak,
            low_streak=self.low_streak,
        )
