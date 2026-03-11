"""LR scheduling utilities for VLA training."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


class CosineDecayWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Cosine decay with linear warmup as a first-class ``_LRScheduler``.

    Suitable for use cases that need ``get_lr()`` introspection (e.g. RL
    trainers) rather than the ``LambdaLR``-based alternative.
    """

    def __init__(self, optimizer, peak_lr, decay_lr, warmup_steps, decay_steps, last_epoch=-1):
        self.peak_lr = peak_lr
        self.decay_lr = decay_lr
        self._warmup_steps = warmup_steps
        self._decay_steps = decay_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self._warmup_steps:
            lr = self.peak_lr * step / max(1, self._warmup_steps)
        elif step < self._decay_steps:
            progress = (step - self._warmup_steps) / max(1, self._decay_steps - self._warmup_steps)
            lr = self.decay_lr + (self.peak_lr - self.decay_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            lr = self.decay_lr
        return [lr for _ in self.base_lrs]


def cosine_decay_with_warmup_lambda_lr(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    decay_steps: int,
    peak_lr: float,
    decay_lr: float,
    total_steps: int,
) -> LambdaLR:
    """Create a :class:`LambdaLR` with cosine decay + linear warmup.

    If ``total_steps < decay_steps`` both warmup and decay durations are
    scaled proportionally (matching the LeRobot auto-scale behaviour).

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Steps for linear warmup.
        decay_steps: Total scheduled duration (warmup + cosine phase).
        peak_lr: Maximum learning rate.
        decay_lr: Minimum learning rate at the end of cosine decay.
        total_steps: Actual total training steps (triggers auto-scaling).

    Returns:
        Configured :class:`LambdaLR` scheduler.
    """
    if decay_steps > total_steps:
        warmup_steps = int(warmup_steps * total_steps / decay_steps)
        decay_steps = total_steps

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return (current_step + 1) / (warmup_steps + 1)
        progress = min((current_step - warmup_steps) / max(decay_steps - warmup_steps, 1), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = decay_lr + (peak_lr - decay_lr) * cosine
        return lr / peak_lr

    return LambdaLR(optimizer, _lr_lambda)
