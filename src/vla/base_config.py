"""Shared base training configuration.

Lives at the top of the ``vla`` package so that both ``vla.training`` and
``vla.rl`` can import it without circular dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
import torch.nn as nn

from vla.constants import Simulator


@dataclass
class BaseTrainingConfig:
    """Fields shared by all training modes (SFT, SRPO, etc.).

    Subclass this to define mode-specific hyperparameters while keeping
    the common knobs in one place.
    """

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-10
    max_grad_norm: float = 10.0
    eval_every: int = 10
    eval_episodes: int = 50
    max_steps: int = 280
    save_dir: str = "checkpoints"
    env_id: str = "PickCube-v1"
    seed: int = 42
    simulator: Simulator = Simulator.MANISKILL

    def to_dict(self) -> dict[str, Any]:
        """Serialize all fields to a plain dict (useful for W&B config)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def build_optimizer(
        self,
        policy: nn.Module,
    ) -> tuple[torch.optim.Optimizer, list[nn.Parameter]]:
        """Build an AdamW optimizer from this config's shared hyperparameters.

        Returns:
            ``(optimizer, trainable_params)`` tuple.
        """
        trainable = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )
        return optimizer, trainable
