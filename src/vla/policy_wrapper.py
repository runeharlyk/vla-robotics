"""
Adapts our custom models to the LeRobot Policy interface for LIBERO evaluation.

Any model that implements ``select_action(batch) -> Tensor`` can be wrapped
with :class:`PolicyWrapper` so that LeRobot's evaluation harness can drive it.
"""

from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class ActionPolicy(Protocol):
    """Minimal interface our models must implement."""

    def select_action(self, batch: dict) -> torch.Tensor: ...
    def reset(self) -> None: ...


class PolicyWrapper(nn.Module):
    """Wraps a custom model so it looks like a LeRobot-compatible policy.

    LeRobot evaluation calls ``policy.select_action(batch)`` and expects
    a ``(batch, action_dim)`` tensor back. This wrapper forwards to the
    underlying model and handles device placement.

    Args:
        model: Any model implementing the ActionPolicy protocol
        device: Torch device
    """

    def __init__(self, model: ActionPolicy, device: torch.device | str = "cuda"):
        super().__init__()
        self.model = model
        self._device = torch.device(device)

    @property
    def device(self) -> torch.device:
        return self._device

    def reset(self) -> None:
        self.model.reset()

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        return self.model.select_action(batch)

    def forward(self, batch: dict) -> dict:
        return self.model.forward(batch)
