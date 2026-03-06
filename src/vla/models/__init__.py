from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vla.models.smolvla import SmolVLAPolicy


@dataclass
class LoadedPolicy:
    policy: Any
    preprocessor: Callable[[dict], dict]
    postprocessor: Callable[[torch.Tensor], torch.Tensor]
    state_dim: int
    action_dim: int


class _SmolVLAAdapter:
    """Adapts the standalone SmolVLAPolicy to the select_action / reset interface
    expected by the simulator-agnostic evaluation pipeline."""

    def __init__(self, inner: SmolVLAPolicy) -> None:
        self._inner = inner

    def reset(self) -> None:
        pass

    def eval(self):
        self._inner.eval()
        return self

    def train(self, mode: bool = True):
        self._inner.train(mode)
        return self

    def parameters(self):
        return self._inner.parameters()

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        image_key = next((k for k in batch if k.startswith("observation.images.")), None)
        if image_key is None:
            raise ValueError(f"No observation.images.* key in batch. Keys: {list(batch.keys())}")

        image = batch[image_key]
        if image.ndim in (4, 5):
            image = image[0]

        state = batch.get("observation.state")
        if state is not None and state.ndim == 2:
            state = state[0]

        task = batch.get("task", "")
        if isinstance(task, (list, tuple)):
            task = task[0]

        action = self._inner.predict_action(image, task, state)
        return action.unsqueeze(0)


def load_policy(model: str, checkpoint: str, device: str) -> LoadedPolicy:
    if model.lower() != "smolvla":
        raise ValueError(f"Unknown model {model!r}. Available: smolvla")
    return _load_smolvla(checkpoint, device)


def _load_smolvla(checkpoint: str, device: str) -> LoadedPolicy:
    p = Path(checkpoint)

    if p.is_dir() and (p / "policy.pt").exists():
        inner = torch.load(p / "policy.pt", map_location="cpu", weights_only=False)
        base_ckpt = inner["checkpoint"]
        action_dim = inner.get("action_dim", 8)
        state_dim = inner.get("state_dim", 0)
        policy = SmolVLAPolicy(checkpoint=base_ckpt, action_dim=action_dim, state_dim=state_dim, device=device)
        policy.load_checkpoint(p)
    else:
        policy = SmolVLAPolicy(checkpoint=checkpoint, device=device)
        ckpt_cfg = policy.ckpt_config
        output_features = ckpt_cfg.get("output_features", {})
        action_shape = output_features.get("action", {}).get("shape")
        action_dim = action_shape[0] if action_shape else policy.action_dim
        input_features = ckpt_cfg.get("input_features", {})
        state_shape = input_features.get("observation.state", {}).get("shape")
        state_dim = state_shape[0] if state_shape else policy.state_dim
        policy.action_dim = action_dim
        policy.state_dim = state_dim

    adapter = _SmolVLAAdapter(policy)

    def _identity_pre(batch: dict) -> dict:
        return batch

    def _identity_post(action: torch.Tensor) -> torch.Tensor:
        return action

    return LoadedPolicy(
        policy=adapter,
        preprocessor=_identity_pre,
        postprocessor=_identity_post,
        state_dim=state_dim,
        action_dim=action_dim,
    )


__all__ = ["SmolVLAPolicy", "LoadedPolicy", "load_policy"]
