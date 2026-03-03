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
    loaders = {
        "smolvla": _load_smolvla,
        "custom": _load_custom_vla,
    }
    loader = loaders.get(model.lower())
    if loader is None:
        raise ValueError(f"Unknown model {model!r}. Available: {sorted(loaders)}")
    return loader(checkpoint, device)


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
        action_dim = policy.action_dim
        state_dim = policy.state_dim

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


def _load_custom_vla(checkpoint: str, device: str) -> LoadedPolicy:
    from vla.models.custom_vla import CustomVLA

    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device_obj, weights_only=False)

    config = ckpt["config"]
    model_kwargs = config["model_kwargs"]
    action_dim = config.get("action_dim", model_kwargs.get("action_dim", 7))

    model = CustomVLA(**model_kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    dtype = torch.bfloat16 if device_obj.type == "cuda" else torch.float32
    model = model.to(device_obj, dtype=dtype)

    state_dim = action_dim

    def preprocessor(batch: dict) -> dict:
        image_key = next((k for k in batch if k.startswith("observation.images.")), None)
        images = batch[image_key] if image_key else torch.zeros(1, 3, 256, 256, device=device_obj)
        state = batch.get("observation.state", torch.zeros(1, action_dim, device=device_obj))
        instruction = batch.get("task", [""])
        return {"images": images, "state": state, "instruction": instruction}

    def postprocessor(action: torch.Tensor) -> torch.Tensor:
        return action

    return LoadedPolicy(
        policy=model,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        state_dim=state_dim,
        action_dim=action_dim,
    )


__all__ = ["SmolVLAPolicy", "LoadedPolicy", "load_policy"]
