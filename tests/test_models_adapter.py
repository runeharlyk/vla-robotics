from __future__ import annotations

import torch

from vla.models import _SmolVLAAdapter


class _FakePolicy:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, str, torch.Tensor | None]] = []

    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def parameters(self):
        return iter(())

    def predict_action_batch(
        self,
        image: torch.Tensor,
        instruction: str,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls.append((image.clone(), instruction, None if state is None else state.clone()))
        return torch.arange(7, dtype=torch.float32).unsqueeze(0)


def test_smolvla_adapter_preserves_multi_view_inputs() -> None:
    policy = _FakePolicy()
    adapter = _SmolVLAAdapter(policy)

    batch = {
        "observation.images.robot0_eye_in_hand_image": torch.full((1, 3, 8, 8), 2.0),
        "observation.images.agentview_image": torch.ones((1, 3, 8, 8)),
        "observation.state": torch.arange(8, dtype=torch.float32).unsqueeze(0),
        "task": ["pick up the bowl"],
    }

    action = adapter.select_action(batch)

    assert action.shape == (1, 7)
    assert len(policy.calls) == 1

    image, instruction, state = policy.calls[0]
    assert image.shape == (1, 2, 3, 8, 8)
    assert instruction == "pick up the bowl"
    assert state is not None
    assert state.shape == (1, 8)

    # Sorted keys keep the agent view first, then wrist view.
    assert torch.equal(image[0, 0], batch["observation.images.agentview_image"][0])
    assert torch.equal(image[0, 1], batch["observation.images.robot0_eye_in_hand_image"][0])
