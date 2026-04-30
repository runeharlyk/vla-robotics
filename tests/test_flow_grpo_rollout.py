from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from vla.rl.vec_env import StepResult, collect_wave_chunked


@dataclass
class _Sample:
    actions: torch.Tensor
    flow_states: torch.Tensor
    flow_next_states: torch.Tensor
    flow_times: torch.Tensor
    flow_dts: torch.Tensor
    flow_sigmas: torch.Tensor


class _Adapter:
    num_envs = 2

    def __init__(self) -> None:
        self.steps = [0, 0]

    def reset(self, seed: int | None) -> object:
        self.steps = [0, 0]
        return object()

    def extract_batch_obs(self, raw_obs: object) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(2, 3, 8, 8, dtype=torch.uint8), torch.zeros(2, 1)

    def step(self, actions: np.ndarray) -> StepResult:
        for i, action in enumerate(actions):
            if np.any(action):
                self.steps[i] += 1
        done = [step >= 2 for step in self.steps]
        return StepResult(
            raw_obs=object(),
            rewards=[0.0, 1.0],
            terminateds=done,
            truncateds=[False, False],
            successes=[False, True],
        )


def test_chunked_vector_rollout_preserves_flow_grpo_path_metadata() -> None:
    def policy(images: torch.Tensor, instruction: str, states: torch.Tensor) -> _Sample:
        batch = images.shape[0]
        actions = torch.ones(batch, 3, 2)
        path_shape = (batch, 2, 3, 4)
        return _Sample(
            actions=actions,
            flow_states=torch.zeros(path_shape),
            flow_next_states=torch.ones(path_shape),
            flow_times=torch.full((batch, 2), 0.5),
            flow_dts=torch.full((batch, 2), -0.5),
            flow_sigmas=torch.full((batch, 2), 0.1),
        )

    trajs = collect_wave_chunked(
        _Adapter(),
        policy,
        instruction="test",
        active_n=2,
        seed=123,
        max_steps=2,
        n_action_steps=2,
    )

    assert len(trajs) == 2
    assert trajs[0].flow_states is not None
    assert trajs[0].flow_states.shape == (1, 2, 3, 4)
    assert trajs[0].flow_next_states is not None
    assert trajs[0].flow_next_states.shape == (1, 2, 3, 4)
    assert trajs[0].flow_times is not None
    assert trajs[0].flow_times.shape == (1, 2)
    assert trajs[0].executed_chunks is not None
    assert trajs[0].executed_chunks.shape == (1, 2, 2)
    assert trajs[0].chunk_mask is not None
    assert trajs[0].chunk_mask.tolist() == [[True, True]]
