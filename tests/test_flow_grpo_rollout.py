from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType

import numpy as np
import torch

from vla.diagnostics import eval as eval_mod
from vla.diagnostics.eval import metrics_from_trajectories
from vla.rl.rollout import Trajectory
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


def test_chunked_eval_metrics_count_environment_steps_not_decisions() -> None:
    trajs = collect_wave_chunked(
        _Adapter(),
        lambda images, instruction, states: _Sample(
            actions=torch.ones(images.shape[0], 3, 2),
            flow_states=torch.zeros(images.shape[0], 2, 3, 4),
            flow_next_states=torch.ones(images.shape[0], 2, 3, 4),
            flow_times=torch.full((images.shape[0], 2), 0.5),
            flow_dts=torch.full((images.shape[0], 2), -0.5),
            flow_sigmas=torch.full((images.shape[0], 2), 0.1),
        ),
        instruction="test",
        active_n=1,
        seed=123,
        max_steps=2,
        n_action_steps=2,
    )

    metrics = metrics_from_trajectories(trajs)

    assert metrics.mean_episode_length == 2.0


def test_libero_flow_sde_eval_uses_flow_sampler(monkeypatch) -> None:
    calls: list[dict] = []

    class _Policy:
        state_dim = 8
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.configured: tuple[float, int] | None = None

        def set_eval_fixed_noise(self, seed: int | None) -> None:
            self.fixed_noise_seed = seed

        def configure_flow_grpo_sampler(self, *, sigma: float, sde_steps: int = 0) -> None:
            self.configured = (sigma, sde_steps)

        def predict_action(self, *args, **kwargs):
            raise AssertionError("normal single-action sampler should not be used")

        def predict_action_batch(self, *args, **kwargs):
            raise AssertionError("normal batched sampler should not be used")

        def predict_action_chunk(self, *args, **kwargs):
            raise AssertionError("normal chunk sampler should not be used")

        def predict_action_chunk_batch(self, *args, **kwargs):
            raise AssertionError("normal batched chunk sampler should not be used")

        def predict_action_flow_grpo(self, *args, **kwargs):
            return torch.zeros(7)

        def predict_action_flow_grpo_batch(self, *args, **kwargs):
            return torch.zeros(1, 7)

        def predict_action_chunk_flow_grpo(self, *args, **kwargs):
            return torch.zeros(5, 7)

        def predict_action_chunk_flow_grpo_batch(self, *args, **kwargs):
            return torch.zeros(1, 5, 7)

    class _FakeFactory:
        num_tasks = 1

    class _FakeRollout:
        task_description = "test instruction"

        def __init__(self, *args, **kwargs) -> None:
            pass

        def reconfigure(self, *args, **kwargs) -> None:
            pass

        def collect_batch(self, **kwargs):
            calls.append(kwargs)
            sample = kwargs["policy_chunk_batch_fn"](
                torch.zeros(1, 1, 3, 8, 8),
                "test instruction",
                torch.zeros(1, 8),
            )
            assert sample.shape == (1, 5, 7)
            return [
                Trajectory(
                    images=torch.zeros(1, 3, 8, 8),
                    states=torch.zeros(1, 8),
                    actions=torch.zeros(1, 7),
                    rewards=torch.ones(1),
                    dones=torch.ones(1),
                    success=True,
                    length=1,
                )
            ]

        def close(self) -> None:
            pass

    fake_module = ModuleType("vla.rl.libero_rollout")
    fake_module.LiberoRollout = _FakeRollout
    monkeypatch.setitem(sys.modules, "vla.rl.libero_rollout", fake_module)
    monkeypatch.setattr(eval_mod, "make_env_factory", lambda *args, **kwargs: _FakeFactory())

    policy = _Policy()
    metrics = eval_mod.evaluate_smolvla(
        policy,
        instruction="test instruction",
        simulator="libero",
        suite="spatial",
        num_episodes=1,
        num_envs=2,
        n_action_steps=5,
        rollout_sampler="flow_sde",
        flow_grpo_sigma=0.03,
        flow_grpo_sde_steps=10,
    )

    assert metrics.success_rate == 1.0
    assert policy.configured == (0.03, 10)
    assert calls[0]["policy_fn"].__name__ == "_single_fn"
    assert calls[0]["policy_batch_fn"].__name__ == "_batch_fn"
    assert calls[0]["policy_chunk_fn"].__name__ == "_chunk_single_fn"
    assert calls[0]["policy_chunk_batch_fn"].__name__ == "_chunk_batch_fn"
