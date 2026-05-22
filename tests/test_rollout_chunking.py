from __future__ import annotations

import numpy as np
import torch

from vla.models.smolvla import SmolVLAPolicy
from vla.rl.policy_update.base import _actions_and_mask_for_loss
from vla.rl.rollout import SingleStepResult, collect_single_episode_chunked
from vla.rl.vec_env import StepResult, collect_wave_chunked


class _FakeSingleAdapter:
    def __init__(self, terminate_after: int) -> None:
        self._terminate_after = terminate_after
        self._step_count = 0

    def reset(self, seed: int | None) -> int:
        self._step_count = 0
        return 0

    def obs_to_tensors(self, raw_obs: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.full((1, 1, 1), float(raw_obs))
        state = torch.tensor([float(raw_obs)])
        return image, state

    def step(self, action: np.ndarray) -> SingleStepResult:
        self._step_count += 1
        terminated = self._step_count >= self._terminate_after
        return SingleStepResult(
            raw_obs=self._step_count,
            reward=1.0,
            terminated=terminated,
            truncated=False,
            success=terminated,
        )


class _FakeVecAdapter:
    def __init__(self) -> None:
        self._obs = [0, 10]
        self._step_counts = [0, 0]
        self._done = [False, False]
        self._terminate_after = [3, 2]

    @property
    def num_envs(self) -> int:
        return 2

    def reset(self, seed: int | None) -> list[int]:
        self._obs = [0, 10]
        self._step_counts = [0, 0]
        self._done = [False, False]
        return list(self._obs)

    def extract_batch_obs(self, raw_obs: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        images = [torch.full((1, 1, 1), float(obs)) for obs in raw_obs]
        states = [torch.tensor([float(obs)]) for obs in raw_obs]
        return torch.stack(images), torch.stack(states)

    def step(self, actions: np.ndarray) -> StepResult:
        rewards: list[float] = []
        terminateds: list[bool] = []
        truncateds: list[bool] = []
        successes: list[bool] = []
        for i in range(self.num_envs):
            if not self._done[i]:
                self._step_counts[i] += 1
                self._obs[i] += 1
                if self._step_counts[i] >= self._terminate_after[i]:
                    self._done[i] = True
            rewards.append(1.0 if not self._done[i] or self._step_counts[i] <= self._terminate_after[i] else 0.0)
            terminateds.append(self._done[i])
            truncateds.append(False)
            successes.append(self._done[i])
        return StepResult(
            raw_obs=list(self._obs),
            rewards=rewards,
            terminateds=terminateds,
            truncateds=truncateds,
            successes=successes,
        )


class _ChunkBuilderStub:
    chunk_size = 4
    max_action_dim = 3

    @staticmethod
    def _normalize_action(actions: torch.Tensor) -> torch.Tensor:
        return actions

    @staticmethod
    def _prepare_action(actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] == 3:
            return actions
        padded = torch.zeros(*actions.shape[:-1], 3, dtype=actions.dtype, device=actions.device)
        padded[..., : actions.shape[-1]] = actions
        return padded

    def _build_chunks_from_executed(
        self,
        executed_chunks: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return SmolVLAPolicy._build_chunks_from_executed(self, executed_chunks, chunk_mask)


def test_collect_single_episode_chunked_records_executed_masks() -> None:
    adapter = _FakeSingleAdapter(terminate_after=3)

    def policy_chunk_fn(image: torch.Tensor, instruction: str, state: torch.Tensor | None = None) -> torch.Tensor:
        base = int(state[0].item())
        return torch.tensor([[base], [base + 1], [base + 2]], dtype=torch.float32)

    traj = collect_single_episode_chunked(
        adapter,
        policy_chunk_fn,
        "task",
        max_steps=5,
        n_action_steps=2,
        seed=0,
    )

    assert traj.length == 2
    assert traj.success is True
    assert traj.n_action_steps == 2
    assert traj.rewards.tolist() == [2.0, 1.0]
    assert traj.dones.tolist() == [0.0, 1.0]
    assert traj.actions.squeeze(-1).tolist() == [0.0, 2.0]
    assert traj.executed_chunks is not None
    assert traj.chunk_mask is not None
    assert traj.executed_chunks.squeeze(-1).tolist() == [[0.0, 1.0], [2.0, 0.0]]
    assert traj.chunk_mask.tolist() == [[True, True], [True, False]]


def test_chunked_rollout_can_reconstruct_v28_sliding_targets() -> None:
    adapter = _FakeSingleAdapter(terminate_after=5)

    def policy_chunk_fn(image: torch.Tensor, instruction: str, state: torch.Tensor | None = None) -> torch.Tensor:
        base = int(state[0].item())
        return torch.tensor([[base], [base + 1], [base + 2]], dtype=torch.float32)

    traj = collect_single_episode_chunked(
        adapter,
        policy_chunk_fn,
        "task",
        max_steps=5,
        n_action_steps=2,
        seed=0,
    )

    constructed, constructed_mask = _actions_and_mask_for_loss(traj, chunk_size=4, full_chunk_target=True)
    direct, direct_mask = _actions_and_mask_for_loss(traj, chunk_size=4, full_chunk_target=False)

    assert constructed.squeeze(-1).tolist() == [
        [0.0, 1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
    ]
    assert constructed_mask is not None
    assert constructed_mask.tolist() == [
        [True, True, True, True],
        [True, True, True, False],
        [True, False, False, False],
    ]

    assert direct.squeeze(-1).tolist() == [[0.0, 1.0], [2.0, 3.0], [4.0, 0.0]]
    assert direct_mask is not None
    assert direct_mask.tolist() == [[True, True], [True, True], [True, False]]


def test_collect_wave_chunked_tracks_per_env_masks() -> None:
    adapter = _FakeVecAdapter()

    def policy_chunk_batch_fn(images: torch.Tensor, instruction: str, states: torch.Tensor | None = None) -> torch.Tensor:
        bases = states[:, 0].to(torch.float32)
        chunks = []
        for base in bases:
            chunks.append(torch.tensor([[base.item()], [base.item() + 1], [base.item() + 2]], dtype=torch.float32))
        return torch.stack(chunks)

    trajs = collect_wave_chunked(
        adapter,
        policy_chunk_batch_fn,
        "task",
        active_n=2,
        seed=0,
        max_steps=5,
        n_action_steps=2,
    )

    assert len(trajs) == 2

    env0, env1 = trajs
    assert env0.length == 2
    assert env0.rewards.tolist() == [2.0, 1.0]
    assert env0.chunk_mask is not None
    assert env0.chunk_mask.tolist() == [[True, True], [True, False]]

    assert env1.length == 1
    assert env1.rewards.tolist() == [2.0]
    assert env1.chunk_mask is not None
    assert env1.chunk_mask.tolist() == [[True, True]]


def test_build_action_chunks_masks_unexecuted_chunk_tail() -> None:
    policy = _ChunkBuilderStub()
    executed = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    chunk_mask = torch.tensor([[True, False], [True, True]])

    chunks, mask = SmolVLAPolicy._build_action_chunks(policy, executed, chunk_mask)

    assert chunks.shape == (2, 4, 3)
    assert mask.shape == (2, 4)
    assert chunks[0, 0].tolist() == [1.0, 2.0, 0.0]
    assert chunks[0, 1].tolist() == [3.0, 4.0, 0.0]
    assert chunks[0, 2].tolist() == [0.0, 0.0, 0.0]
    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
    ]
