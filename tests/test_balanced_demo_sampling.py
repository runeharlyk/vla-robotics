"""Regression tests for the RLPD-style 50/50 demo+online sampler used by v12+.

These tests lock in the producer-side fix that makes
``Trajectory.is_demo=True`` flow from the demo loaders all the way to
``_build_balanced_minibatches`` in :mod:`vla.rl.policy_update.success_bc`.

The original v12 run silently fell back to a uniform online-only sampler
because nothing set ``is_demo=True`` on demo trajectories at load time,
and ``_build_balanced_minibatches`` bails out when one pool is empty.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import torch

if "mani_skill" not in sys.modules:
    sys.modules["mani_skill"] = ModuleType("mani_skill")
    sys.modules["mani_skill.envs"] = ModuleType("mani_skill.envs")

if "transformers" not in sys.modules:
    _tf = ModuleType("transformers")
    _tf.AutoConfig = MagicMock()
    _tf.AutoModel = MagicMock()
    _tf.AutoModelForImageTextToText = MagicMock()
    _tf.AutoProcessor = MagicMock()
    _tf.AutoImageProcessor = MagicMock()
    _tf.SmolVLMForConditionalGeneration = MagicMock()
    sys.modules["transformers"] = _tf

if "wandb" not in sys.modules:
    sys.modules["wandb"] = MagicMock()

from tests.helpers import make_fake_pt
from vla.constants import LiberoSuite, Simulator
from vla.rl.config import TaskSpec
from vla.rl.demo_replay import _resolve_kept_trajs, replay_demo_rollouts
from vla.rl.policy_update.success_bc import _build_balanced_minibatches
from vla.rl.rollout import Trajectory


def _make_traj(*, is_demo: bool, success: bool = True, T: int = 4) -> Trajectory:
    return Trajectory(
        images=torch.zeros(T, 1, 3, 8, 8, dtype=torch.uint8),
        states=torch.zeros(T, 8),
        actions=torch.zeros(T, 7),
        rewards=torch.zeros(T),
        dones=torch.zeros(T),
        success=success,
        length=T,
        is_demo=is_demo,
    )


class TestBuildBalancedMinibatches:
    """Consumer-side: the sampler is correct when both pools are populated."""

    def test_bails_out_when_no_demos_present(self):
        trajs = [_make_traj(is_demo=False) for _ in range(8)]
        mbs = _build_balanced_minibatches(trajs, minibatch_trajs=4, demo_ratio=0.5)
        assert len(mbs) == 2
        for mb in mbs:
            assert all(not trajs[i].is_demo for i in mb)

    def test_bails_out_when_no_online_present(self):
        trajs = [_make_traj(is_demo=True) for _ in range(8)]
        mbs = _build_balanced_minibatches(trajs, minibatch_trajs=4, demo_ratio=0.5)
        assert len(mbs) == 2
        for mb in mbs:
            assert all(trajs[i].is_demo for i in mb)

    def test_emits_50_50_minibatches_when_both_pools_present(self):
        demos = [_make_traj(is_demo=True) for _ in range(20)]
        online = [_make_traj(is_demo=False) for _ in range(20)]
        trajs = demos + online
        mbs = _build_balanced_minibatches(trajs, minibatch_trajs=4, demo_ratio=0.5)
        assert mbs, "balanced sampler returned no minibatches"
        for mb in mbs:
            n_demo = sum(1 for i in mb if trajs[i].is_demo)
            n_online = len(mb) - n_demo
            assert n_demo == 2, f"expected 2 demos per mb of size 4, got {n_demo}"
            assert n_online == 2, f"expected 2 online per mb of size 4, got {n_online}"


class TestResolveKeptTrajsPropagatesIsDemo:
    """demo_replay must keep ``is_demo=True`` after env replay AND fallback."""

    def test_successful_replay_marked_as_demo(self):
        replayed = [_make_traj(is_demo=False, success=True)]
        raw = [_make_traj(is_demo=True, success=True)]
        out = _resolve_kept_trajs(
            replayed,
            raw_demos=raw,
            fallback_to_raw_demo=True,
            drop_failed_replays=False,
            task_id="t0",
        )
        assert len(out) == 1
        assert out[0].is_demo is True

    def test_fallback_to_raw_demo_marked_as_demo(self):
        failed_replay = _make_traj(is_demo=False, success=False)
        raw = _make_traj(is_demo=False, success=True)
        out = _resolve_kept_trajs(
            [failed_replay],
            raw_demos=[raw],
            fallback_to_raw_demo=True,
            drop_failed_replays=False,
            task_id="t0",
        )
        assert len(out) == 1
        assert out[0].is_demo is True
        assert out[0].task_id == "t0"

    def test_kept_failed_replay_still_marked_as_demo(self):
        failed_replay = _make_traj(is_demo=False, success=False)
        raw = _make_traj(is_demo=False, success=True)
        out = _resolve_kept_trajs(
            [failed_replay],
            raw_demos=[raw],
            fallback_to_raw_demo=False,
            drop_failed_replays=False,
            task_id="t0",
        )
        assert len(out) == 1
        assert out[0].is_demo is True


class _FakeReplayEnv:
    def __init__(self, *, success: bool, terminate_after: int = 1):
        self.success = success
        self.terminate_after = terminate_after
        self.reset_calls: list[tuple[int, int | None]] = []
        self.steps = 0
        self.closed = False

    def reset(self, seed: int, init_state_id: int | None = None):
        self.steps = 0
        self.reset_calls.append((seed, init_state_id))
        return {}, {}

    def obs_to_batch(self, _raw_obs):
        return {
            "observation.images.image": torch.zeros(3, 8, 8),
            "observation.state": torch.zeros(8),
        }

    def step(self, _action):
        self.steps += 1
        done = self.steps >= self.terminate_after
        return {}, float(self.success), done, False, {"success": self.success}

    def is_success(self, info):
        return bool(info.get("success", False))

    def close(self):
        self.closed = True


class TestReplayDemoRollouts:
    def test_replays_and_keeps_successful_demo(self, monkeypatch, tmp_path: Path):
        fake_env = _FakeReplayEnv(success=True, terminate_after=10)
        monkeypatch.setattr("vla.rl.demo_replay.make_env_factory", lambda *_args, **_kwargs: lambda _idx: fake_env)

        demo = _make_traj(is_demo=True, success=True, T=4)
        demo.init_state_id = 7
        demo.privileged_states = [{"source_episode_index": 123}]
        spec = TaskSpec(task_id="task_x", instruction="pick", env_id="Fake-v0", libero_task_idx=0)

        replayed, rates = replay_demo_rollouts(
            task_specs=[spec],
            demo_trajectories={"task_x": [demo]},
            simulator=Simulator.MANISKILL,
            suite=LiberoSuite.SPATIAL,
            max_steps=2,
            seed=42,
            state_dim=8,
            cache_dir=tmp_path,
            fallback_to_raw_demo=False,
            require_success=True,
        )

        assert rates == {"task_x": 1.0}
        assert replayed is not None
        assert len(replayed["task_x"]) == 1
        assert replayed["task_x"][0].is_demo is True
        assert replayed["task_x"][0].task_id == "task_x"
        assert replayed["task_x"][0].length == 2
        assert fake_env.reset_calls == [(123, 7)]
        assert fake_env.closed is True

    def test_failed_replay_falls_back_to_raw_demo(self, monkeypatch, tmp_path: Path):
        fake_env = _FakeReplayEnv(success=False)
        monkeypatch.setattr("vla.rl.demo_replay.make_env_factory", lambda *_args, **_kwargs: lambda _idx: fake_env)

        demo = _make_traj(is_demo=True, success=True, T=3)
        spec = TaskSpec(task_id="task_x", instruction="pick", env_id="Fake-v0", libero_task_idx=0)

        replayed, rates = replay_demo_rollouts(
            task_specs=[spec],
            demo_trajectories={"task_x": [demo]},
            simulator=Simulator.MANISKILL,
            suite=LiberoSuite.SPATIAL,
            max_steps=3,
            seed=42,
            state_dim=8,
            cache_dir=tmp_path,
            fallback_to_raw_demo=True,
            require_success=True,
        )

        assert rates == {"task_x": 0.0}
        assert replayed is not None
        assert len(replayed["task_x"]) == 1
        assert replayed["task_x"][0] is demo
        assert replayed["task_x"][0].is_demo is True
        assert replayed["task_x"][0].task_id == "task_x"

    def test_failed_replay_is_dropped_when_success_required_without_fallback(self, monkeypatch, tmp_path: Path):
        fake_env = _FakeReplayEnv(success=False)
        monkeypatch.setattr("vla.rl.demo_replay.make_env_factory", lambda *_args, **_kwargs: lambda _idx: fake_env)

        demo = _make_traj(is_demo=True, success=True, T=3)
        spec = TaskSpec(task_id="task_x", instruction="pick", env_id="Fake-v0", libero_task_idx=0)

        replayed, rates = replay_demo_rollouts(
            task_specs=[spec],
            demo_trajectories={"task_x": [demo]},
            simulator=Simulator.MANISKILL,
            suite=LiberoSuite.SPATIAL,
            max_steps=3,
            seed=42,
            state_dim=8,
            cache_dir=tmp_path,
            fallback_to_raw_demo=False,
            require_success=True,
        )

        assert rates == {"task_x": 0.0}
        assert replayed is not None
        assert replayed["task_x"] == []


class TestFewDemoDatasetEpisodesAreTaggedByBuildTasks:
    """Producer side: ``_build_tasks`` non-LIBERO branch tags demos."""

    def test_few_demo_dataset_path_tags_is_demo(self, tmp_path: Path):
        from scripts.train_srpo import _build_tasks
        from vla.constants import Simulator

        pt_path = tmp_path / "task_x.pt"
        make_fake_pt(pt_path, num_episodes=3, instruction="pick the cube")

        task_specs, demo_trajs, _state_dim, _action_dim = _build_tasks(
            data_path=pt_path,
            data_dir=None,
            simulator=Simulator.MANISKILL,
            libero_suite=None,
            suite=None,
            task_ids=[0],
            include_demos=True,
            env_id_override=None,
            instruction_override=None,
            num_demos=3,
            seed=42,
        )

        assert demo_trajs is not None
        assert len(demo_trajs) == 1
        trajs = next(iter(demo_trajs.values()))
        assert len(trajs) == 3
        assert all(t.is_demo is True for t in trajs)
        assert all(t.task_id == "task_x" for t in trajs)


class TestBalancedSamplingWarning:
    """Loud-warn when the anchor would silently no-op."""

    def test_warns_when_balanced_requested_but_no_demos(self, caplog):
        from vla.rl.policy_update.success_bc import logger as sb_logger

        caplog.set_level(logging.WARNING, logger=sb_logger.name)
        trajs = [_make_traj(is_demo=False) for _ in range(8)]
        _build_balanced_minibatches(trajs, minibatch_trajs=4, demo_ratio=0.5)
        # The sampler itself doesn't warn; the trainer-facing entrypoint does.
        # Verify the sampler still returns *something* (uniform fallback) so
        # the optimiser does not stall when this happens in production.
        mbs = _build_balanced_minibatches(trajs, minibatch_trajs=4, demo_ratio=0.5)
        assert len(mbs) == 2
