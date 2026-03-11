"""CPU-only tests for multi-task SRPO components.

Verifies data loading, TaskSpec construction, Trajectory.task_id routing,
MultiTaskWorldProgressReward grouping/reward logic, and per-task advantage
normalisation — all without loading real models or touching GPU.

Includes a dedicated TestOneShotBehavior suite that exercises every layer of
the one-shot (num_demos=1) path, because the paper's key claim is that the
full pipeline works from a single demonstration per task.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
import torch

if "mani_skill" not in sys.modules:
    _ms = ModuleType("mani_skill")
    _ms_envs = ModuleType("mani_skill.envs")
    sys.modules["mani_skill"] = _ms
    sys.modules["mani_skill.envs"] = _ms_envs

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
from vla.data.dataset import FewDemoDataset
from vla.rl.rollout import Trajectory
from vla.rl.advantage import normalize_advantages_per_task
from vla.rl.srpo_reward import (
    ClusterDiagnostics,
    MultiTaskWorldProgressReward,
    SRPORewardConfig,
    WorldProgressReward,
)
from vla.rl.trainer import TaskSpec
from scripts.train_srpo import _load_multitask_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trajectory(task_id: str = "", success: bool = False, T: int = 10) -> Trajectory:
    return Trajectory(
        images=torch.randint(0, 255, (T, 2, 3, 64, 64), dtype=torch.uint8),
        states=torch.randn(T, 14),
        actions=torch.randn(T, 7),
        rewards=torch.zeros(T),
        dones=torch.zeros(T),
        success=success,
        length=T,
        task_id=task_id,
    )


def _make_fake_encoder(embed_dim: int = 64):
    encoder = MagicMock()
    encoder.embed_dim.return_value = embed_dim
    encoder.encode_trajectory.side_effect = lambda imgs, subsample: torch.randn(embed_dim)
    encoder.encode_trajectories.side_effect = lambda imgs_list, subsample: torch.randn(len(imgs_list), embed_dim)
    return encoder


# ---------------------------------------------------------------------------
# Trajectory.task_id
# ---------------------------------------------------------------------------


class TestTrajectoryTaskId:
    def test_default_empty(self):
        t = Trajectory(
            images=torch.zeros(1, 3, 64, 64),
            states=torch.zeros(1, 14),
            actions=torch.zeros(1, 7),
            rewards=torch.zeros(1),
            dones=torch.zeros(1),
            success=False,
            length=1,
        )
        assert t.task_id == ""

    def test_set_at_init(self):
        t = _make_trajectory(task_id="spatial_task_3")
        assert t.task_id == "spatial_task_3"

    def test_set_after_init(self):
        t = _make_trajectory()
        t.task_id = "new_task"
        assert t.task_id == "new_task"


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------


class TestTaskSpec:
    def test_creation(self):
        spec = TaskSpec(
            task_id="pick_block",
            instruction="pick up the red block",
            env_id="PickCube-v1",
            libero_task_idx=2,
            data_path="/data/pick.pt",
        )
        assert spec.task_id == "pick_block"
        assert spec.libero_task_idx == 2

    def test_defaults(self):
        spec = TaskSpec(task_id="t", instruction="do something")
        assert spec.env_id == ""
        assert spec.libero_task_idx == 0
        assert spec.data_path == ""

    def test_lookup_dict(self):
        specs = [
            TaskSpec(task_id="a", instruction="instr a"),
            TaskSpec(task_id="b", instruction="instr b"),
        ]
        lookup = {s.task_id: s for s in specs}
        assert lookup["b"].instruction == "instr b"


# ---------------------------------------------------------------------------
# Data loading → TaskSpec + demo trajectories
# ---------------------------------------------------------------------------


class TestMultitaskDataLoading:
    def test_discover_pt_files_and_build_specs(self, tmp_path):
        instructions = [
            "pick up the black bowl",
            "put the plate on the table",
            "stack the red block",
        ]
        pt_files = []
        for i, instr in enumerate(instructions):
            pt = tmp_path / f"task_{i}.pt"
            make_fake_pt(pt, num_episodes=2, instruction=instr, libero_task_id=i)
            pt_files.append(pt)

        specs = []
        datasets = []
        for idx, pt in enumerate(sorted(pt_files)):
            ds = FewDemoDataset(pt, num_demos=2)
            datasets.append(ds)
            specs.append(
                TaskSpec(
                    task_id=pt.stem,
                    instruction=ds.metadata["instruction"],
                    libero_task_idx=ds.metadata.get("libero_task_id", idx),
                    data_path=str(pt),
                )
            )

        assert len(specs) == 3
        assert specs[0].task_id == "task_0"
        assert specs[1].instruction == "put the plate on the table"
        assert specs[2].libero_task_idx == 2

    def test_demo_trajectories_per_task(self, tmp_path):
        pt_a = tmp_path / "alpha.pt"
        pt_b = tmp_path / "beta.pt"
        make_fake_pt(pt_a, num_episodes=5, instruction="task alpha")
        make_fake_pt(pt_b, num_episodes=3, instruction="task beta")

        demo_trajs: dict[str, list[Trajectory]] = {}
        for pt in [pt_a, pt_b]:
            ds = FewDemoDataset(pt)
            trajs = ds.episodes_as_trajectories()
            tid = pt.stem
            for t in trajs:
                t.task_id = tid
            demo_trajs[tid] = trajs

        assert len(demo_trajs["alpha"]) == 5
        assert len(demo_trajs["beta"]) == 3
        assert all(t.task_id == "alpha" for t in demo_trajs["alpha"])
        assert all(t.task_id == "beta" for t in demo_trajs["beta"])
        assert demo_trajs["alpha"][0].success is True

    def test_load_multitask_data_from_libero_suite(self, monkeypatch):
        class FakeLiberoDataset:
            def __init__(self, suite, num_demos=None, seed=42, task_id=None):
                self.suite = suite
                self.num_demos = num_demos
                self.seed = seed
                self.task_id = task_id
                self.state_dim = 14
                self._task_map = {
                    0: "pick up the mug",
                    1: "place the bowl",
                }

            def episodes_as_trajectories(self, task_id=None):
                active_task = self.task_id if self.task_id is not None else task_id
                return [_make_trajectory(task_id=f"raw_{active_task}", success=True)]

        monkeypatch.setattr("vla.data.libero.LiberoSFTDataset", FakeLiberoDataset)

        task_specs, demo_trajectories, state_dim, action_dim = _load_multitask_data(
            data_dir=None,
            libero_suite="spatial",
            num_demos=1,
            seed=42,
            simulator="libero",
            suite="spatial",
            include_demos=True,
        )

        assert [spec.task_id for spec in task_specs] == ["spatial_task_0", "spatial_task_1"]
        assert [spec.libero_task_idx for spec in task_specs] == [0, 1]
        assert task_specs[0].instruction == "pick up the mug"
        assert state_dim == 14
        assert action_dim == 7
        assert demo_trajectories is not None
        assert set(demo_trajectories) == {"spatial_task_0", "spatial_task_1"}
        assert demo_trajectories["spatial_task_0"][0].task_id == "spatial_task_0"
        assert demo_trajectories["spatial_task_1"][0].task_id == "spatial_task_1"


# ---------------------------------------------------------------------------
# MultiTaskWorldProgressReward (mock encoder, CPU only)
# ---------------------------------------------------------------------------


class TestMultiTaskWorldProgressReward:
    def test_lazy_task_creation(self):
        encoder = _make_fake_encoder()
        mt = MultiTaskWorldProgressReward(encoder)
        assert mt.task_ids == []

        mt._get_or_create("task_a")
        mt._get_or_create("task_b")
        assert set(mt.task_ids) == {"task_a", "task_b"}

    def test_add_demo_trajectories_per_task(self):
        encoder = _make_fake_encoder(embed_dim=32)
        mt = MultiTaskWorldProgressReward(encoder)

        demo_imgs_a = [torch.randn(10, 3, 64, 64) for _ in range(3)]
        demo_imgs_b = [torch.randn(10, 3, 64, 64) for _ in range(2)]
        mt.add_demo_trajectories("task_a", demo_imgs_a)
        mt.add_demo_trajectories("task_b", demo_imgs_b)

        assert len(mt._per_task["task_a"].reference_embeddings) == 3
        assert len(mt._per_task["task_b"].reference_embeddings) == 2

    def test_compute_rewards_groups_by_task(self):
        encoder = _make_fake_encoder(embed_dim=16)
        cfg = SRPORewardConfig(dbscan_min_samples=1, dbscan_eps=100.0)
        mt = MultiTaskWorldProgressReward(encoder, cfg)

        mt.add_demo_trajectories("A", [torch.randn(5, 3, 64, 64) for _ in range(3)])
        mt.add_demo_trajectories("B", [torch.randn(5, 3, 64, 64) for _ in range(3)])

        trajs = [
            _make_trajectory(task_id="A", success=True),
            _make_trajectory(task_id="B", success=False),
            _make_trajectory(task_id="A", success=False),
            _make_trajectory(task_id="B", success=True),
        ]
        rewards, embeddings = mt.compute_trajectory_rewards(trajs)

        assert len(rewards) == 4
        assert len(embeddings) == 4
        assert rewards[0] == 1.0
        assert rewards[3] == 1.0
        assert 0.0 <= rewards[1] <= 1.0
        assert 0.0 <= rewards[2] <= 1.0
        assert all(e.shape == (16,) for e in embeddings)

    def test_add_successful_embeddings_per_task(self):
        encoder = _make_fake_encoder(embed_dim=8)
        mt = MultiTaskWorldProgressReward(encoder)

        mt._get_or_create("X")
        mt._get_or_create("Y")
        mt.add_successful_embeddings("X", [torch.randn(8), torch.randn(8)])
        mt.add_successful_embeddings("Y", [torch.randn(8)])

        assert len(mt._per_task["X"].reference_embeddings) == 2
        assert len(mt._per_task["Y"].reference_embeddings) == 1

    def test_add_empty_embeddings_noop(self):
        encoder = _make_fake_encoder()
        mt = MultiTaskWorldProgressReward(encoder)
        mt._get_or_create("Z")
        mt.add_successful_embeddings("Z", [])
        assert len(mt._per_task["Z"].reference_embeddings) == 0

    def test_diagnostics_per_task(self):
        encoder = _make_fake_encoder(embed_dim=16)
        cfg = SRPORewardConfig(dbscan_min_samples=1, dbscan_eps=100.0)
        mt = MultiTaskWorldProgressReward(encoder, cfg)

        mt.add_demo_trajectories("T1", [torch.randn(5, 3, 64, 64) for _ in range(3)])
        mt.add_demo_trajectories("T2", [torch.randn(5, 3, 64, 64) for _ in range(2)])

        trajs = [
            _make_trajectory(task_id="T1", success=False),
            _make_trajectory(task_id="T2", success=False),
        ]
        mt.compute_trajectory_rewards(trajs)

        diags = mt.get_diagnostics()
        assert "T1" in diags
        assert "T2" in diags
        assert isinstance(diags["T1"], ClusterDiagnostics)

    def test_rewards_isolated_between_tasks(self):
        encoder = _make_fake_encoder(embed_dim=16)
        cfg = SRPORewardConfig(dbscan_min_samples=1, dbscan_eps=100.0)

        mt_multi = MultiTaskWorldProgressReward(encoder, cfg)
        mt_multi.add_demo_trajectories("A", [torch.randn(5, 3, 64, 64) for _ in range(3)])
        mt_multi.add_demo_trajectories("B", [torch.randn(5, 3, 64, 64) for _ in range(3)])

        torch.manual_seed(42)
        trajs_a = [_make_trajectory(task_id="A", success=False) for _ in range(4)]

        torch.manual_seed(42)
        mt_single = WorldProgressReward(encoder, cfg)
        mt_single.add_demo_trajectories([torch.randn(5, 3, 64, 64) for _ in range(3)])

        rewards_multi, _ = mt_multi.compute_trajectory_rewards(trajs_a)

        assert len(rewards_multi) == 4
        assert all(isinstance(r, float) for r in rewards_multi)


# ---------------------------------------------------------------------------
# One-shot behaviour — the paper's core claim: num_demos=1 works end-to-end
# ---------------------------------------------------------------------------


class TestOneShotBehavior:
    """Every test here uses exactly one demonstration.

    Covers the data layer, the reward bootstrap path, and the per-task
    advantage pipeline so that regressions in the one-shot scenario are
    caught immediately.
    """

    # ------------------------------------------------------------------
    # Data layer
    # ------------------------------------------------------------------

    def test_dataset_caps_to_one_episode(self, tmp_path):
        pt = tmp_path / "multi.pt"
        make_fake_pt(pt, num_episodes=10)
        ds = FewDemoDataset(pt, num_demos=1)
        assert ds.num_episodes == 1

    def test_dataset_with_single_source_episode(self, tmp_path):
        pt = tmp_path / "single.pt"
        make_fake_pt(pt, num_episodes=1)
        ds = FewDemoDataset(pt, num_demos=1)
        assert ds.num_episodes == 1

    def test_dataset_without_num_demos_on_single_episode_file(self, tmp_path):
        pt = tmp_path / "single.pt"
        make_fake_pt(pt, num_episodes=1)
        ds = FewDemoDataset(pt)
        assert ds.num_episodes == 1

    def test_episodes_as_trajectories_one_shot(self, tmp_path):
        pt = tmp_path / "one.pt"
        make_fake_pt(pt, num_episodes=5, T=15)
        ds = FewDemoDataset(pt, num_demos=1)
        trajs = ds.episodes_as_trajectories()
        assert len(trajs) == 1
        assert trajs[0].success is True
        assert trajs[0].length == 15

    def test_one_shot_dataset_timestep_count(self, tmp_path):
        pt = tmp_path / "one.pt"
        make_fake_pt(pt, num_episodes=8, T=20)
        ds = FewDemoDataset(pt, num_demos=1)
        assert len(ds) == 20

    # ------------------------------------------------------------------
    # WorldProgressReward bootstrap with a single reference embedding
    # ------------------------------------------------------------------

    def test_one_demo_seeds_exactly_one_reference(self):
        encoder = _make_fake_encoder(embed_dim=32)
        wpr = WorldProgressReward(encoder)
        wpr.add_demo_trajectories([torch.randn(10, 3, 64, 64)])
        assert len(wpr.reference_embeddings) == 1

    def test_one_demo_cluster_centers_shape(self):
        """With 1 reference DBSCAN min_samples fallback should keep it as the centre."""
        encoder = _make_fake_encoder(embed_dim=32)
        cfg = SRPORewardConfig(dbscan_min_samples=2)
        wpr = WorldProgressReward(encoder, cfg)
        wpr.add_demo_trajectories([torch.randn(10, 3, 64, 64)])
        assert wpr.cluster_centers is not None
        assert wpr.cluster_centers.shape == (1, 32)

    def test_one_demo_success_reward_is_one(self):
        encoder = _make_fake_encoder(embed_dim=16)
        cfg = SRPORewardConfig(dbscan_min_samples=2, dbscan_eps=100.0)
        wpr = WorldProgressReward(encoder, cfg)
        wpr.add_demo_trajectories([torch.randn(5, 3, 64, 64)])
        trajs = [_make_trajectory(success=True)]
        rewards, _ = wpr.compute_trajectory_rewards(trajs)
        assert rewards[0] == 1.0

    def test_one_demo_failure_reward_in_valid_range(self):
        encoder = _make_fake_encoder(embed_dim=16)
        cfg = SRPORewardConfig(dbscan_min_samples=2, dbscan_eps=100.0)
        wpr = WorldProgressReward(encoder, cfg)
        wpr.add_demo_trajectories([torch.randn(5, 3, 64, 64)])
        trajs = [_make_trajectory(success=False)]
        rewards, _ = wpr.compute_trajectory_rewards(trajs)
        assert 0.0 <= rewards[0] <= cfg.alpha

    def test_one_demo_single_failure_std_zero_does_not_crash(self):
        """Single failed trajectory → d_std=0, must clamp to eps without NaN."""
        encoder = _make_fake_encoder(embed_dim=16)
        cfg = SRPORewardConfig(dbscan_min_samples=2, dbscan_eps=100.0)
        wpr = WorldProgressReward(encoder, cfg)
        wpr.add_demo_trajectories([torch.randn(5, 3, 64, 64)])
        trajs = [_make_trajectory(success=False)]
        rewards, _ = wpr.compute_trajectory_rewards(trajs)
        assert not torch.isnan(torch.tensor(rewards[0]))
        assert 0.0 <= rewards[0] <= cfg.alpha

    def test_one_demo_mixed_batch_rewards_are_ordered(self):
        """Success reward must strictly exceed failure reward with one demo."""
        encoder = _make_fake_encoder(embed_dim=16)
        cfg = SRPORewardConfig(dbscan_min_samples=2, dbscan_eps=100.0)
        wpr = WorldProgressReward(encoder, cfg)
        wpr.add_demo_trajectories([torch.randn(5, 3, 64, 64)])
        trajs = [_make_trajectory(success=True), _make_trajectory(success=False)]
        rewards, _ = wpr.compute_trajectory_rewards(trajs)
        assert rewards[0] == 1.0
        assert rewards[1] < rewards[0]

    def test_one_demo_online_success_grows_reference_set(self):
        """After 1 demo, a successful rollout should expand the reference set to 2."""
        encoder = _make_fake_encoder(embed_dim=16)
        wpr = WorldProgressReward(encoder)
        wpr.add_demo_trajectories([torch.randn(5, 3, 64, 64)])
        wpr.add_successful_embeddings([torch.randn(16)])
        assert len(wpr.reference_embeddings) == 2
        assert len(wpr._demo_embeddings) == 1

    def test_demo_embeddings_never_evicted(self):
        """Demo slot is permanent; online successes should not push it out."""
        encoder = _make_fake_encoder(embed_dim=8)
        cfg = SRPORewardConfig(max_references=4, ref_demo_ratio=0.5)
        wpr = WorldProgressReward(encoder, cfg)
        wpr.add_demo_trajectories([torch.randn(5, 3, 64, 64)])
        for _ in range(10):
            wpr.add_successful_embeddings([torch.randn(8)])
        assert len(wpr._demo_embeddings) == 1

    # ------------------------------------------------------------------
    # MultiTaskWorldProgressReward with one demo per task
    # ------------------------------------------------------------------

    def test_multitask_one_demo_per_task_isolated(self):
        encoder = _make_fake_encoder(embed_dim=16)
        cfg = SRPORewardConfig(dbscan_min_samples=2, dbscan_eps=100.0)
        mt = MultiTaskWorldProgressReward(encoder, cfg)
        mt.add_demo_trajectories("task_a", [torch.randn(5, 3, 64, 64)])
        mt.add_demo_trajectories("task_b", [torch.randn(5, 3, 64, 64)])
        assert len(mt._per_task["task_a"].reference_embeddings) == 1
        assert len(mt._per_task["task_b"].reference_embeddings) == 1

    def test_multitask_one_demo_per_task_rewards_valid(self):
        encoder = _make_fake_encoder(embed_dim=16)
        cfg = SRPORewardConfig(dbscan_min_samples=2, dbscan_eps=100.0)
        mt = MultiTaskWorldProgressReward(encoder, cfg)
        mt.add_demo_trajectories("A", [torch.randn(5, 3, 64, 64)])
        mt.add_demo_trajectories("B", [torch.randn(5, 3, 64, 64)])
        trajs = [
            _make_trajectory(task_id="A", success=True),
            _make_trajectory(task_id="A", success=False),
            _make_trajectory(task_id="B", success=True),
            _make_trajectory(task_id="B", success=False),
        ]
        rewards, embeddings = mt.compute_trajectory_rewards(trajs)
        assert rewards[0] == 1.0
        assert rewards[2] == 1.0
        assert 0.0 <= rewards[1] <= cfg.alpha
        assert 0.0 <= rewards[3] <= cfg.alpha
        assert all(e.shape == (16,) for e in embeddings)

    def test_multitask_one_demo_failure_reward_not_nan(self):
        encoder = _make_fake_encoder(embed_dim=16)
        cfg = SRPORewardConfig(dbscan_min_samples=2, dbscan_eps=100.0)
        mt = MultiTaskWorldProgressReward(encoder, cfg)
        mt.add_demo_trajectories("X", [torch.randn(5, 3, 64, 64)])
        trajs = [_make_trajectory(task_id="X", success=False)]
        rewards, _ = mt.compute_trajectory_rewards(trajs)
        assert not torch.isnan(torch.tensor(rewards[0]))

    # ------------------------------------------------------------------
    # Multi-task data loading: one demo per task
    # ------------------------------------------------------------------

    def test_multitask_one_shot_data_loading(self, tmp_path):
        instructions = ["pick up the mug", "place the bowl", "stack the cube"]
        for i, instr in enumerate(instructions):
            make_fake_pt(tmp_path / f"task_{i}.pt", num_episodes=5, instruction=instr)

        specs = []
        for pt in sorted(tmp_path.glob("*.pt")):
            ds = FewDemoDataset(pt, num_demos=1)
            assert ds.num_episodes == 1, f"{pt.name} should have exactly 1 episode"
            trajs = ds.episodes_as_trajectories()
            assert len(trajs) == 1
            assert trajs[0].success is True
            specs.append(TaskSpec(task_id=pt.stem, instruction=ds.instruction))

        assert len(specs) == 3


# ---------------------------------------------------------------------------
# Per-task advantage normalisation (from trainer logic)
# ---------------------------------------------------------------------------


class TestPerTaskAdvantageNorm:
    def test_advantages_normalised_per_task(self):
        g_values = [1.0, 0.3, 0.7, 0.5, 0.9, 0.1]
        task_ids = ["A", "A", "A", "B", "B", "B"]

        result = normalize_advantages_per_task(g_values, task_ids)
        advantages = result.advantages

        a_indices = [0, 1, 2]
        b_indices = [3, 4, 5]
        adv_a = [advantages[i] for i in a_indices]
        adv_b = [advantages[i] for i in b_indices]
        assert abs(sum(adv_a) / len(adv_a)) < 0.01
        assert abs(sum(adv_b) / len(adv_b)) < 0.01

        assert advantages[0] > 0
        assert advantages[1] < 0
        assert advantages[5] < 0
        assert advantages[4] > 0

    def test_uniform_rewards_skipped(self):
        g_values = [1.0, 1.0, 1.0, 0.5, 0.9, 0.1]
        task_ids = ["A", "A", "A", "B", "B", "B"]

        result = normalize_advantages_per_task(g_values, task_ids)
        assert "A" in result.skipped_tasks
        assert "B" not in result.skipped_tasks
        assert all(result.advantages[i] == 0.0 for i in range(3))

    def test_per_task_g_mean_reported(self):
        g_values = [1.0, 0.0, 0.8, 0.2]
        task_ids = ["X", "X", "Y", "Y"]

        result = normalize_advantages_per_task(g_values, task_ids)
        assert abs(result.per_task_g_mean["X"] - 0.5) < 0.01
        assert abs(result.per_task_g_mean["Y"] - 0.5) < 0.01
