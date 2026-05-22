from __future__ import annotations

import torch

from vla.data.dataset import FewDemoDataset
from vla.rl.rollout import Trajectory
from vla.rl.trajectory_io import save_trajectories_as_sft_pt


def _trajectory(T: int = 4, action_dim: int = 7, state_dim: int = 8) -> Trajectory:
    return Trajectory(
        images=torch.randint(0, 255, (T, 2, 3, 32, 32), dtype=torch.uint8),
        states=torch.randn(T, state_dim),
        actions=torch.arange(T * action_dim, dtype=torch.float32).reshape(T, action_dim),
        rewards=torch.ones(T),
        dones=torch.zeros(T),
        success=True,
        length=T,
        task_id="spatial_task_0",
    )


def test_export_flat_trajectory_loads_with_sliding_action_chunks(tmp_path):
    path = tmp_path / "success.pt"
    traj = _trajectory(T=4)

    save_trajectories_as_sft_pt(
        [traj],
        path,
        metadata={"env_id": "libero_spatial", "simulator": "libero"},
        instructions_by_task={"spatial_task_0": "pick up the object"},
        action_chunk_size=3,
    )

    ds = FewDemoDataset(path, action_chunk_size=3)
    assert len(ds) == 4
    assert ds.metadata["instruction"] == "pick up the object"
    sample = ds[2]
    assert sample["instruction"] == "pick up the object"
    assert sample["action_chunk"].shape == (3, 7)
    assert sample["action_mask"].tolist() == [True, True, False]
    torch.testing.assert_close(sample["action_chunk"][0], traj.actions[2])
    torch.testing.assert_close(sample["action_chunk"][1], traj.actions[3])


def test_export_chunked_trajectory_preserves_executed_chunk_targets(tmp_path):
    path = tmp_path / "chunked.pt"
    traj = _trajectory(T=2)
    traj.n_action_steps = 2
    traj.executed_chunks = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]],
            [[15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ]
    )
    traj.chunk_mask = torch.tensor([[True, True], [True, False]])

    save_trajectories_as_sft_pt(
        [traj],
        path,
        metadata={"env_id": "libero_spatial", "simulator": "libero"},
        instructions_by_task={"spatial_task_0": "move the mug"},
        action_chunk_size=5,
    )

    ds = FewDemoDataset(path, action_chunk_size=5)
    first = ds[0]
    second = ds[1]
    assert first["action_mask"].tolist() == [True, True, False, False, False]
    assert second["action_mask"].tolist() == [True, False, False, False, False]
    torch.testing.assert_close(first["action_chunk"][:2], traj.executed_chunks[0])
    torch.testing.assert_close(second["action_chunk"][0], traj.executed_chunks[1, 0])


def test_export_filters_failed_trajectories(tmp_path):
    path = tmp_path / "filtered.pt"
    success = _trajectory(T=2)
    failed = _trajectory(T=2)
    failed.success = False

    save_trajectories_as_sft_pt(
        [failed, success],
        path,
        metadata={"env_id": "libero_spatial", "simulator": "libero"},
    )

    ds = FewDemoDataset(path)
    assert ds.num_episodes == 1
    assert len(ds) == 2
