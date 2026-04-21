from __future__ import annotations

import torch
import unittest

from vla.rl.robocasa_rollout import RoboCasaRollout


def test_robocasa_parallel_rollout_smoke() -> None:
    rollout = RoboCasaRollout(
        env_id="PickPlaceCounterToCabinet",
        num_envs=2,
        max_steps=2,
        layout_id=20,
        style_id=58,
        split="all",
        instruction="pick and place from counter to cabinet",
    )
    try:
        trajectories = rollout.collect_batch(
            policy_fn=None,
            instruction="pick and place from counter to cabinet",
            num_trajectories=2,
            seed=0,
            policy_batch_fn=lambda images, instruction, states: torch.zeros(
                (images.shape[0], 12), dtype=torch.float32
            ),
        )
    finally:
        rollout.close()

    assert len(trajectories) == 2
    assert all(traj.length > 0 for traj in trajectories)
    assert all(traj.actions.shape[-1] == 12 for traj in trajectories)


class RoboCasaParallelRolloutSmokeTest(unittest.TestCase):
    def test_parallel_rollout(self) -> None:
        test_robocasa_parallel_rollout_smoke()


if __name__ == "__main__":
    unittest.main()
