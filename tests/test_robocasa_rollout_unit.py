from __future__ import annotations

import numpy as np
import torch

from vla.rl.robocasa_rollout import _raw_obs_to_tensors


def test_raw_obs_to_tensors_shapes() -> None:
    raw_obs = {
        "video.robot0_agentview_left": np.zeros((8, 8, 3), dtype=np.uint8),
        "video.robot0_agentview_right": np.zeros((8, 8, 3), dtype=np.uint8),
        "video.robot0_eye_in_hand": np.zeros((8, 8, 3), dtype=np.uint8),
        "state.gripper_qpos": np.zeros(2, dtype=np.float32),
        "state.base_position": np.zeros(3, dtype=np.float32),
        "state.base_rotation": np.zeros(4, dtype=np.float32),
        "state.end_effector_position_relative": np.zeros(3, dtype=np.float32),
        "state.end_effector_rotation_relative": np.zeros(4, dtype=np.float32),
    }

    images, state = _raw_obs_to_tensors(raw_obs)

    assert isinstance(images, torch.Tensor)
    assert isinstance(state, torch.Tensor)
    assert images.shape == (3, 3, 8, 8)
    assert state.shape == (16,)
