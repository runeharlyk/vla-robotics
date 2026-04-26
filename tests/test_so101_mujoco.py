from __future__ import annotations

from pathlib import Path

import numpy as np

from vla.envs.so101_mujoco import MOTOR_NAMES, SO101MujocoEnv


def test_so101_mujoco_step_shapes(tmp_path: Path) -> None:
    env = SO101MujocoEnv(
        asset_path=_write_minimal_so101_mjcf(tmp_path),
        image_size=64,
        max_episode_steps=4,
        objects=False,
        seed=0,
    )
    try:
        obs, info = env.reset(seed=123)
        assert info["seed"] == 123
        assert obs["pixels"]["front"].shape == (64, 64, 3)
        assert obs["agent_state"].shape == (18,)

        action = np.zeros(len(MOTOR_NAMES), dtype=np.float32)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        assert next_obs["pixels"]["front"].shape == (64, 64, 3)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert step_info["step"] == 1
    finally:
        env.close()


def test_so101_mujoco_batch_contract(tmp_path: Path) -> None:
    env = SO101MujocoEnv(
        asset_path=_write_minimal_so101_mjcf(tmp_path),
        image_size=32,
        max_episode_steps=2,
        instruction="teleoperate the SO101 arm",
        objects=False,
        seed=0,
    )
    try:
        obs, _ = env.reset(seed=0)
        batch = env.obs_to_batch(obs)
        assert batch["observation.images.front"].shape == (1, 3, 32, 32)
        assert batch["observation.state"].shape == (1, 18)
        assert batch["task"] == ["teleoperate the SO101 arm"]
    finally:
        env.close()


def _write_minimal_so101_mjcf(tmp_path: Path) -> Path:
    path = tmp_path / "so101_test.xml"
    path.write_text(
        """
<mujoco model="so101_test">
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <headlight diffuse="0.8 0.8 0.8" ambient="0.3 0.3 0.3"/>
  </visual>
  <worldbody>
    <camera name="front" pos="0.6 -0.7 0.45" xyaxes="0.76 0.65 0 -0.28 0.33 0.90"/>
    <geom name="floor" type="plane" size="1 1 0.02"/>
    <body name="base" pos="0 0 0.04">
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-1.9 1.9"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.05" size="0.02"/>
      <body name="shoulder" pos="0 0 0.05">
        <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.7 1.7"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02"/>
        <body name="elbow" pos="0.1 0 0">
          <joint name="elbow_flex" type="hinge" axis="0 1 0" range="-1.7 1.7"/>
          <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02"/>
          <body name="wrist" pos="0.1 0 0">
            <joint name="wrist_flex" type="hinge" axis="0 1 0" range="-1.6 1.6"/>
            <geom type="capsule" fromto="0 0 0 0.06 0 0" size="0.015"/>
            <body name="roll" pos="0.06 0 0">
              <joint name="wrist_roll" type="hinge" axis="1 0 0" range="-2.8 2.8"/>
              <geom type="box" pos="0.02 0 0" size="0.02 0.03 0.01"/>
              <body name="jaw" pos="0.04 0.015 0">
                <joint name="gripper" type="hinge" axis="0 0 1" range="-0.17 1.75"/>
                <geom type="box" pos="0.02 0 0" size="0.02 0.005 0.01"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
""".strip(),
        encoding="utf-8",
    )
    return path
