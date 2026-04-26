from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla.envs.base import SimEnv
from vla.utils.tensor import to_float01

MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

DEFAULT_LOCAL_SCENE_PATH = Path("third_party/SO-ARM100/Simulation/SO101/scene.xml")
SO_ARM100_URL = "https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101"


class SO101MujocoEnv(SimEnv):
    """MuJoCo loader for the real SO101 asset.

    The recommended model is TheRobotStudio's SO101 MJCF scene because it
    includes the calibrated robot XML, STL assets, materials, joint ranges, and
    STS3215 actuator metadata. URDF paths are also accepted, but MJCF is the
    practical default for MuJoCo.
    """

    def __init__(
        self,
        asset_path: Path | str | None = None,
        image_size: int = 256,
        max_episode_steps: int = 250,
        instruction: str = "teleoperate the SO101 arm",
        objects: bool = True,
        seed: int | None = None,
        base_pos: tuple[float, float, float] | None = (0, 0, 0.05),
        base_quat: tuple[float, float, float, float] | None = (0, 0, 0, 1),
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self._asset_path = _resolve_asset_path(asset_path)
        self._scene_path = _write_interactive_scene(self._asset_path) if objects else self._asset_path
        self._model = mujoco.MjModel.from_xml_path(str(self._scene_path))
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=image_size, width=image_size)
        self._image_size = image_size
        self._max_episode_steps = max_episode_steps
        self._instruction = instruction
        self._objects_enabled = objects
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        self._joint_ids = np.array(
            [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in MOTOR_NAMES],
            dtype=np.int32,
        )
        missing = [name for name, joint_id in zip(MOTOR_NAMES, self._joint_ids, strict=True) if joint_id < 0]
        if missing:
            raise ValueError(f"SO101 model {self._asset_path} is missing joints: {missing}")

        self._qposadr = self._model.jnt_qposadr[self._joint_ids].astype(np.int32)
        self._qveladr = self._model.jnt_dofadr[self._joint_ids].astype(np.int32)
        self._actuator_ids = np.array(
            [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in MOTOR_NAMES],
            dtype=np.int32,
        )

        self._ranges = self._model.jnt_range[self._joint_ids].astype(np.float32)
        self._home = _range_midpoint(self._ranges)
        self._camera_name = _first_camera_name(self._model, mujoco)

        print("\n[DEBUG] Found cameras in model:")
        for i in range(self._model.ncam):
            print(f"  - Camera {i}: {mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_CAMERA, i)}")

        self._object_joint_qpos = _freejoint_qpos_addresses(
            self._model,
            mujoco,
            ("cube_free", "cylinder_free", "ball_free"),
        )

        # Rotate the robot base 180 degrees around the Z axis to face the workspace at -X by default.
        # MuJoCo quaternions are (w, x, y, z).
        base_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_body_id >= 0:
            if base_pos is not None:
                self._model.body_pos[base_body_id] = base_pos
            if base_quat is not None:
                self._model.body_quat[base_body_id] = base_quat

    @property
    def task_description(self) -> str:
        return self._instruction

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @property
    def action_dim(self) -> int:
        return len(MOTOR_NAMES)

    @property
    def state_dim(self) -> int:
        return len(MOTOR_NAMES) * 2 + 6

    @property
    def asset_path(self) -> Path:
        return self._asset_path

    @property
    def mujoco_model(self) -> Any:
        return self._model

    @property
    def mujoco_data(self) -> Any:
        return self._data

    def reset(self, seed: int = 0) -> tuple[dict, dict]:
        self._mujoco.mj_resetData(self._model, self._data)
        self._step_count = 0
        self._data.qpos[self._qposadr] = self._home
        self._data.qvel[self._qveladr] = 0.0
        if self._has_actuators:
            self._data.ctrl[self._actuator_ids] = self._home
        self._reset_objects()
        self._mujoco.mj_forward(self._model, self._data)
        return self._obs(), {"seed": seed, "asset_path": str(self._asset_path)}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        qpos_target = self._normalized_action_to_qpos(action)
        if self._has_actuators:
            self._data.ctrl[self._actuator_ids] = qpos_target
            # Force the Shoulder Pan to move if it's being stubborn
            self._data.qpos[self._qposadr[0]] = qpos_target[0]
            for _ in range(5):
                self._mujoco.mj_step(self._model, self._data)
        else:
            self._data.qpos[self._qposadr] = qpos_target
            self._data.qvel[self._qveladr] = 0.0
            self._mujoco.mj_forward(self._model, self._data)
        self._step_count += 1
        truncated = self._step_count >= self._max_episode_steps
        info = {"success": False, "is_success": False, "step": self._step_count}
        return self._obs(), 0.0, False, truncated, info

    def close(self) -> None:
        self._renderer.close()
        if self._scene_path != self._asset_path:
            self._scene_path.unlink(missing_ok=True)

    def obs_to_batch(self, raw_obs: dict, device: torch.device | None = None) -> dict:
        img = to_float01(torch.from_numpy(raw_obs["pixels"]["front"]))
        img = img.permute(2, 0, 1).unsqueeze(0).contiguous()
        state = torch.from_numpy(raw_obs["agent_state"]).float().unsqueeze(0)
        if device is not None:
            img = img.to(device, non_blocking=True)
            state = state.to(device, non_blocking=True)
        return {
            "observation.images.front": img,
            "observation.state": state,
            "task": [self.task_description],
        }

    def get_frame(self, raw_obs: dict) -> np.ndarray:
        return raw_obs["pixels"]["front"]

    def is_success(self, info: dict) -> bool:
        return bool(info.get("success", False))

    def normalized_action_to_qpos(self, action: np.ndarray) -> np.ndarray:
        return self._normalized_action_to_qpos(action)

    @property
    def _has_actuators(self) -> bool:
        return bool(np.all(self._actuator_ids >= 0))

    def _obs(self) -> dict:
        # Get pixels from both cameras
        self._renderer.update_scene(self._data, camera="workspace_front")
        front_pixels = self._renderer.render().astype(np.uint8)

        self._renderer.update_scene(self._data, camera="wrist")
        wrist_pixels = self._renderer.render().astype(np.uint8)

        return {
            "pixels": {
                "front": front_pixels,
                "wrist": wrist_pixels,
            },
            "agent_state": self._state(),
        }

    def _state(self) -> np.ndarray:
        arm_qpos = self._data.qpos[self._qposadr]
        arm_qvel = self._data.qvel[self._qveladr]
        return np.concatenate(
            [
                self._qpos_to_normalized_action(arm_qpos),
                arm_qvel.astype(np.float32),
                np.zeros(6, dtype=np.float32),
            ]
        ).astype(np.float32)

    def _normalized_action_to_qpos(self, action: np.ndarray) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.shape[0] != len(MOTOR_NAMES):
            raise ValueError(f"Expected {len(MOTOR_NAMES)} actions, got {arr.shape[0]}")
        qpos = np.empty_like(arr)
        qpos[:5] = (
            self._ranges[:5, 0]
            + (np.clip(arr[:5], -100.0, 100.0) + 100.0) * 0.5 * (self._ranges[:5, 1] - self._ranges[:5, 0]) / 100.0
        )
        qpos[5] = self._ranges[5, 0] + np.clip(arr[5], 0.0, 100.0) * (self._ranges[5, 1] - self._ranges[5, 0]) / 100.0
        return qpos

    def _qpos_to_normalized_action(self, qpos: np.ndarray) -> np.ndarray:
        arr = np.asarray(qpos, dtype=np.float32).reshape(-1)
        out = np.empty_like(arr)
        out[:5] = ((arr[:5] - self._ranges[:5, 0]) / (self._ranges[:5, 1] - self._ranges[:5, 0])) * 200.0 - 100.0
        out[5] = ((arr[5] - self._ranges[5, 0]) / (self._ranges[5, 1] - self._ranges[5, 0])) * 100.0
        return out.astype(np.float32)

    def _reset_objects(self) -> None:
        if not self._objects_enabled or not self._object_joint_qpos:
            return

        object_poses = {
            "cube_free": np.array([-0.27, -0.055, 0.068, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "cylinder_free": np.array([-0.22, 0.07, 0.074, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "ball_free": np.array([-0.34, 0.055, 0.068, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        }
        for joint_name, base_pose in object_poses.items():
            qposadr = self._object_joint_qpos.get(joint_name)
            if qposadr is None:
                continue
            pose = base_pose.copy()
            pose[:2] += self._rng.uniform(-0.015, 0.015, size=2)
            self._data.qpos[qposadr : qposadr + 7] = pose


def _resolve_asset_path(asset_path: Path | str | None) -> Path:
    if asset_path is None:
        env_path = os.environ.get("SO101_MUJOCO_SCENE")
        asset_path = Path(env_path) if env_path else DEFAULT_LOCAL_SCENE_PATH

    resolved = Path(asset_path).expanduser().resolve()
    if resolved.is_file():
        return resolved

    raise FileNotFoundError(
        f"SO101 MuJoCo asset not found: {resolved}\n"
        f"Download or clone {SO_ARM100_URL}, then pass "
        "--asset-path path\\to\\SO-ARM100\\Simulation\\SO101\\scene.xml "
        "or set SO101_MUJOCO_SCENE."
    )


def _write_interactive_scene(asset_path: Path) -> Path:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".xml",
            prefix="so101_interactive_",
            dir=asset_path.parent,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(_interactive_scene_xml(asset_path))
            tmp_path = Path(tmp.name)
        return tmp_path
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise


def _interactive_scene_xml(asset_path: Path) -> str:
    # Read the provided asset (could be scene.xml or robot.xml)
    with open(asset_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    # Camera at the end of the arm, looking forward and down at the gripper
    wrist_cam = '<camera name="wrist" pos="0.0 0.05 -0.04" euler="-0.2 0 0" fovy="85"/>'

    # If this is a scene file that includes a robot, we need to find the robot file
    robot_include_match = re.search(r'<include\s+file="([^"]+)"', xml_content)

    if robot_include_match and "gripper" not in xml_content:
        # It's a scene file. Load the robot file instead for injection.
        robot_file_path = asset_path.parent / robot_include_match.group(1)
        if robot_file_path.exists():
            with open(robot_file_path, "r", encoding="utf-8") as f:
                robot_xml = f.read()
        else:
            robot_xml = xml_content
    else:
        robot_xml = xml_content

    # Inject camera into robot XML
    pattern = r'(<body\s+name="gripper"[^>]*>)'
    if re.search(pattern, robot_xml, re.IGNORECASE):
        robot_xml = re.sub(pattern, r"\1" + wrist_cam, robot_xml, flags=re.IGNORECASE)
        print("[DEBUG] Injected wrist camera into 'gripper' body.")
    else:
        pattern_wrist = r'(<body\s+name="wrist"[^>]*>)'
        if re.search(pattern_wrist, robot_xml, re.IGNORECASE):
            robot_xml = re.sub(pattern_wrist, r"\1" + wrist_cam, robot_xml, flags=re.IGNORECASE)
            print("[DEBUG] Injected wrist camera into 'wrist' body.")
        else:
            print("[DEBUG] WARNING: Could not find 'gripper' or 'wrist' body for camera injection!")

    # Boost actuator gains for better grasping
    robot_xml = robot_xml.replace('kp="250"', 'kp="800"').replace('kp="25"', 'kp="100"')

    # Save the modified robot to a temp file
    temp_robot = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8")
    temp_robot.write(robot_xml)
    temp_robot.close()
    temp_robot_path = Path(temp_robot.name)

    # Values for objects (scaled 0.6)
    return f"""
<mujoco model="so101_interactive_workspace">
  <include file="{temp_robot_path.as_posix()}"/>

  <option timestep="0.005" gravity="0 0 -9.81" iterations="50" tolerance="1e-10"/>

  <asset>
    <material name="workspace_table_mat" rgba="0.48 0.38 0.27 1" reflectance="0.1"/>
    <material name="object_red_mat" rgba="0.85 0.12 0.08 1" reflectance="0.2"/>
    <material name="object_blue_mat" rgba="0.05 0.28 0.85 1" reflectance="0.2"/>
    <material name="material_green_mat" rgba="0.15 0.65 0.22 1" reflectance="0.2"/>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <camera name="workspace_front" pos="0.35 -0.70 0.42" xyaxes="0.88 0.48 0 -0.23 0.42 0.88" fovy="52"/>
    <light name="workspace_key" pos="-0.25 -0.45 1.2" dir="0.2 0.3 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <light name="workspace_top" pos="0 0 2" dir="0 0 -1" diffuse="0.5 0.5 0.5"/>

    <geom name="floor" size="10 10 0.05" pos="0 0 0" type="plane" material="groundplane"/>
    <geom name="workspace_table" type="box" pos="-0.28 0 0.025" size="0.36 0.25 0.025"
          material="workspace_table_mat" friction="1.5 0.1 0.002" condim="3"/>

    <body name="red_cube" pos="-0.27 -0.055 0.068">
      <freejoint name="cube_free"/>
      <geom name="red_cube_geom" type="box" size="0.018 0.018 0.018" mass="0.08"
            material="object_red_mat" friction="2.5 0.1 0.002" condim="4" 
            solimp="0.99 0.99 0.01" solref="0.01 1"/>
    </body>

    <body name="blue_cylinder" pos="-0.22 0.070 0.074">
      <freejoint name="cylinder_free"/>
      <geom name="blue_cylinder_geom" type="cylinder" size="0.015 0.024" mass="0.1"
            material="object_blue_mat" friction="2.5 0.1 0.002" condim="4"
            solimp="0.99 0.99 0.01" solref="0.01 1"/>
    </body>

    <body name="green_ball" pos="-0.34 0.055 0.068">
      <freejoint name="ball_free"/>
      <geom name="green_ball_geom" type="sphere" size="0.018" mass="0.06"
            rgba="0.15 0.65 0.22 1" friction="2.5 0.1 0.002" condim="4"
            solimp="0.99 0.99 0.01" solref="0.01 1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()


def _freejoint_qpos_addresses(model: Any, mujoco: Any, joint_names: tuple[str, ...]) -> dict[str, int]:
    addresses = {}
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            addresses[joint_name] = int(model.jnt_qposadr[joint_id])
    return addresses


def _range_midpoint(ranges: np.ndarray) -> np.ndarray:
    return ((ranges[:, 0] + ranges[:, 1]) * 0.5).astype(np.float32)


def _first_camera_name(model: Any, mujoco: Any) -> str | None:
    if model.ncam <= 0:
        return None
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, 0)
