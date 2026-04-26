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
        if objects:
            self._scene_path, self._robot_path = _write_interactive_scene(self._asset_path)
        else:
            self._scene_path = self._asset_path
            self._robot_path = None
        self._model = mujoco.MjModel.from_xml_path(str(self._scene_path))
        self._data = mujoco.MjData(self._model)

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

        if self._has_actuators:
            for actuator_id, joint_id in zip(self._actuator_ids, self._joint_ids, strict=True):
                if actuator_id >= 0:
                    self._model.actuator_ctrlrange[actuator_id] = self._model.jnt_range[joint_id]
                    self._model.actuator_ctrllimited[actuator_id] = 1

            # Gripper force limits (Robosuite scale refined for SO101)
            gripper_actuator_id = int(self._actuator_ids[-1])
            if gripper_actuator_id >= 0:
                self._model.actuator_forcerange[gripper_actuator_id] = np.array([-30.0, 30.0])
                self._model.actuator_forcelimited[gripper_actuator_id] = 1

        for camera in ("workspace_front", "wrist"):
            if mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
                raise ValueError(f"Missing camera: {camera}")

        self._renderer = mujoco.Renderer(self._model, height=image_size, width=image_size)
        self._image_size = image_size
        self._max_episode_steps = max_episode_steps
        self._instruction = instruction
        self._objects_enabled = objects
        self._rng = np.random.default_rng(seed)
        self._step_count = 0

        self._ranges = self._model.jnt_range[self._joint_ids].astype(np.float32)
        self._home = _range_midpoint(self._ranges)
        self._camera_name = _first_camera_name(self._model, mujoco)

        self._object_joint_qpos = _freejoint_qpos_addresses(
            self._model,
            mujoco,
            ("cube_free", "cylinder_free", "wood_free", "ball_free", "plate_free", "bowl_free"),
        )

        base_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_body_id >= 0:
            if base_pos is not None:
                self._model.body_pos[base_body_id] = base_pos
            if base_quat is not None:
                self._model.body_quat[base_body_id] = base_quat

        self._mujoco.mj_forward(self._model, self._data)

        # Debug gripper range to verify teleop direction
        print(f"[DEBUG] Case: 30g Cube Testing")
        print(f"[DEBUG] Gripper Joint Range: {self._ranges[5]}")
        closed_target = self._normalized_action_to_qpos(np.array([0, 0, 0, 0, 0, 0]))[5]
        open_target = self._normalized_action_to_qpos(np.array([0, 0, 0, 0, 0, 100]))[5]
        print(f"[DEBUG] Closed Target (action=0): {closed_target:.4f}")
        print(f"[DEBUG] Open Target (action=100): {open_target:.4f}")

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

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._mujoco.mj_resetData(self._model, self._data)
        self._step_count = 0
        self._data.qpos[self._qposadr] = self._home
        self._data.qvel[self._qveladr] = 0.0

        if self._has_actuators:
            self._data.ctrl[self._actuator_ids] = self._home

        self._reset_objects()
        self._mujoco.mj_forward(self._model, self._data)

        # Allow objects to settle before the first observation
        for _ in range(25):
            self._mujoco.mj_step(self._model, self._data)

        return self._obs(), {"seed": seed, "asset_path": str(self._asset_path)}

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        qpos_target = self._normalized_action_to_qpos(action)
        if self._has_actuators:
            self._data.ctrl[self._actuator_ids] = qpos_target
            # 30 Hz control loop (15 * 0.002s) - smoother for teleop lift
            for _ in range(15):
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
        if self._robot_path is not None:
            self._robot_path.unlink(missing_ok=True)

    def obs_to_batch(self, raw_obs: dict, device: torch.device | None = None) -> dict:
        front = to_float01(torch.from_numpy(raw_obs["pixels"]["front"]))
        wrist = to_float01(torch.from_numpy(raw_obs["pixels"]["wrist"]))
        front = front.permute(2, 0, 1).unsqueeze(0).contiguous()
        wrist = wrist.permute(2, 0, 1).unsqueeze(0).contiguous()
        state = torch.from_numpy(raw_obs["agent_state"]).float().unsqueeze(0)

        if device is not None:
            front = front.to(device, non_blocking=True)
            wrist = wrist.to(device, non_blocking=True)
            state = state.to(device, non_blocking=True)

        return {
            "observation.images.front": front,
            "observation.images.wrist": wrist,
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

    def debug_contacts(self) -> None:
        """Print current contacts for debugging grasp issues."""
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            g1 = self._mujoco.mj_id2name(self._model, self._mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            g2 = self._mujoco.mj_id2name(self._model, self._mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            if g1 and g2 and ("cube" in g1 or "cube" in g2 or "pad" in g1 or "pad" in g2 or "jaw" in g1 or "jaw" in g2):
                print(f"[CONTACT {i}] {g1} <-> {g2} dist={contact.dist:.5f}")

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
        qvel_normalized = np.clip(arm_qvel, -10.0, 10.0) / 10.0
        return np.concatenate(
            [
                self._qpos_to_normalized_action(arm_qpos) / 100.0,
                qvel_normalized.astype(np.float32),
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
            "cube_free": np.array([-0.25, -0.08, 0.02, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "cylinder_free": np.array([-0.20, 0.08, 0.03, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "wood_free": np.array([-0.35, 0.05, 0.02, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "ball_free": np.array([-0.32, -0.10, 0.018, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "plate_free": np.array([-0.45, -0.05, 0.005, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "bowl_free": np.array([-0.40, 0.10, 0.015, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
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


def _write_interactive_scene(asset_path: Path) -> tuple[Path, Path | None]:
    tmp_path = None
    tmp_robot_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix="so101_interactive_",
            dir=asset_path.parent,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            xml_str, tmp_robot_path = _interactive_scene_xml(asset_path)
            tmp.write(xml_str)
            tmp_path = Path(tmp.name)
        return tmp_path, tmp_robot_path
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if tmp_robot_path is not None:
            tmp_robot_path.unlink(missing_ok=True)
        raise


def _interactive_scene_xml(asset_path: Path) -> tuple[str, Path | None]:
    # Read the provided asset (could be scene.xml or robot.xml)
    with open(asset_path, "r", encoding="utf-8") as f:
        xml_content = f.read()

    # Camera at the end of the arm, looking forward and down at the gripper
    wrist_cam = '<camera name="wrist" pos="0.0 0.05 -0.07" euler="-0.85 0 0" fovy="70"/>'

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

    # Full Robot collision baseline (Moderate friction)
    robot_xml, n_collision = re.subn(
        r'<geom\s+group="3"\s*/>',
        '<geom group="3" friction="0.8 0.005 0.0001" condim="3" solimp="0.9 0.95 0.001" solref="0.02 1" margin="0.0005" gap="0"/>',
        robot_xml,
    )
    print(f"[DEBUG] Patched {n_collision} robot collision geoms with moderate friction.")

    # Inject Robosuite-style High-Friction Primitive Pads into finger bodies
    # Static Finger Pad (Gripper Body)
    static_pad = '<geom name="static_pad" type="box" size="0.008 0.005 0.01" pos="-0.016 0 -0.095" friction="2.0 0.05 0.0001" condim="4" solref="0.01 0.5" solimp="0.9 0.95 0.001" margin="0.0005" gap="0" contype="1" conaffinity="1" group="4"/>'
    pattern_gripper = r'(<body\s+name="gripper"[^>]*>)'
    robot_xml = re.sub(pattern_gripper, r"\1" + static_pad, robot_xml, flags=re.IGNORECASE)

    # Moving Finger Pad (Moving Jaw Body)
    moving_pad = '<geom name="moving_pad" type="box" size="0.008 0.015 0.01" pos="0 -0.03 0.02" friction="2.0 0.05 0.0001" condim="4" solref="0.01 0.5" solimp="0.9 0.95 0.001" margin="0.0005" gap="0" contype="1" conaffinity="1" group="4"/>'
    pattern_jaw = r'(<body\s+name="moving_jaw_so101_v1"[^>]*>)'
    robot_xml = re.sub(pattern_jaw, r"\1" + moving_pad, robot_xml, flags=re.IGNORECASE)

    # Robosuite style actuator settings (KP 80 for SO101 scale)
    actuator_xml = """<actuator>
    <position name="shoulder_pan" joint="shoulder_pan" kp="250"/>
    <position name="shoulder_lift" joint="shoulder_lift" kp="250"/>
    <position name="elbow_flex" joint="elbow_flex" kp="250"/>
    <position name="wrist_flex" joint="wrist_flex" kp="150"/>
    <position name="wrist_roll" joint="wrist_roll" kp="100"/>
    <position name="gripper" joint="gripper" kp="80" forcerange="-30 30"/>
  </actuator>"""

    robot_xml, n_actuators = re.subn(r"<actuator>.*?</actuator>", actuator_xml, robot_xml, flags=re.DOTALL)
    if n_actuators == 0:
        robot_xml = robot_xml.replace("</mujoco>", f"{actuator_xml}\n</mujoco>")

    # Save the modified robot to a temp file next to the original asset to preserve relative mesh paths
    temp_robot = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", prefix="so101_robot_", dir=asset_path.parent, delete=False, encoding="utf-8"
    )
    temp_robot.write(robot_xml)
    temp_robot.close()
    temp_robot_path = Path(temp_robot.name)

    # Values for objects (scaled 0.6)
    xml_str = f"""
<mujoco model="so101_interactive_workspace">
  <compiler angle="radian"/>
  <include file="{temp_robot_path.as_posix()}"/>

  <option
    timestep="0.002"
    gravity="0 0 -9.81"
    integrator="implicitfast"
    cone="elliptic"
    impratio="20"
    iterations="100"
    tolerance="1e-10"
    density="1.2"
    viscosity="0.00002"
  />
  <size nconmax="5000" njmax="5000"/>

  <asset>
    <material name="workspace_table_mat" rgba="0.4 0.3 0.2 1" reflectance="0.1"/>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.1 0.1 0.1" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.1"/>
    
    <!-- Object Textures -->
    <texture name="wood_tex" type="2d" builtin="flat" rgb1="0.6 0.4 0.2" width="128" height="128"/>
    <material name="wood_mat" texture="wood_tex" reflectance="0.1"/>
    <material name="metal_mat" rgba="0.7 0.7 0.7 1" reflectance="0.8" shininess="0.9"/>
    <material name="plastic_red_mat" rgba="0.8 0.1 0.1 1" reflectance="0.1"/>
    <material name="plastic_blue_mat" rgba="0.1 0.1 0.8 1" reflectance="0.1"/>
    <material name="plastic_green_mat" rgba="0.1 0.8 0.1 1" reflectance="0.1"/>
  </asset>

  <worldbody>
    <camera name="workspace_front" pos="0.30 -0.55 0.35" xyaxes="0.88 0.48 0 -0.23 0.42 0.88" fovy="58"/>
    <light name="workspace_key" pos="-0.5 -1.0 1.5" dir="0.5 1.0 -1" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="true"/>
    <light name="workspace_fill" pos="0.5 0.5 1.5" dir="-0.5 -0.5 -1" diffuse="0.3 0.3 0.3"/>
    <light name="workspace_top" pos="0 0 3" dir="0 0 -1" diffuse="0.4 0.4 0.4" ambient="0.2 0.2 0.2"/>

    <geom name="floor" size="10 10 0.05" pos="0 0 0" type="plane" material="groundplane"/>
    <geom name="workspace_table" type="box" pos="-0.28 0 -0.025" size="0.40 0.30 0.025"
          material="workspace_table_mat" friction="0.9 0.01 0.001" condim="3"
          solimp="0.9 0.97 0.001" solref="0.01 1" contype="1" conaffinity="1"/>

    <!-- Object 1: Large Red Cube (Plastic) -->
    <body name="red_cube" pos="-0.25 -0.08 0.02">
      <freejoint name="cube_free"/>
      <geom name="red_cube_geom" type="box" size="0.02 0.02 0.02" mass="0.03"
            material="plastic_red_mat" friction="1.0 0.005 0.0001" condim="4" 
            solimp="0.9 0.95 0.001" solref="0.02 1" contype="1" conaffinity="1"/>
    </body>

    <!-- Object 2: Blue Cylinder (Metal-like) -->
    <body name="blue_cylinder" pos="-0.20 0.08 0.03">
      <freejoint name="cylinder_free"/>
      <geom name="blue_cylinder_geom" type="cylinder" size="0.018 0.03" mass="0.15"
            material="metal_mat" friction="1.0 0.005 0.0001" condim="4" 
            solimp="0.9 0.95 0.001" solref="0.02 1" contype="1" conaffinity="1"/>
    </body>

    <!-- Object 3: Wooden Block (Box) -->
    <body name="wood_block" pos="-0.35 0.05 0.02">
      <freejoint name="wood_free"/>
      <geom name="wood_block_geom" type="box" size="0.025 0.015 0.02" mass="0.08"
            material="wood_mat" friction="1.0 0.005 0.0001" condim="4" 
            solimp="0.9 0.95 0.001" solref="0.02 1" contype="1" conaffinity="1"/>
    </body>

    <!-- Object 4: Green Ball -->
    <body name="green_ball" pos="-0.32 -0.10 0.018">
      <freejoint name="ball_free"/>
      <geom name="green_ball_geom" type="sphere" size="0.018" mass="0.05"
            material="plastic_green_mat" friction="1.0 0.005 0.0001" condim="4" 
            solimp="0.9 0.95 0.001" solref="0.02 1" contype="1" conaffinity="1"/>
    </body>

    <!-- Object 5: Plate -->
    <body name="plate" pos="-0.45 -0.05 0.005">
      <freejoint name="plate_free"/>
      <geom name="plate_geom" type="cylinder" size="0.05 0.005" mass="0.2"
            material="metal_mat" friction="1.0 0.005 0.0001" condim="4" 
            solimp="0.9 0.95 0.001" solref="0.02 1" contype="1" conaffinity="1"/>
    </body>

    <!-- Object 6: Bowl -->
    <body name="bowl" pos="-0.40 0.10 0.015">
      <freejoint name="bowl_free"/>
      <geom name="bowl_geom" type="cylinder" size="0.04 0.015" mass="0.12" material="plastic_blue_mat" 
            friction="1.0 0.005 0.0001" condim="4" solimp="0.9 0.95 0.001" solref="0.02 1" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
</mujoco>
""".strip()
    return xml_str, temp_robot_path


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
