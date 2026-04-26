"""Record SO-101 leader-arm teleoperation against a real MuJoCo/URDF asset.

Examples:
    # Hardware-free smoke recording against a local SO-ARM100 scene
    python scripts/record_so101_mujoco.py --source scripted --asset-path third_party/SO-ARM100/...

    # Use a physical SO-101 leader arm on Windows
    python scripts/record_so101_mujoco.py --leader-port COM6 --asset-path third_party/SO-ARM100/...
"""

from __future__ import annotations

import json
import logging
try:
    import msvcrt
except ImportError:
    msvcrt = None
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import typer

from vla.constants import PREPROCESSED_DIR
from vla.envs.so101_mujoco import MOTOR_NAMES, SO101MujocoEnv

logger = logging.getLogger(__name__)


@dataclass
class TeleopActionMapping:
    """Maps raw leader normalized values into simulator normalized actions."""

    neutral: np.ndarray
    source_indices: np.ndarray
    sign: np.ndarray
    scale: np.ndarray
    output_neutral: np.ndarray

    @classmethod
    def identity(cls) -> TeleopActionMapping:
        return cls(
            neutral=np.zeros(len(MOTOR_NAMES), dtype=np.float32),
            source_indices=_default_source_indices(),
            sign=_default_action_signs(),
            scale=np.ones(len(MOTOR_NAMES), dtype=np.float32),
            output_neutral=np.zeros(len(MOTOR_NAMES), dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path | None) -> TeleopActionMapping:
        if path is None:
            return cls.identity()
        with path.open() as f:
            data = json.load(f)
        mapping = cls(
            neutral=np.asarray(data["neutral"], dtype=np.float32),
            source_indices=np.asarray(data.get("source_indices", _default_source_indices()), dtype=np.int32),
            sign=np.asarray(data.get("sign", [1.0] * len(MOTOR_NAMES)), dtype=np.float32),
            scale=np.asarray(data.get("scale", [1.0] * len(MOTOR_NAMES)), dtype=np.float32),
            output_neutral=np.asarray(data.get("output_neutral", [0.0] * len(MOTOR_NAMES)), dtype=np.float32),
        )
        mapping._validate()
        logger.info("Loaded action mapping from %s", path)
        return mapping

    def apply(self, action: np.ndarray) -> np.ndarray:
        arranged = np.asarray(action, dtype=np.float32)[self.source_indices]
        mapped = (arranged - self.neutral) * self.sign * self.scale + self.output_neutral
        mapped[:5] = np.clip(mapped[:5], -100.0, 100.0)
        mapped[5] = np.clip(mapped[5], 0.0, 100.0)
        return mapped.astype(np.float32)

    def to_json_dict(self) -> dict:
        return {
            "motor_names": list(MOTOR_NAMES),
            "neutral": self.neutral.tolist(),
            "source_indices": self.source_indices.tolist(),
            "sign": self.sign.tolist(),
            "scale": self.scale.tolist(),
            "output_neutral": self.output_neutral.tolist(),
        }

    def _validate(self) -> None:
        for name, arr in {
            "neutral": self.neutral,
            "source_indices": self.source_indices,
            "sign": self.sign,
            "scale": self.scale,
            "output_neutral": self.output_neutral,
        }.items():
            if arr.shape != (len(MOTOR_NAMES),):
                raise ValueError(f"Action mapping {name!r} must have shape ({len(MOTOR_NAMES)},), got {arr.shape}")
        if sorted(self.source_indices.tolist()) != list(range(len(MOTOR_NAMES))):
            raise ValueError(
                "Action mapping 'source_indices' must be a permutation of "
                f"0..{len(MOTOR_NAMES) - 1}, got {self.source_indices.tolist()}"
            )


class ActionSource(Protocol):
    def connect(self) -> None: ...
    def get_action(self) -> np.ndarray: ...
    def close(self) -> None: ...


class LivePreview:
    """Native MuJoCo passive viewer."""

    def __init__(
        self,
        enabled: bool,
        env: SO101MujocoEnv,
        title: str = "SO-101 MuJoCo",
    ) -> None:
        self.enabled = enabled
        self.title = title
        self._viewer = None
        if enabled:
            import mujoco.viewer

            logger.info("Opening MuJoCo viewer window: %s", title)
            self._viewer = mujoco.viewer.launch_passive(env.mujoco_model, env.mujoco_data)

    def show(self, env: SO101MujocoEnv) -> bool:
        del env
        if not self.enabled or self._viewer is None:
            return True
        if not self._viewer.is_running():
            return False
        self._viewer.sync()
        return True

    @property
    def is_alive(self) -> bool:
        if not self.enabled or self._viewer is None:
            return True
        return self._viewer.is_running()

    def close(self) -> None:
        if self.enabled and self._viewer is not None:
            self._viewer.close()


class ScriptedActionSource:
    """Simple deterministic source for testing the recorder without hardware."""

    def __init__(self) -> None:
        self._t = 0

    def connect(self) -> None:
        return

    def get_action(self) -> np.ndarray:
        self._t += 1
        t = self._t / 30.0
        return np.array(
            [
                35.0 * np.sin(t * 0.9),
                -35.0 + 16.0 * np.sin(t * 0.7),
                30.0 + 22.0 * np.sin(t * 0.5),
                -20.0 + 12.0 * np.sin(t * 1.1),
                25.0 * np.sin(t * 0.4),
                80.0 if (self._t // 45) % 2 == 0 else 25.0,
            ],
            dtype=np.float32,
        )

    def close(self) -> None:
        return


class SO101LeaderActionSource:
    """Reads SO-101 leader positions directly through the Feetech SDK.

    This intentionally avoids importing ``lerobot.teleoperators`` at runtime.
    The installed LeRobot version in this Windows environment currently has a
    circular import through ``lerobot.processor``; direct SDK access is enough
    for read-only leader teleoperation.
    """

    def __init__(
        self,
        port: str,
        robot_id: str,
        baudrate: int = 1_000_000,
        calibration_path: Path | None = None,
        tolerate_packet_errors: bool = True,
    ) -> None:
        del robot_id
        import scservo_sdk as scs

        self._scs = scs
        self._port = port
        self._baudrate = baudrate
        self._handler = scs.PortHandler(port)
        self._packet = scs.PacketHandler(0)
        self._calibration = _load_calibration(calibration_path)
        self._motor_ids = _motor_ids_from_calibration(self._calibration)
        self._is_connected = False
        self._tolerate_packet_errors = tolerate_packet_errors
        self._warned_packet_errors: set[tuple[str, int]] = set()

    def connect(self) -> None:
        if not self._handler.openPort():
            raise RuntimeError(f"Failed to open leader arm port {self._port!r}")
        if not self._handler.setBaudRate(self._baudrate):
            self._handler.closePort()
            raise RuntimeError(f"Failed to set leader arm baudrate to {self._baudrate}")
        self._is_connected = True

    def get_action(self) -> np.ndarray:
        raw = {}
        for name in MOTOR_NAMES:
            motor_id = self._motor_ids[name]
            value, comm_result, packet_error = self._packet.read2ByteTxRx(self._handler, motor_id, 56)
            if comm_result != self._scs.COMM_SUCCESS:
                msg = self._packet.getTxRxResult(comm_result)
                raise RuntimeError(f"Failed to read {name} motor id={motor_id}: {msg}")
            if packet_error:
                msg = self._packet.getRxPacketError(packet_error)
                if not self._tolerate_packet_errors:
                    raise RuntimeError(f"Packet error while reading {name} motor id={motor_id}: {msg}")
                warn_key = (name, int(packet_error))
                if warn_key not in self._warned_packet_errors:
                    logger.warning(
                        "Packet status while reading %s motor id=%d: %s. "
                        "Continuing because --tolerate-packet-errors is enabled.",
                        name,
                        motor_id,
                        msg,
                    )
                    self._warned_packet_errors.add(warn_key)
            raw[name] = int(value)
        normalized = [_normalize_motor(name, raw[name], self._calibration) for name in MOTOR_NAMES]
        return np.array(normalized, dtype=np.float32)

    def close(self) -> None:
        if self._is_connected:
            self._handler.closePort()
            self._is_connected = False


def _default_calibration_path(robot_id: str) -> Path | None:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    base = Path(os.environ.get("HF_LEROBOT_CALIBRATION", hf_home / "lerobot" / "calibration"))
    candidates = [
        base / "teleoperators" / "so_leader" / f"{robot_id}.json",
        base / "teleoperators" / "so101_leader" / f"{robot_id}.json",
        base / "robots" / "so_follower" / f"{robot_id}.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_calibration(calibration_path: Path | None) -> dict[str, dict] | None:
    if calibration_path is None:
        return None
    with calibration_path.open() as f:
        data = json.load(f)
    missing = [name for name in MOTOR_NAMES if name not in data]
    if missing:
        raise ValueError(f"Calibration file {calibration_path} is missing motors: {missing}")
    logger.info("Loaded leader calibration from %s", calibration_path)
    return data


def _motor_ids_from_calibration(calibration: dict[str, dict] | None) -> dict[str, int]:
    if calibration is None:
        return {name: idx for idx, name in enumerate(MOTOR_NAMES, start=1)}
    return {name: int(calibration[name]["id"]) for name in MOTOR_NAMES}


def _normalize_motor(name: str, raw_value: int, calibration: dict[str, dict] | None) -> float:
    if calibration is None:
        if name == "gripper":
            return float(np.clip(raw_value / 4095.0 * 100.0, 0.0, 100.0))
        return float(np.clip(raw_value / 4095.0 * 200.0 - 100.0, -100.0, 100.0))

    cal = calibration[name]
    min_value = float(cal["range_min"])
    max_value = float(cal["range_max"])
    if max_value == min_value:
        raise ValueError(f"Invalid calibration for {name}: range_min == range_max")
    bounded = float(np.clip(raw_value, min_value, max_value))
    drive_mode = int(cal.get("drive_mode", 0))
    if name == "gripper":
        norm = (bounded - min_value) / (max_value - min_value) * 100.0
        return float(100.0 - norm if drive_mode else norm)
    norm = (bounded - min_value) / (max_value - min_value) * 200.0 - 100.0
    return float(-norm if drive_mode else norm)


def _resize_image(image: np.ndarray, image_size: int) -> torch.Tensor:
    if image.shape[0] != image_size or image.shape[1] != image_size:
        raise ValueError(f"Expected rendered image size {image_size}x{image_size}, got {image.shape[:2]}")
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


def _record_episode(
    env: SO101MujocoEnv,
    source: ActionSource,
    action_mapping: TeleopActionMapping,
    episode_seed: int,
    fps: int,
    image_size: int,
    preview: LivePreview,
) -> dict | None:
    if msvcrt:
        print(f"\n[EPISODE {episode_seed}] Previewing. Press SPACE in terminal to START, or ESC to skip.")
        while True:
            t0 = time.perf_counter()
            raw = source.get_action()
            mapped = action_mapping.apply(raw)
            env.step(mapped)
            if not preview.show(env):
                return None
            if msvcrt.kbhit():
                key = msvcrt.getch()
                if key == b" ":
                    break
                if key == b"\x1b":
                    return None
            time.sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))

    logger.info("--- RECORDING EPISODE %d ---", episode_seed)
    obs, _info = env.reset(seed=episode_seed)
    images_front: list[torch.Tensor] = []
    images_wrist: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []

    for _ in range(env.max_episode_steps):
        t0 = time.perf_counter()
        raw_action = source.get_action()
        action = action_mapping.apply(raw_action)

        if not preview.show(env):
            break

        if msvcrt and msvcrt.kbhit():
            if msvcrt.getch() == b" ":
                logger.info("Early stop requested via Space.")
                break

        images_front.append(_resize_image(obs["pixels"]["front"], image_size).unsqueeze(0))
        images_wrist.append(_resize_image(obs["pixels"]["wrist"], image_size).unsqueeze(0))
        states.append(torch.from_numpy(obs["agent_state"]).float())
        actions.append(torch.from_numpy(action).float())

        obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
        time.sleep(max(1.0 / fps - (time.perf_counter() - t0), 0.0))

    logger.info("--- FINISHED EPISODE %d ---", episode_seed)
    return {
        "images": torch.stack(images_front, dim=0),
        "images_wrist": torch.stack(images_wrist, dim=0),
        "states": torch.stack(states, dim=0),
        "actions": torch.stack(actions, dim=0),
        "instruction": env.task_description,
    }


def _print_diagnostics(
    env: SO101MujocoEnv,
    source: ActionSource,
    action_mapping: TeleopActionMapping,
    samples: int,
    fps: int,
    preview: LivePreview,
) -> None:
    logger.info("Printing %d leader/mapping diagnostic sample(s).", samples)
    logger.info("motor_names: %s", list(MOTOR_NAMES))
    logger.info("source_indices: %s", action_mapping.source_indices.tolist())
    logger.info("sign: %s", action_mapping.sign.tolist())
    logger.info("neutral: %s", np.round(action_mapping.neutral, 2).tolist())
    for idx in range(samples):
        raw = source.get_action()
        mapped = action_mapping.apply(raw)
        env.step(mapped)
        preview.show(env)
        qpos = env.normalized_action_to_qpos(mapped)
        logger.info(
            "sample=%d raw=%s arranged=%s mapped=%s qpos_rad=%s",
            idx + 1,
            np.round(raw, 2).tolist(),
            np.round(raw[action_mapping.source_indices], 2).tolist(),
            np.round(mapped, 2).tolist(),
            np.round(qpos, 3).tolist(),
        )
        time.sleep(1.0 / fps)


def _make_env(
    asset_path: Path | None,
    image_size: int,
    max_steps: int,
    instruction: str,
    objects: bool,
    seed: int,
    base_pos: tuple[float, float, float] | None = None,
    base_quat: tuple[float, float, float, float] | None = None,
) -> SO101MujocoEnv:
    return SO101MujocoEnv(
        asset_path=asset_path,
        image_size=image_size,
        max_episode_steps=max_steps,
        instruction=instruction,
        objects=objects,
        seed=seed,
        base_pos=base_pos,
        base_quat=base_quat,
    )


def _capture_neutral_mapping(
    source: ActionSource,
    output_path: Path,
    samples: int = 60,
    fps: int = 30,
) -> None:
    logger.info("Hold the leader arm in the desired digital-twin neutral pose.")
    logger.info("Capturing %d samples...", samples)
    values = []
    for _ in range(samples):
        values.append(source.get_action())
        time.sleep(1.0 / fps)
    source_indices = _default_source_indices()
    neutral = np.mean(np.stack(values, axis=0), axis=0)[_default_source_indices()].astype(np.float32)
    mapping = TeleopActionMapping(
        neutral=neutral,
        source_indices=source_indices,
        sign=_default_action_signs(),
        scale=np.ones(len(MOTOR_NAMES), dtype=np.float32),
        output_neutral=np.array([0.0, -35.0, 35.0, -20.0, 0.0, 60.0], dtype=np.float32),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(mapping.to_json_dict(), f, indent=2)
    logger.info("Saved neutral action mapping to %s", output_path)
    logger.info("If a joint moves in the opposite direction, edit its sign in that file.")
    logger.info("If a leader joint controls the wrong sim joint, edit source_indices in that file.")


def _default_source_indices() -> np.ndarray:
    # Target order is MOTOR_NAMES. Source order comes from the leader read.
    return np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)


def _default_action_signs() -> np.ndarray:
    # Applied after source_indices. Keep this file-editable via --capture-neutral.
    return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)


def _make_source(
    source: str,
    leader_port: str | None,
    leader_id: str,
    leader_baudrate: int,
    calibration_path: Path | None,
    tolerate_packet_errors: bool,
) -> ActionSource:
    if source == "scripted":
        return ScriptedActionSource()
    if source == "leader":
        if not leader_port:
            raise typer.BadParameter("--leader-port is required when --source=leader")
        return SO101LeaderActionSource(
            leader_port,
            leader_id,
            baudrate=leader_baudrate,
            calibration_path=calibration_path or _default_calibration_path(leader_id),
            tolerate_packet_errors=tolerate_packet_errors,
        )
    raise typer.BadParameter(f"Unknown source {source!r}")


def main(
    source: str = typer.Option("leader", "--source", help="Action source: leader or scripted"),
    leader_port: str = typer.Option(None, "--leader-port", help="Serial port for the SO-101 leader arm"),
    leader_id: str = typer.Option("leader", "--leader-id"),
    leader_baudrate: int = typer.Option(1_000_000, "--leader-baudrate"),
    leader_calibration: Path = typer.Option(None, "--leader-calibration", path_type=Path),
    tolerate_packet_errors: bool = typer.Option(
        True,
        "--tolerate-packet-errors/--strict-packet-errors",
        help="Continue when a servo returns a status warning but the position read succeeds.",
    ),
    output: Path = typer.Option(None, "--output", "-o", path_type=Path),
    action_map: Path = typer.Option(None, "--action-map", path_type=Path),
    capture_neutral: Path = typer.Option(None, "--capture-neutral", path_type=Path),
    diagnostics: bool = typer.Option(False, "--diagnostics", help="Print leader/action/qpos mapping samples and exit."),
    diagnostics_samples: int = typer.Option(20, "--diagnostics-samples", min=1),
    episodes: int = typer.Option(10, "--episodes", "-n", min=1),
    max_steps: int = typer.Option(250, "--max-steps", min=1),
    image_size: int = typer.Option(256, "--image-size", min=32),
    fps: int = typer.Option(30, "--fps", min=1),
    instruction: str = typer.Option("teleoperate the SO101 arm", "--instruction"),
    asset_path: Path = typer.Option(
        None,
        "--asset-path",
        "--scene-path",
        "--urdf-path",
        path_type=Path,
        help="Path to TheRobotStudio SO101 scene.xml, calibrated MJCF, or URDF.",
    ),
    objects: bool = typer.Option(True, "--objects/--no-objects", help="Add table and interactable objects."),
    render: bool = typer.Option(True, "--render/--no-render", help="Show a live MuJoCo viewer window"),
    seed: int = typer.Option(42, "--seed"),
    base_pos: str = typer.Option(None, "--base-pos", help="Base position as 'x,y,z'"),
    base_quat: str = typer.Option("0,0,0,1", "--base-quat", help="Base quaternion as 'w,x,y,z'"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if output is None:
        output = PREPROCESSED_DIR / "so101_mujoco.pt"
    output.parent.mkdir(parents=True, exist_ok=True)

    b_pos = tuple(map(float, base_pos.split(","))) if base_pos else None
    b_quat = tuple(map(float, base_quat.split(","))) if base_quat else None

    env = _make_env(asset_path, image_size, max_steps, instruction, objects, seed, b_pos, b_quat)
    action_source = _make_source(
        source,
        leader_port,
        leader_id,
        leader_baudrate,
        leader_calibration,
        tolerate_packet_errors,
    )
    action_mapping = TeleopActionMapping.load(action_map)
    preview = LivePreview(render, env)

    episodes_out = []
    try:
        action_source.connect()
        if diagnostics:
            _print_diagnostics(env, action_source, action_mapping, diagnostics_samples, fps, preview)
            return
        if capture_neutral is not None:
            _capture_neutral_mapping(action_source, capture_neutral, fps=fps)
            return
        for ep_idx in range(episodes):
            ep_seed = seed + ep_idx
            data = _record_episode(env, action_source, action_mapping, ep_seed, fps, image_size, preview)
            if data is not None:
                episodes_out.append(data)
            else:
                if not preview.is_alive:
                    break
                logger.info("Episode skipped.")
    finally:
        preview.close()
        action_source.close()
        env.close()

    if not episodes_out:
        logger.info("No episodes recorded. Exiting.")
        return

    torch.save(
        {
            "episodes": episodes_out,
            "metadata": {
                "env_id": "SO101Mujoco-v0",
                "skill": "SO101Mujoco-v0",
                "simulator": "so101_mujoco",
                "asset_path": str(env.asset_path),
                "num_episodes": len(episodes_out),
                "action_dim": len(MOTOR_NAMES),
                "state_dim": int(episodes_out[0]["states"].shape[-1]),
                "image_size": image_size,
                "num_cameras": 1,
                "instruction": instruction,
                "control_mode": "so101_normalized_joint_position",
                "source": source,
                "objects": objects,
                "motor_names": list(MOTOR_NAMES),
                "action_mapping": action_mapping.to_json_dict(),
            },
        },
        output,
    )
    logger.info("Saved %d episode(s) to %s", len(episodes_out), output)


if __name__ == "__main__":
    typer.run(main)
