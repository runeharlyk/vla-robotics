"""
Preprocess raw ManiSkill demos into VLA-ready .pt files in data/preprocessed.

Replays each trajectory in simulation to collect RGB, state, and actions,
then saves a single .pt file with everything needed to train a VLA.

Usage:
    uv run python scripts/preprocess_data.py
    uv run python scripts/preprocess_data.py --skill PickCube-v1 --max-episodes 100
"""
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
import typer
from PIL import Image
from tqdm import tqdm

from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import common, io_utils

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
DEFAULT_SKILL = "PickCube-v1"
PICK_CUBE_INSTRUCTION = "pick up the red cube and move it to the green goal"
_warned_camera_fallback = False
_warned_render_fallback = False


def find_trajectory_files(raw_dir: Path, env_id: str) -> tuple[Path, Path]:
    env_dir = raw_dir / env_id
    if not env_dir.exists():
        raise FileNotFoundError(f"No directory {env_dir}. Run download_data.py first.")
    for path in env_dir.rglob("trajectory*.h5"):
        json_path = path.with_suffix(".json")
        if json_path.exists():
            return path, json_path
    raise FileNotFoundError(f"No trajectory.h5 + .json under {env_dir}")


def extract_rgb_from_render(env) -> np.ndarray:
    frame = env.render()
    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    return np.asarray(frame, dtype=np.uint8)


def ensure_rgb_hwc(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {arr.shape}")
    if arr.shape[-1] == 3:
        return arr.astype(np.uint8)
    if arr.shape[0] == 3:
        return np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    raise ValueError(f"Unsupported image shape {arr.shape}")


def extract_sensor_rgb_views(obs: dict) -> list[np.ndarray]:
    sensor_data = obs.get("sensor_data", {})
    if not isinstance(sensor_data, dict):
        return []
    arrays = []
    for cam_data in sensor_data.values():
        if not isinstance(cam_data, dict) or "rgb" not in cam_data:
            continue
        rgb = cam_data["rgb"]
        if hasattr(rgb, "cpu"):
            rgb = rgb.cpu().numpy()
        arr = np.asarray(rgb)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3:
            arrays.append(ensure_rgb_hwc(arr))
    return arrays


def extract_render_rgb_views(env) -> list[np.ndarray]:
    frame = env.render()
    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    arrays = []
    if isinstance(frame, dict):
        for value in frame.values():
            if hasattr(value, "cpu"):
                value = value.cpu().numpy()
            value = np.asarray(value)
            if value.ndim == 3:
                arrays.append(ensure_rgb_hwc(value))
            elif value.ndim == 4:
                arrays.extend([ensure_rgb_hwc(value[i]) for i in range(value.shape[0])])
    else:
        arr = np.asarray(frame)
        if arr.ndim == 3:
            arrays = [ensure_rgb_hwc(arr)]
        elif arr.ndim == 4:
            if arr.shape[0] == 1:
                arrays = [ensure_rgb_hwc(arr[0])]
            else:
                arrays = [ensure_rgb_hwc(arr[i]) for i in range(arr.shape[0])]
        else:
            arrays = [extract_rgb_from_render(env)]
    if not arrays:
        arrays = [extract_rgb_from_render(env)]
    return arrays


def extract_rgb_views(obs: dict, env, num_cameras: int) -> list[np.ndarray]:
    global _warned_camera_fallback, _warned_render_fallback
    sensor_views = extract_sensor_rgb_views(obs)
    if len(sensor_views) >= num_cameras:
        return sensor_views[:num_cameras]
    render_views = extract_render_rgb_views(env)
    if len(sensor_views) < num_cameras and render_views and not _warned_render_fallback:
        typer.echo(
            "Warning: fewer sensor camera views than requested; filling remaining views from render().",
            err=True,
        )
        _warned_render_fallback = True
    arrays = sensor_views + render_views
    if len(arrays) < num_cameras:
        if not _warned_camera_fallback:
            typer.echo(
                "Warning: fewer total camera views than requested after sensor+render; duplicating last camera view.",
                err=True,
            )
            _warned_camera_fallback = True
        arrays.extend([arrays[-1].copy() for _ in range(num_cameras - len(arrays))])
    return arrays[:num_cameras]


def flatten_obs_state(obs: dict) -> np.ndarray:
    if "state" in obs:
        x = obs["state"]
    elif "state_dict" in obs:
        x = common.flatten_state_dict(obs["state_dict"], use_torch=False)
    else:
        agent = obs.get("agent", {})
        if "qpos" in agent and "qvel" in agent:
            x = np.concatenate(
                [
                    np.asarray(agent["qpos"]).flatten(),
                    np.asarray(agent["qvel"]).flatten(),
                ],
                axis=-1,
            )
        else:
            raise ValueError(f"Cannot get state from obs. Keys: {list(obs.keys())}")
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    return np.asarray(x, dtype=np.float32).flatten()


def load_json(path: Path) -> dict:
    return io_utils.load_json(str(path))


def reset_env_for_episode(env, ep_meta: dict) -> tuple[dict, int]:
    reset_kwargs = ep_meta.get("reset_kwargs", {}) or {}
    seed = ep_meta.get("episode_seed", reset_kwargs.get("seed", 0))
    if isinstance(seed, list):
        seed = seed[0]
    options = {k: v for k, v in reset_kwargs.items() if k != "seed"}
    try:
        obs, _ = env.reset(seed=seed, options=options if options else None)
    except TypeError:
        obs, _ = env.reset(seed=seed)
    return obs, int(seed)


def collect_episode_with_actions(
    env,
    actions: np.ndarray,
    image_size: int,
    num_cameras: int,
) -> tuple[np.ndarray, np.ndarray]:
    images = []
    states = []
    obs = env.unwrapped.get_obs(unflattened=True)
    if hasattr(obs, "keys"):
        obs = common.index_dict_array(obs, 0, inplace=False)
    T = len(actions)
    for t in range(T):
        rgbs = extract_rgb_views(obs, env, num_cameras=num_cameras)
        state = flatten_obs_state(obs)
        cams = []
        for rgb in rgbs:
            img_pil = Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR)
            cams.append(np.array(img_pil))
        images.append(np.stack(cams, axis=0))
        states.append(np.asarray(state, dtype=np.float32).flatten())
        action_env = np.asarray(actions[t], dtype=np.float32).reshape(1, -1)
        obs, _, terminated, truncated, _ = env.step(action_env)
        if bool(terminated) or bool(truncated):
            break
        if hasattr(obs, "keys"):
            obs = common.index_dict_array(obs, 0, inplace=False)
    return np.stack(images, axis=0), np.stack(states, axis=0)


def main(
    skill: str = typer.Option(DEFAULT_SKILL, "--skill", "-s", help="ManiSkill env ID"),
    raw_dir: Path = typer.Option(RAW_DIR, "--raw-dir", "-r", path_type=Path),
    output_dir: Path = typer.Option(PREPROCESSED_DIR, "--output-dir", "-o", path_type=Path),
    max_episodes: int = typer.Option(None, "--max-episodes", "-n", help="Cap number of episodes (default: all)"),
    image_size: int = typer.Option(256, "--image-size", help="Resize RGB to this width/height"),
    num_cameras: int = typer.Option(2, "--num-cameras", help="Number of camera views to store per timestep"),
    obs_mode: str = typer.Option("rgb+state", "--obs-mode", help="Observation mode for replay"),
) -> None:
    """Replay raw trajectories to collect RGB + state + actions and save a single .pt file."""
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path, json_path = find_trajectory_files(raw_dir, skill)
    json_data = load_json(json_path)
    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    env_kwargs = {
        "num_envs": 1,
        "obs_mode": obs_mode,
        "control_mode": env_info.get("env_kwargs", {}).get("control_mode", "pd_joint_pos"),
        "render_mode": "rgb_array",
        "sim_backend": "physx_cpu",
        "render_backend": "cpu",
    }

    episodes_meta = json_data["episodes"]
    if max_episodes is not None:
        episodes_meta = episodes_meta[:max_episodes]

    typer.echo(f"Replaying {len(episodes_meta)} episodes from {h5_path}")

    env = gym.make(env_id, **env_kwargs)
    episodes_out = []
    base = env.unwrapped

    with h5py.File(h5_path, "r") as f:
        for ep_meta in tqdm(episodes_meta, desc="Episodes"):
            ep_id = ep_meta["episode_id"]
            traj_id = f"traj_{ep_id}"
            if traj_id not in f:
                continue
            try:
                traj = f[traj_id]
                env_states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
                actions = np.asarray(traj["actions"][:])
                reset_env_for_episode(env, ep_meta)

                try:
                    base.set_state_dict(common.batch(env_states[0]))
                    T = len(actions)
                    images = []
                    states = []
                    for t in range(T):
                        base.set_state_dict(common.batch(env_states[t]))
                        obs = base.get_obs(unflattened=True)
                        if hasattr(obs, "keys"):
                            obs = common.index_dict_array(obs, 0, inplace=False)
                        rgbs = extract_rgb_views(obs, env, num_cameras=num_cameras)
                        state = flatten_obs_state(obs)
                        cams = []
                        for rgb in rgbs:
                            img_pil = Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR)
                            cams.append(np.array(img_pil))
                        images.append(np.stack(cams, axis=0))
                        states.append(np.asarray(state, dtype=np.float32).flatten())
                    images_np = np.stack(images, axis=0)
                    states_np = np.stack(states, axis=0)
                    actions_np = np.asarray(actions[: images_np.shape[0]], dtype=np.float32)
                except KeyError as e:
                    tqdm.write(
                        f"Episode {ep_id}: env-state replay failed ({e}); falling back to action replay for this episode."
                    )
                    obs0, _ = reset_env_for_episode(env, ep_meta)
                    if hasattr(obs0, "keys"):
                        _ = common.index_dict_array(obs0, 0, inplace=False)
                    images_np, states_np = collect_episode_with_actions(
                        env=env,
                        actions=actions,
                        image_size=image_size,
                        num_cameras=num_cameras,
                    )
                    actions_np = np.asarray(actions[: images_np.shape[0]], dtype=np.float32)
                episodes_out.append({
                    "images": torch.from_numpy(images_np).permute(0, 1, 4, 2, 3),
                    "states": torch.from_numpy(states_np),
                    "actions": torch.from_numpy(actions_np),
                    "instruction": PICK_CUBE_INSTRUCTION,
                })
            except Exception as e:
                tqdm.write(f"Episode {ep_id} failed: {e}")
                raise

    env.close()

    if not episodes_out:
        typer.echo("No episodes were processed. Check that trajectory keys exist (traj_0, traj_1, ...).", err=True)
        raise typer.Exit(1)

    action_dim = int(episodes_out[0]["actions"].shape[-1])
    state_dim = int(episodes_out[0]["states"].shape[-1])
    out_name = skill.replace("-v1", "").replace("-", "_").lower() + ".pt"
    out_path = output_dir / out_name
    torch.save({
        "episodes": episodes_out,
        "metadata": {
            "env_id": env_id,
            "skill": skill,
            "num_episodes": len(episodes_out),
            "action_dim": action_dim,
            "state_dim": state_dim,
            "image_size": image_size,
            "num_cameras": num_cameras,
            "instruction": PICK_CUBE_INSTRUCTION,
        },
    }, out_path)
    typer.echo(f"Saved {len(episodes_out)} episodes to {out_path}")


if __name__ == "__main__":
    typer.run(main)
