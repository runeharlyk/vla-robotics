"""
Preprocess raw ManiSkill demos into VLA-ready .pt files in data/preprocessed.

Replays each trajectory in simulation to collect RGB, state, and actions,
then saves a single .pt file with everything needed to train a VLA.

Usage:
    uv run python scripts/preprocess_data.py
    uv run python scripts/preprocess_data.py --skill PickCube-v1 --max-episodes 100
"""

import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import torch
import typer
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import common, io_utils
from PIL import Image
from tqdm import tqdm

from vla.constants import PREPROCESSED_DIR, RAW_DIR
DEFAULT_SKILL = "PickCube-v1"
DEFAULT_INSTRUCTION = "complete the manipulation task"
_warned_camera_fallback = False
_warned_render_fallback = False


def get_task_instruction(env_id: str) -> str:
    """Extract a task instruction from the ManiSkill environment docstring.

    Reads the class docstring via the gymnasium registry without instantiating
    the environment, avoiding SAPIEN/PhysX resource conflicts with worker processes.

    Falls back to a generic instruction if the docstring does not contain
    a ``**Task Description:**`` section.
    """
    import re

    try:
        spec = gym.spec(env_id)
        entry_point = spec.entry_point
        if isinstance(entry_point, str):
            module_path, class_name = entry_point.rsplit(":", 1)
            import importlib

            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        else:
            cls = entry_point
        doc = cls.__doc__ or ""
    except Exception:
        return DEFAULT_INSTRUCTION

    match = re.search(r"\*\*Task Description:\*\*\s*\n\s*(.+?)(?:\n\s*\n|\n\s*\*\*)", doc, re.DOTALL)
    if match:
        desc = match.group(1).strip()
        desc = re.sub(r"\s+", " ", desc)
        desc = re.sub(r"\*[^*]*\*", "", desc).strip()
        first_sentence = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0]
        if first_sentence:
            return first_sentence[0].lower() + first_sentence[1:]
    return DEFAULT_INSTRUCTION


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


def build_state_key_map(env_state_dict: dict, saved_state_dict: dict) -> dict[str, str]:
    """Build a mapping from saved state dict actor/articulation names to env names.

    Handles the common case where saved demos use suffixed names (e.g. 'peg_0')
    but the current ManiSkill version uses bare names (e.g. 'peg').
    """
    import re

    key_map: dict[str, str] = {}
    for category in ("actors", "articulations"):
        env_keys = set(env_state_dict.get(category, {}).keys())
        saved_keys = set(saved_state_dict.get(category, {}).keys())
        for sk in saved_keys:
            if sk in env_keys:
                continue
            bare = re.sub(r"_\d+$", "", sk)
            if bare in env_keys:
                key_map[sk] = bare
    return key_map


def remap_state_dict_keys(state_dict: dict, key_map: dict[str, str]) -> dict:
    """Remap actor/articulation keys in a state dict using a pre-built key map."""
    if not key_map:
        return state_dict
    remapped = {}
    for category, sub in state_dict.items():
        if category in ("actors", "articulations") and isinstance(sub, dict):
            remapped[category] = {key_map.get(k, k): v for k, v in sub.items()}
        else:
            remapped[category] = sub
    return remapped


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
    initial_obs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    images = []
    states = []
    obs = initial_obs if initial_obs is not None else env.unwrapped.get_obs(unflattened=True)
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


_worker_env_cache: dict[str, Any] = {}


def _get_worker_env(env_id: str, obs_mode: str, control_mode: str) -> tuple[Any, Any]:
    """Return a cached (env, base) pair for the current worker process."""
    key = f"{env_id}:{obs_mode}:{control_mode}"
    if key not in _worker_env_cache:
        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode="rgb_array",
            sim_backend="physx_cpu",
            render_backend="cpu",
        )
        env.reset()
        _worker_env_cache[key] = (env, env.unwrapped)
    return _worker_env_cache[key]


def _replay_episode_chunk(
    episode_infos: list[tuple[dict, str]],
    env_id: str,
    control_mode: str,
    obs_mode: str,
    h5_path: str,
    state_key_map: dict[str, str],
    image_size: int,
    num_cameras: int,
    threads_per_worker: int = 0,
) -> list[dict]:
    """Worker: replay a chunk of episodes in a separate process.

    Each worker process lazily creates a single ManiSkill env instance
    (cached across calls) and opens the HDF5 file per invocation.

    Args:
        episode_infos: List of ``(ep_meta, traj_id)`` for episodes to process.
        env_id: ManiSkill environment ID.
        control_mode: Robot control mode string.
        obs_mode: Observation mode string.
        h5_path: Path to the HDF5 trajectory file.
        state_key_map: Actor/articulation key remapping.
        image_size: Target image size (square).
        num_cameras: Number of camera views per timestep.
        threads_per_worker: Max threads for BLAS/OMP libraries (0 = no limit).

    Returns:
        List of episode dicts with numpy arrays (images NCTHWC, states, actions).
    """
    if threads_per_worker > 0:
        t = str(threads_per_worker)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = t

    env, base = _get_worker_env(env_id, obs_mode, control_mode)
    results: list[dict] = []

    with h5py.File(h5_path, "r") as f:
        for ep_meta, traj_id in episode_infos:
            traj = f[traj_id]
            env_states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
            actions = np.asarray(traj["actions"][:])

            try:
                T = len(actions)
                images: list[np.ndarray] = []
                states: list[np.ndarray] = []
                for t in range(T):
                    st = remap_state_dict_keys(env_states[t], state_key_map)
                    base.set_state_dict(common.batch(st))
                    obs = base.get_obs(unflattened=True)
                    if hasattr(obs, "keys"):
                        obs = common.index_dict_array(obs, 0, inplace=False)
                    rgbs = extract_rgb_views(obs, env, num_cameras=num_cameras)
                    state = flatten_obs_state(obs)
                    cams = [
                        np.array(
                            Image.fromarray(rgb).resize(
                                (image_size, image_size),
                                Image.BILINEAR,
                            )
                        )
                        for rgb in rgbs
                    ]
                    images.append(np.stack(cams, axis=0))
                    states.append(np.asarray(state, dtype=np.float32).flatten())
                images_np = np.stack(images)
                states_np = np.stack(states)
                actions_np = np.asarray(actions[: images_np.shape[0]], dtype=np.float32)
            except KeyError:
                obs0, _ = reset_env_for_episode(env, ep_meta)
                images_np, states_np = collect_episode_with_actions(
                    env=env,
                    actions=actions,
                    image_size=image_size,
                    num_cameras=num_cameras,
                    initial_obs=obs0,
                )
                actions_np = np.asarray(actions[: images_np.shape[0]], dtype=np.float32)

            results.append(
                {
                    "images": images_np,
                    "states": states_np,
                    "actions": actions_np,
                }
            )

    return results


def main(
    skill: str = typer.Option(DEFAULT_SKILL, "--skill", "-s", help="ManiSkill env ID"),
    raw_dir: Path = typer.Option(RAW_DIR, "--raw-dir", "-r", path_type=Path),
    output_dir: Path = typer.Option(PREPROCESSED_DIR, "--output-dir", "-o", path_type=Path),
    max_episodes: int = typer.Option(None, "--max-episodes", "-n", help="Cap number of episodes (default: all)"),
    image_size: int = typer.Option(256, "--image-size", help="Resize RGB to this width/height"),
    num_cameras: int = typer.Option(2, "--num-cameras", help="Number of camera views to store per timestep"),
    obs_mode: str = typer.Option("rgb+state", "--obs-mode", help="Observation mode for replay"),
    num_workers: int = typer.Option(
        min(os.cpu_count() or 1, 8),
        "--num-workers",
        "-w",
        help="Parallel worker processes (default: min(cpu_count, 8))",
    ),
) -> None:
    """Replay raw trajectories to collect RGB + state + actions and save a single .pt file."""
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    h5_path, json_path = find_trajectory_files(raw_dir, skill)
    json_data = load_json(json_path)
    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    control_mode = env_info.get("env_kwargs", {}).get("control_mode", "pd_joint_pos")

    episodes_meta = json_data["episodes"]
    if max_episodes is not None:
        episodes_meta = episodes_meta[:max_episodes]

    typer.echo(f"Replaying {len(episodes_meta)} episodes from {h5_path} (num_workers={num_workers})")

    # Build state key map using a disposable single-env probe
    state_key_map: dict[str, str] = {}
    probe_env = gym.make(
        env_id,
        num_envs=1,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode="rgb_array",
        sim_backend="physx_cpu",
        render_backend="cpu",
    )
    probe_env.reset()
    with h5py.File(h5_path, "r") as f:
        for ep_meta in episodes_meta:
            traj_id = f"traj_{ep_meta['episode_id']}"
            if traj_id in f:
                sample = trajectory_utils.dict_to_list_of_dicts(f[traj_id]["env_states"])
                state_key_map = build_state_key_map(
                    probe_env.unwrapped.get_state_dict(),
                    sample[0],
                )
                break
    probe_env.close()
    if state_key_map:
        typer.echo(f"Remapping saved state keys: {state_key_map}")

    # Filter to episodes that exist in the HDF5 file
    h5_path_str = str(h5_path)
    with h5py.File(h5_path_str, "r") as f:
        valid: list[tuple[dict, str]] = []
        for ep_meta in episodes_meta:
            traj_id = f"traj_{ep_meta['episode_id']}"
            if traj_id in f:
                valid.append((ep_meta, traj_id))

    if not valid:
        typer.echo("No valid episodes found in HDF5 file.", err=True)
        raise typer.Exit(1)

    task_instruction = get_task_instruction(env_id)
    typer.echo(f"Task instruction: {task_instruction!r}")

    # Split episodes into small sub-chunks so that the progress bar ticks
    # frequently even when processing many episodes.  Each sub-chunk is
    # submitted as a separate Future; the ProcessPoolExecutor limits the
    # concurrency to ``effective_workers``.
    effective_workers = min(num_workers, len(valid))
    sub_chunk_size = max(1, min(10, math.ceil(len(valid) / effective_workers)))
    sub_chunks = [valid[i : i + sub_chunk_size] for i in range(0, len(valid), sub_chunk_size)]

    typer.echo(f"Dispatching {len(valid)} episodes in {len(sub_chunks)} sub-chunks across {effective_workers} workers")

    threads_per_worker = max(1, (os.cpu_count() or 1) // effective_workers)
    episodes_out: list[dict] = []

    def _convert_results(chunk_results: list[dict]) -> None:
        for ep_data in chunk_results:
            episodes_out.append(
                {
                    "images": torch.from_numpy(ep_data["images"]).permute(0, 1, 4, 2, 3),
                    "states": torch.from_numpy(ep_data["states"]),
                    "actions": torch.from_numpy(ep_data["actions"]),
                    "instruction": task_instruction,
                }
            )

    if effective_workers == 1:
        # Single-worker: run directly in the main process (no IPC overhead)
        pbar = tqdm(total=len(valid), desc="Episodes")
        for sub_chunk in sub_chunks:
            chunk_results = _replay_episode_chunk(
                sub_chunk,
                env_id,
                control_mode,
                obs_mode,
                h5_path_str,
                state_key_map,
                image_size,
                num_cameras,
                threads_per_worker,
            )
            _convert_results(chunk_results)
            pbar.update(len(sub_chunk))
        pbar.close()
    else:
        # Multi-worker: each process creates its own env + HDF5 handle
        mp_context = multiprocessing.get_context("spawn")
        futures_map: dict[object, list] = {}
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=mp_context,
        ) as executor:
            for sub_chunk in sub_chunks:
                future = executor.submit(
                    _replay_episode_chunk,
                    sub_chunk,
                    env_id,
                    control_mode,
                    obs_mode,
                    h5_path_str,
                    state_key_map,
                    image_size,
                    num_cameras,
                    threads_per_worker,
                )
                futures_map[future] = sub_chunk

            pbar = tqdm(total=len(valid), desc="Episodes")
            for future in as_completed(futures_map):
                chunk_results = future.result()
                _convert_results(chunk_results)
                pbar.update(len(futures_map[future]))
            pbar.close()

    if not episodes_out:
        typer.echo(
            "No episodes were processed. Check that trajectory keys exist (traj_0, traj_1, ...).",
            err=True,
        )
        raise typer.Exit(1)

    action_dim = int(episodes_out[0]["actions"].shape[-1])
    state_dim = int(episodes_out[0]["states"].shape[-1])
    out_name = skill.replace("-v1", "").replace("-", "_").lower() + ".pt"
    out_path = output_dir / out_name
    torch.save(
        {
            "episodes": episodes_out,
            "metadata": {
                "env_id": env_id,
                "skill": skill,
                "num_episodes": len(episodes_out),
                "action_dim": action_dim,
                "state_dim": state_dim,
                "image_size": image_size,
                "num_cameras": num_cameras,
                "instruction": task_instruction,
                "control_mode": control_mode,
            },
        },
        out_path,
    )
    typer.echo(f"Saved {len(episodes_out)} episodes to {out_path}")


if __name__ == "__main__":
    typer.run(main)
