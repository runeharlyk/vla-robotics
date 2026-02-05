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

import mani_skill.envs
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import common, io_utils

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
DEFAULT_SKILL = "PickCube-v1"
PICK_CUBE_INSTRUCTION = "pick up the red cube and move it to the green goal"


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


def main(
    skill: str = typer.Option(DEFAULT_SKILL, "--skill", "-s", help="ManiSkill env ID"),
    raw_dir: Path = typer.Option(RAW_DIR, "--raw-dir", "-r", path_type=Path),
    output_dir: Path = typer.Option(PREPROCESSED_DIR, "--output-dir", "-o", path_type=Path),
    max_episodes: int = typer.Option(None, "--max-episodes", "-n", help="Cap number of episodes (default: all)"),
    image_size: int = typer.Option(256, "--image-size", help="Resize RGB to this width/height"),
    obs_mode: str = typer.Option("state", "--obs-mode", help="Observation mode for replay"),
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
                seed = ep_meta.get("episode_seed", ep_meta.get("reset_kwargs", {}).get("seed", 0))
                if isinstance(seed, list):
                    seed = seed[0]
                env.reset(seed=seed)
                base.set_state_dict(common.batch(env_states[0]))
                T = len(actions)
                images = []
                states = []
                for t in range(T):
                    base.set_state_dict(common.batch(env_states[t]))
                    obs = base.get_obs(unflattened=True)
                    if hasattr(obs, "keys"):
                        obs = common.index_dict_array(obs, 0, inplace=False)
                    rgb = extract_rgb_from_render(env)
                    state = flatten_obs_state(obs)
                    img_pil = Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR)
                    images.append(np.array(img_pil))
                    states.append(np.asarray(state, dtype=np.float32).flatten())
                images_np = np.stack(images, axis=0)
                states_np = np.stack(states, axis=0)
                actions_np = np.asarray(actions, dtype=np.float32)
                episodes_out.append({
                    "images": torch.from_numpy(images_np).permute(0, 3, 1, 2),
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
            "instruction": PICK_CUBE_INSTRUCTION,
        },
    }, out_path)
    typer.echo(f"Saved {len(episodes_out)} episodes to {out_path}")


if __name__ == "__main__":
    typer.run(main)
