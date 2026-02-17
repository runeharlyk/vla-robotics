"""
Preprocess raw ManiSkill demos into VLA-ready HDF5 files in data/preprocessed.

Replays each trajectory in simulation to collect RGB, state, and actions,
then saves a single .h5 file with JPEG-compressed images for efficient training.

Usage:
    uv run python scripts/preprocess_data.py
    uv run python scripts/preprocess_data.py --skill PickCube-v1 --max-episodes 100
"""

import io
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import typer
from PIL import Image
from tqdm import tqdm

from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import common, io_utils

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

INSTRUCTION_MAP = {
    "PickCube-v1": "pick up the red cube and move it to the green goal",
    "StackCube-v1": "stack the red cube on top of the green cube",
    "PegInsertionSide-v1": "insert the peg into the hole from the side",
    "PushCube-v1": "push the cube to the target location",
    "PushT-v1": "push the T-shaped block to the target pose",
    "PullCube-v1": "pull the cube to the target location",
    "PullCubeTool-v1": "use the tool to pull the cube to the goal",
    "PokeCube-v1": "poke the cube toward the goal",
    "RollBall-v1": "roll the ball to the target area",
    "StackPyramid-v1": "stack the cubes into a pyramid",
    "PlugCharger-v1": "plug the charger into the socket",
    "TwoRobotPickCube-v1": "use both arms to pick up and place the cube",
    "TwoRobotStackCube-v1": "use both arms to stack one cube on another",
}

JPEG_QUALITY = 95


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
    """
    Handles:
    - single-arm Panda
    - two-arm Panda (TwoRobot*)
    - state / state_dict modes
    """

    # Case 1: already flattened
    if "state" in obs:
        x = obs["state"]

    # Case 2: ManiSkill state_dict
    elif "state_dict" in obs:
        x = common.flatten_state_dict(obs["state_dict"], use_torch=False)

    # Case 3: agent-based structure (single or multi robot)
    elif "agent" in obs:
        agent = obs["agent"]

        # MULTI-ROBOT CASE
        if isinstance(agent, dict) and all(isinstance(v, dict) for v in agent.values()):
            parts = []

            # Sort keys for deterministic ordering
            for robot_name in sorted(agent.keys()):
                robot = agent[robot_name]

                if "qpos" not in robot or "qvel" not in robot:
                    raise ValueError(f"Missing qpos/qvel in {robot_name}")

                qpos = robot["qpos"]
                qvel = robot["qvel"]

                if hasattr(qpos, "cpu"):
                    qpos = qpos.cpu().numpy()
                if hasattr(qvel, "cpu"):
                    qvel = qvel.cpu().numpy()

                parts.append(np.concatenate([qpos.flatten(), qvel.flatten()]))

            x = np.concatenate(parts)

        # SINGLE ROBOT CASE
        elif "qpos" in agent and "qvel" in agent:
            qpos = agent["qpos"]
            qvel = agent["qvel"]

            if hasattr(qpos, "cpu"):
                qpos = qpos.cpu().numpy()
            if hasattr(qvel, "cpu"):
                qvel = qvel.cpu().numpy()

            x = np.concatenate([qpos.flatten(), qvel.flatten()])

        else:
            raise ValueError(f"Unsupported agent structure: {agent.keys()}")

    else:
        raise ValueError(f"Cannot get state from obs. Keys: {list(obs.keys())}")

    if hasattr(x, "cpu"):
        x = x.cpu().numpy()

    return np.asarray(x, dtype=np.float32).flatten()


def load_json(path: Path) -> dict:
    return io_utils.load_json(str(path))


def encode_jpeg(rgb: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Encode an RGB numpy array as JPEG bytes."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def main(
    skill: str = typer.Option(..., "--skill", "-s", help="ManiSkill env ID"),
    raw_dir: Path = typer.Option(RAW_DIR, "--raw-dir", "-r", path_type=Path),
    output_dir: Path = typer.Option(PREPROCESSED_DIR, "--output-dir", "-o", path_type=Path),
    max_episodes: int = typer.Option(None, "--max-episodes", "-n", help="Cap number of episodes (default: all)"),
    image_size: int = typer.Option(256, "--image-size", help="Resize RGB to this width/height"),
    obs_mode: str = typer.Option("state", "--obs-mode", help="Observation mode for replay"),
    jpeg_quality: int = typer.Option(JPEG_QUALITY, "--jpeg-quality", help="JPEG compression quality (1-100)"),
) -> None:
    """Replay raw trajectories to collect RGB + state + actions and save a single HDF5 file."""
    raw_dir = raw_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    instruction = INSTRUCTION_MAP.get(skill, skill)

    h5_path, json_path = find_trajectory_files(raw_dir, skill)
    json_data = load_json(json_path)
    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    control_mode = env_info.get("env_kwargs", {}).get("control_mode", "pd_joint_pos")
    env_kwargs = {
        "num_envs": 1,
        "obs_mode": obs_mode,
        "control_mode": control_mode,
        "render_mode": "rgb_array",
        "sim_backend": "physx_cpu",
        "render_backend": "cpu",
    }

    episodes_meta = json_data["episodes"]
    if max_episodes is not None:
        episodes_meta = episodes_meta[:max_episodes]

    typer.echo(f"Replaying {len(episodes_meta)} episodes from {h5_path}")

    env = gym.make(env_id, **env_kwargs)
    base = env.unwrapped

    out_name = skill.replace("-v1", "").replace("-", "_").lower() + ".h5"
    out_path = output_dir / out_name

    jpeg_dtype = h5py.special_dtype(vlen=np.uint8)
    num_written = 0
    first_action_dim = None
    first_state_dim = None

    with h5py.File(out_path, "w") as out_h5, h5py.File(h5_path, "r") as raw_h5:
        for ep_meta in tqdm(episodes_meta, desc="Episodes"):
            ep_id = ep_meta["episode_id"]
            traj_id = f"traj_{ep_id}"
            if traj_id not in raw_h5:
                continue
            try:
                traj = raw_h5[traj_id]
                env_states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
                actions = np.asarray(traj["actions"][:], dtype=np.float32)
                seed = ep_meta.get("episode_seed", ep_meta.get("reset_kwargs", {}).get("seed", 0))
                if isinstance(seed, list):
                    seed = seed[0]
                env.reset(seed=seed)
                base.set_state_dict(common.batch(env_states[0]))

                T = len(actions)
                jpeg_list = []
                state_list = []

                for t in range(T):
                    base.set_state_dict(common.batch(env_states[t]))
                    obs = base.get_obs(unflattened=True)
                    if hasattr(obs, "keys"):
                        obs = common.index_dict_array(obs, 0, inplace=False)
                    rgb = extract_rgb_from_render(env)
                    state = flatten_obs_state(obs)
                    img_pil = Image.fromarray(rgb).resize((image_size, image_size), Image.BILINEAR)
                    jpeg_list.append(np.frombuffer(encode_jpeg(np.array(img_pil), jpeg_quality), dtype=np.uint8))
                    state_list.append(state)

                states_np = np.stack(state_list, axis=0)

                grp = out_h5.create_group(f"episode_{num_written}")
                img_ds = grp.create_dataset("images", shape=(T,), dtype=jpeg_dtype)
                for t, jpg in enumerate(jpeg_list):
                    img_ds[t] = jpg
                grp.create_dataset("states", data=states_np, dtype=np.float32)
                grp.create_dataset("actions", data=actions, dtype=np.float32)

                if first_action_dim is None:
                    first_action_dim = int(actions.shape[-1])
                    first_state_dim = int(states_np.shape[-1])

                num_written += 1
            except Exception as e:
                tqdm.write(f"Episode {ep_id} failed: {e}")
                raise

    env.close()

    if num_written == 0:
        typer.echo("No episodes were processed.", err=True)
        raise typer.Exit(1)

    with h5py.File(out_path, "a") as out_h5:
        out_h5.attrs["env_id"] = env_id
        out_h5.attrs["skill"] = skill
        out_h5.attrs["num_episodes"] = num_written
        out_h5.attrs["action_dim"] = first_action_dim
        out_h5.attrs["state_dim"] = first_state_dim
        out_h5.attrs["image_size"] = image_size
        out_h5.attrs["instruction"] = instruction
        out_h5.attrs["jpeg_quality"] = jpeg_quality
        out_h5.attrs["control_mode"] = control_mode

    size_mb = out_path.stat().st_size / (1024 * 1024)
    typer.echo(f"Saved {num_written} episodes to {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    typer.run(main)
