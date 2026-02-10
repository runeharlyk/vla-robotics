"""Convert ManiSkill demos to include RGB observations.

ManiSkill demos are often saved with state-only observations for efficiency.
This script replays them with RGB rendering to create a dataset for vision-based learning.
"""

from pathlib import Path
from typing import Optional

import gymnasium as gym
import h5py
import mani_skill.envs  # noqa: F401
import numpy as np
import typer
from tqdm import tqdm

app = typer.Typer()


def replay_trajectory_with_rgb(
    env: gym.Env,
    h5_file: h5py.File,
    traj_key: str,
    output_file: h5py.File,
) -> bool:
    """Replay a single trajectory and save with RGB observations.

    Args:
        env: ManiSkill environment with RGB rendering.
        h5_file: Source HDF5 file.
        traj_key: Trajectory key (e.g., "traj_0").
        output_file: Output HDF5 file.

    Returns:
        True if successful, False otherwise.
    """
    traj = h5_file[traj_key]
    actions = traj["actions"][:]
    env_states = traj["env_states"]

    first_actor = list(env_states["actors"].keys())[0]
    num_steps = len(env_states["actors"][first_actor])

    rgb_observations = []
    env.reset()

    for step_idx in range(num_steps - 1):
        state_dict = {}

        state_dict["actors"] = {}
        for actor_name in env_states["actors"].keys():
            state_dict["actors"][actor_name] = env_states["actors"][actor_name][step_idx]

        if "articulations" in env_states:
            state_dict["articulations"] = {}
            for art_name in env_states["articulations"].keys():
                state_dict["articulations"][art_name] = env_states["articulations"][art_name][step_idx]

        try:
            env.unwrapped.set_state_dict(state_dict)
        except Exception:
            pass

        frame = env.render()
        if isinstance(frame, np.ndarray):
            rgb_observations.append(frame)
        else:
            rgb_observations.append(np.zeros((224, 224, 3), dtype=np.uint8))

    if len(rgb_observations) != len(actions):
        print(f"Warning: {traj_key} has {len(rgb_observations)} frames but {len(actions)} actions")
        return False

    out_traj = output_file.create_group(traj_key)
    out_traj.create_dataset("rgb", data=np.array(rgb_observations), compression="gzip")
    out_traj.create_dataset("actions", data=actions)

    if "rewards" in traj:
        out_traj.create_dataset("rewards", data=traj["rewards"][:])
    if "success" in traj:
        out_traj.create_dataset("success", data=traj["success"][:])

    return True


@app.command()
def convert(
    env_id: str = typer.Option("PegInsertionSide-v1", "--env", "-e", help="Environment ID"),
    source_dir: str = typer.Option("motionplanning", "--source", "-s", help="Source subdirectory"),
    output_name: str = typer.Option("trajectory_rgb.h5", "--output", "-o", help="Output filename"),
    max_trajectories: Optional[int] = typer.Option(None, "--max", "-m", help="Max trajectories to convert"),
    image_size: int = typer.Option(224, "--size", help="Image size"),
) -> None:
    """Convert demos by replaying with RGB rendering."""
    demo_dir = Path.home() / ".maniskill" / "demos" / env_id / source_dir
    source_file = demo_dir / "trajectory.h5"
    output_file = demo_dir / output_name

    if not source_file.exists():
        print(f"Source file not found: {source_file}")
        raise typer.Exit(1)

    print(f"Converting demos for {env_id}")
    print(f"  Source: {source_file}")
    print(f"  Output: {output_file}")

    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="rgbd",
        render_mode="rgb_array",
    )

    with h5py.File(source_file, "r") as src, h5py.File(output_file, "w") as dst:
        traj_keys = [k for k in src.keys() if k.startswith("traj")]

        if max_trajectories:
            traj_keys = traj_keys[:max_trajectories]

        print(f"Converting {len(traj_keys)} trajectories...")

        success_count = 0
        for traj_key in tqdm(traj_keys):
            if replay_trajectory_with_rgb(env, src, traj_key, dst):
                success_count += 1

        print(f"Successfully converted {success_count}/{len(traj_keys)} trajectories")

    env.close()


@app.command()
def info(
    env_id: str = typer.Option("PegInsertionSide-v1", "--env", "-e", help="Environment ID"),
) -> None:
    """Show info about available demos."""
    demo_dir = Path.home() / ".maniskill" / "demos" / env_id

    if not demo_dir.exists():
        print(f"No demos found for {env_id}")
        return

    print(f"Demos for {env_id}:")
    for subdir in demo_dir.iterdir():
        if subdir.is_dir():
            traj_file = subdir / "trajectory.h5"
            rgb_file = subdir / "trajectory_rgb.h5"

            if traj_file.exists():
                with h5py.File(traj_file, "r") as f:
                    num_trajs = len([k for k in f.keys() if k.startswith("traj")])
                    print(f"  {subdir.name}/trajectory.h5: {num_trajs} trajectories")

            if rgb_file.exists():
                with h5py.File(rgb_file, "r") as f:
                    num_trajs = len([k for k in f.keys() if k.startswith("traj")])
                    print(f"  {subdir.name}/trajectory_rgb.h5: {num_trajs} trajectories (RGB)")


if __name__ == "__main__":
    app()
