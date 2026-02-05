import time
from pathlib import Path

import gymnasium as gym
import h5py
import mani_skill.envs
import numpy as np
import typer
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import common

app = typer.Typer()
DEMO_PATH = Path.home() / ".maniskill" / "demos"


def find_trajectory_file(env_id: str) -> Path:
    candidates = [
        DEMO_PATH / env_id / "motionplanning" / "trajectory.h5",
        DEMO_PATH / env_id / "trajectory.h5",
    ]
    for path in candidates:
        if path.exists():
            return path

    env_dir = DEMO_PATH / env_id
    if env_dir.exists():
        for subdir in env_dir.iterdir():
            if subdir.is_dir() and (subdir / "trajectory.h5").exists():
                return subdir / "trajectory.h5"

    raise FileNotFoundError(f"No trajectory.h5 found for {env_id} in {DEMO_PATH}")


@app.command()
def replay(
    env_id: str = typer.Option("PickCube-v1", "--env", "-e"),
    traj_path: str = typer.Option(None, "--traj", "-t"),
    episode: int = typer.Option(0, "--episode", "-ep"),
    speed: float = typer.Option(1.0, "--speed", "-s"),
    loop: bool = typer.Option(False, "--loop", "-l"),
    use_env_states: bool = typer.Option(True, "--use-env-states/--use-actions"),
) -> None:
    traj_file = Path(traj_path) if traj_path else find_trajectory_file(env_id)

    with h5py.File(traj_file, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj"))
        traj = f[traj_keys[min(episode, len(traj_keys) - 1)]]
        actions = traj["actions"][:]
        env_states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"]) if use_env_states else None

    env = gym.make(env_id, num_envs=1, obs_mode="state", control_mode="pd_joint_delta_pos", render_mode="human")

    while True:
        env.reset()
        if env_states:
            env.unwrapped.set_state_dict(common.batch(env_states[0]))

        num_steps = len(env_states) - 1 if env_states else len(actions)
        for step in range(num_steps):
            if env_states:
                env.unwrapped.set_state_dict(common.batch(env_states[step + 1]))
            else:
                env.step(np.array(actions[step]).reshape(1, -1))
            env.render()
            if speed < 10:
                time.sleep(0.02 / speed)

        if not loop:
            break
    env.close()


@app.command()
def list_demos(env_id: str = typer.Option(None, "--env", "-e")) -> None:
    if not DEMO_PATH.exists():
        print(f"No demos found. Download with: python -m mani_skill.utils.download_demo <env_id>")
        return

    if env_id:
        traj_file = find_trajectory_file(env_id)
        with h5py.File(traj_file, "r") as f:
            traj_keys = sorted(k for k in f.keys() if k.startswith("traj"))
            print(f"{env_id}: {len(traj_keys)} trajectories ({traj_file})")
    else:
        for env_dir in sorted(DEMO_PATH.iterdir()):
            if env_dir.is_dir():
                try:
                    traj_file = find_trajectory_file(env_dir.name)
                    with h5py.File(traj_file, "r") as f:
                        count = sum(1 for k in f.keys() if k.startswith("traj"))
                        print(f"  {env_dir.name}: {count} trajectories")
                except FileNotFoundError:
                    pass


@app.command()
def list_envs() -> None:
    from mani_skill.utils.registration import REGISTERED_ENVS
    for env_id in sorted(REGISTERED_ENVS.keys()):
        print(env_id)


if __name__ == "__main__":
    app()
