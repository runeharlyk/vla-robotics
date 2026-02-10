"""Generate RGB dataset by replaying ManiSkill demonstrations.

This script replays demonstration trajectories and captures RGB images
using the interactive viewer, creating a dataset for vision-based training.
"""

from pathlib import Path

import h5py
import numpy as np
import typer
from PIL import Image
from tqdm import tqdm

app = typer.Typer()


def capture_frame(viewer, image_width: int, image_height: int) -> np.ndarray:
    """Capture a frame from the viewer window."""
    if viewer is None or viewer.window is None:
        return np.zeros((image_height, image_width, 3), dtype=np.uint8)

    try:
        picture = viewer.window.get_picture("Color")
        if picture is None:
            return np.zeros((image_height, image_width, 3), dtype=np.uint8)

        if hasattr(picture, "cpu"):
            rgba = picture.cpu().numpy()
        else:
            rgba = np.array(picture)

        if rgba.dtype == np.float32:
            rgba = (rgba * 255).astype(np.uint8)

        rgb = rgba[:, :, :3]

        if rgb.shape[0] != image_height or rgb.shape[1] != image_width:
            img = Image.fromarray(rgb)
            img = img.resize((image_width, image_height), Image.BILINEAR)
            rgb = np.array(img)

        return rgb
    except Exception:
        return np.zeros((image_height, image_width, 3), dtype=np.uint8)


@app.command()
def generate(
    env_id: str = typer.Option("PegInsertionSide-v1", "--env", "-e", help="Environment ID"),
    source: str = typer.Option("motionplanning", "--source", "-s", help="Source demo folder"),
    output: str = typer.Option("trajectory_rgb.h5", "--output", "-o", help="Output filename"),
    max_trajectories: int = typer.Option(100, "--max", "-m", help="Max trajectories to process"),
    image_width: int = typer.Option(224, "--width", help="Image width"),
    image_height: int = typer.Option(224, "--height", help="Image height"),
) -> None:
    """Generate RGB dataset by replaying demos with the viewer."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    demo_dir = Path.home() / ".maniskill" / "demos" / env_id / source
    source_file = demo_dir / "trajectory.h5"
    output_file = demo_dir / output

    if not source_file.exists():
        print(f"Source file not found: {source_file}")
        raise typer.Exit(1)

    print(f"Generating RGB dataset for {env_id}")
    print(f"  Source: {source_file}")
    print(f"  Output: {output_file}")
    print(f"  Image size: {image_width}x{image_height}")

    print("Creating environment with viewer...")
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state",
        render_mode="human",
    )
    env.reset()

    for _ in range(10):
        env.render()

    viewer = env.unwrapped.viewer
    if viewer is None or viewer.window is None:
        print("ERROR: Could not initialize viewer")
        env.close()
        raise typer.Exit(1)

    viewer.paused = True
    print("Viewer ready!")

    print("Processing trajectories...")
    with h5py.File(str(source_file), "r") as src, h5py.File(str(output_file), "w") as dst:
        traj_keys = sorted([k for k in src.keys() if k.startswith("traj")])[:max_trajectories]
        print(f"Found {len(traj_keys)} trajectories to process")

        for traj_key in tqdm(traj_keys, desc="Capturing"):
            traj = src[traj_key]
            actions = np.array(traj["actions"][:])
            env_states = traj["env_states"]
            num_steps = len(actions)
            rgb_frames = []

            env.reset()

            for _ in range(3):
                if viewer.window is not None:
                    viewer.window.render_and_get_target()

            for step_idx in range(num_steps):
                if viewer.window is None:
                    print(f"ERROR: Viewer window became None at traj {traj_key} step {step_idx}")
                    rgb_frames.append(np.zeros((image_height, image_width, 3), dtype=np.uint8))
                    continue

                state_dict = {"actors": {}}
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

                try:
                    viewer.window.render_and_get_target()
                except Exception:
                    pass

                rgb = capture_frame(viewer, image_width, image_height)
                rgb_frames.append(rgb)

            if len(rgb_frames) == num_steps:
                out_traj = dst.create_group(traj_key)
                out_traj.create_dataset("rgb", data=np.array(rgb_frames), compression="gzip")
                out_traj.create_dataset("actions", data=actions)

                if "rewards" in traj:
                    out_traj.create_dataset("rewards", data=np.array(traj["rewards"][:]))
                if "success" in traj:
                    out_traj.create_dataset("success", data=np.array(traj["success"][:]))

    env.close()
    print(f"Done! Dataset saved to {output_file}")


@app.command()
def check(
    env_id: str = typer.Option("PegInsertionSide-v1", "--env", "-e", help="Environment ID"),
    source: str = typer.Option("motionplanning", "--source", "-s", help="Source folder"),
) -> None:
    """Check available demo files and their contents."""
    demo_dir = Path.home() / ".maniskill" / "demos" / env_id

    if not demo_dir.exists():
        print(f"No demos found for {env_id}")
        return

    print(f"Demos for {env_id}:")
    for subdir in sorted(demo_dir.iterdir()):
        if subdir.is_dir():
            traj_file = subdir / "trajectory.h5"
            rgb_file = subdir / "trajectory_rgb.h5"

            if traj_file.exists():
                with h5py.File(traj_file, "r") as f:
                    num_trajs = len([k for k in f.keys() if k.startswith("traj")])

                    first_traj = f["traj_0"]
                    has_rgb = "rgb" in first_traj
                    has_env_states = "env_states" in first_traj

                    print(f"  {subdir.name}/trajectory.h5:")
                    print(f"    Trajectories: {num_trajs}")
                    print(f"    Has RGB: {has_rgb}")
                    print(f"    Has env_states: {has_env_states}")

            if rgb_file.exists():
                with h5py.File(rgb_file, "r") as f:
                    num_trajs = len([k for k in f.keys() if k.startswith("traj")])
                    first_traj = f["traj_0"]
                    rgb_shape = first_traj["rgb"].shape if "rgb" in first_traj else None

                    print(f"  {subdir.name}/trajectory_rgb.h5:")
                    print(f"    Trajectories: {num_trajs}")
                    print(f"    RGB shape: {rgb_shape}")


if __name__ == "__main__":
    app()
