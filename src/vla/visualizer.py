"""
Visualization tools for VLA policies and demonstrations.

Provides commands to:
- Replay recorded trajectories from demonstrations
- Visualize trained policies in real-time
- Test environment rendering

Usage:
    uv run python src/vla/visualizer.py replay --env PickCube-v1
    uv run python src/vla/visualizer.py policy --model models/rt1_pickcube_v1.pt --render
    uv run python src/vla/visualizer.py test-env
"""
import time
from pathlib import Path
from typing import Optional

import cv2
import gymnasium as gym
import h5py
import numpy as np
import torch
import typer
from mani_skill.trajectory import utils as trajectory_utils
from mani_skill.utils import common
from PIL import Image

import mani_skill.envs

app = typer.Typer(no_args_is_help=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DEMO_PATH = Path.home() / ".maniskill" / "demos"


def find_trajectory_file(env_id: str) -> Path:
    """Find trajectory file for environment, checking multiple locations."""
    candidates = [
        RAW_DIR / env_id / "motionplanning" / "trajectory.h5",
        RAW_DIR / env_id / "trajectory.h5",
        DEMO_PATH / env_id / "motionplanning" / "trajectory.h5",
        DEMO_PATH / env_id / "trajectory.h5",
    ]
    for path in candidates:
        if path.exists():
            return path

    for base in [RAW_DIR, DEMO_PATH]:
        env_dir = base / env_id
        if env_dir.exists():
            for subdir in env_dir.iterdir():
                if subdir.is_dir() and (subdir / "trajectory.h5").exists():
                    return subdir / "trajectory.h5"

    raise FileNotFoundError(f"No trajectory.h5 found for {env_id}")


def create_env(env_id: str, render_mode: str = "rgb_array"):
    """Create ManiSkill environment with consistent settings."""
    return gym.make(
        env_id,
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode=render_mode,
        sim_backend="physx_cpu",
        render_backend="cpu",
    )


def get_frame(env) -> np.ndarray:
    """Get rendered frame from environment."""
    frame = env.render()
    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    return frame


def display_frame(frame: np.ndarray, window_name: str = "Visualization", wait_ms: int = 1) -> bool:
    """Display frame in OpenCV window. Returns False if 'q' pressed."""
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow(window_name, frame_bgr)
    key = cv2.waitKey(wait_ms)
    return not (key & 0xFF == ord('q'))


def bins_to_continuous(bins: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    return (bins.float() / (num_bins - 1)) * 2 - 1


def create_rt1_model(action_dim: int, device: str, model_size: str):
    from robotic_transformer_pytorch import RT1, MaxViT

    configs = {
        "tiny": {
            "dim_conv_stem": 16, "dim": 32, "dim_head": 16, "depth": (1, 1, 1, 1),
            "rt1_depth": 2, "rt1_heads": 2, "rt1_dim_head": 16,
        },
        "small": {
            "dim_conv_stem": 32, "dim": 48, "dim_head": 16, "depth": (1, 1, 2, 1),
            "rt1_depth": 4, "rt1_heads": 4, "rt1_dim_head": 32,
        },
        "base": {
            "dim_conv_stem": 64, "dim": 96, "dim_head": 32, "depth": (2, 2, 5, 2),
            "rt1_depth": 6, "rt1_heads": 8, "rt1_dim_head": 64,
        },
    }
    cfg = configs.get(model_size, configs["small"])

    vit = MaxViT(
        num_classes=1000,
        dim_conv_stem=cfg["dim_conv_stem"],
        dim=cfg["dim"],
        dim_head=cfg["dim_head"],
        depth=cfg["depth"],
        window_size=8,
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
    )

    return RT1(
        vit=vit,
        num_actions=action_dim,
        action_bins=256,
        depth=cfg["rt1_depth"],
        heads=cfg["rt1_heads"],
        dim_head=cfg["rt1_dim_head"],
        cond_drop_prob=0.2,
    ).to(device)


@app.command()
def replay(
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="Environment ID"),
    traj_path: Optional[str] = typer.Option(None, "--traj", "-t", help="Path to trajectory file"),
    episode: int = typer.Option(0, "--episode", "-ep", help="Episode to replay"),
    speed: float = typer.Option(1.0, "--speed", "-s", help="Playback speed"),
    loop: bool = typer.Option(False, "--loop", "-l", help="Loop the replay"),
) -> None:
    """Replay recorded trajectory from demonstrations."""
    traj_file = Path(traj_path) if traj_path else find_trajectory_file(env_id)
    print(f"Loading trajectory from: {traj_file}")

    with h5py.File(traj_file, "r") as f:
        traj_keys = sorted(k for k in f.keys() if k.startswith("traj"))
        if not traj_keys:
            typer.echo("No trajectories found in file", err=True)
            raise typer.Exit(1)

        traj_key = traj_keys[min(episode, len(traj_keys) - 1)]
        traj = f[traj_key]
        actions = traj["actions"][:]
        env_states = trajectory_utils.dict_to_list_of_dicts(traj["env_states"])
        print(f"Loaded {traj_key}: {len(actions)} steps")

    env = create_env(env_id)
    cv2.namedWindow("Trajectory Replay", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Trajectory Replay", 512, 512)

    print(f"\nReplaying trajectory (speed={speed}x)")
    print("Press 'q' to quit")

    try:
        while True:
            env.reset()
            env.unwrapped.set_state_dict(common.batch(env_states[0]))

            for step in range(len(env_states) - 1):
                env.unwrapped.set_state_dict(common.batch(env_states[step + 1]))
                frame = get_frame(env)

                if not display_frame(frame, "Trajectory Replay", int(20 / speed)):
                    raise KeyboardInterrupt

            if not loop:
                time.sleep(1.0)
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    env.close()
    cv2.destroyAllWindows()


@app.command()
def policy(
    model_path: str = typer.Option(..., "--model", "-m", help="Path to trained model"),
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="Environment ID"),
    max_steps: int = typer.Option(100, "--max-steps", "-s", help="Max steps per episode"),
    speed: float = typer.Option(1.0, "--speed", help="Playback speed"),
    loop: bool = typer.Option(False, "--loop", "-l", help="Keep running episodes"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device"),
) -> None:
    """Visualize trained policy running in environment."""
    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        typer.echo(f"Model not found: {model_path}", err=True)
        raise typer.Exit(1)

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    action_dim = config["action_dim"]
    model_size = config["model_size"]
    instruction = config["instruction"]

    model = create_rt1_model(action_dim, device, model_size)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded: {model_size}, action_dim={action_dim}")

    print(f"Creating environment: {env_id}")
    env = create_env(env_id)

    cv2.namedWindow("Policy Visualization", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Policy Visualization", 512, 512)

    print(f"\nVisualizing policy (speed={speed}x)")
    print(f"Instruction: '{instruction}'")
    print("Press 'q' to quit")

    total_rewards = []
    successes = []
    ep = 0

    try:
        while True:
            obs, info = env.reset(seed=ep)
            episode_reward = 0.0
            print(f"\nEpisode {ep + 1}...")

            for step in range(max_steps):
                frame = get_frame(env)
                rgb = np.array(Image.fromarray(frame).resize((256, 256)))

                img_tensor = torch.from_numpy(rgb).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)
                video = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
                video = video.permute(0, 2, 1, 3, 4)

                with torch.no_grad():
                    logits = model(video, texts=[instruction])
                    action_bins = logits.argmax(dim=-1)[0, 0]

                action = bins_to_continuous(action_bins).cpu().numpy()
                action = action.reshape(1, -1)

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward.item() if hasattr(reward, "item") else reward)

                if not display_frame(frame, "Policy Visualization", int(20 / speed)):
                    raise KeyboardInterrupt

                done = terminated.any() if hasattr(terminated, "any") else terminated
                if done:
                    break

            success = info.get("success", False)
            if hasattr(success, "item"):
                success = success.item()

            total_rewards.append(episode_reward)
            successes.append(success)
            ep += 1
            print(f"  Reward: {episode_reward:.2f}, Success: {success}")

            if not loop:
                break

    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    env.close()
    cv2.destroyAllWindows()

    if total_rewards:
        print("\n=== Summary ===")
        print(f"Episodes: {len(total_rewards)}")
        print(f"Mean Reward: {np.mean(total_rewards):.2f} (+/- {np.std(total_rewards):.2f})")
        print(f"Success Rate: {np.mean(successes) * 100:.1f}%")


@app.command()
def test_env(
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="Environment ID"),
    steps: int = typer.Option(100, "--steps", "-s", help="Number of steps"),
    speed: float = typer.Option(1.0, "--speed", help="Playback speed"),
) -> None:
    """Test environment rendering with random actions."""
    print(f"Creating environment: {env_id}")
    env = create_env(env_id)

    cv2.namedWindow("Environment Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Environment Test", 512, 512)

    print(f"\nTesting environment with random actions")
    print("Press 'q' to quit")

    try:
        env.reset(seed=0)
        for step in range(steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            frame = get_frame(env)

            if not display_frame(frame, "Environment Test", int(20 / speed)):
                break

            done = terminated.any() if hasattr(terminated, "any") else terminated
            if done:
                env.reset()

    except KeyboardInterrupt:
        print("\nStopped by user.")

    env.close()
    cv2.destroyAllWindows()
    print("Done!")


@app.command()
def list_demos(
    env_id: Optional[str] = typer.Option(None, "--env", "-e", help="Environment ID"),
) -> None:
    """List available demonstration files."""
    search_dirs = [RAW_DIR, DEMO_PATH]

    if env_id:
        try:
            traj_file = find_trajectory_file(env_id)
            with h5py.File(traj_file, "r") as f:
                traj_keys = sorted(k for k in f.keys() if k.startswith("traj"))
                print(f"{env_id}: {len(traj_keys)} trajectories")
                print(f"  Path: {traj_file}")
        except FileNotFoundError:
            print(f"No demos found for {env_id}")
    else:
        print("Available demonstrations:")
        found_any = False
        for base_dir in search_dirs:
            if not base_dir.exists():
                continue
            for env_dir in sorted(base_dir.iterdir()):
                if env_dir.is_dir():
                    try:
                        traj_file = find_trajectory_file(env_dir.name)
                        with h5py.File(traj_file, "r") as f:
                            count = sum(1 for k in f.keys() if k.startswith("traj"))
                            print(f"  {env_dir.name}: {count} trajectories")
                            found_any = True
                    except FileNotFoundError:
                        pass
        if not found_any:
            print("  No demonstrations found.")
            print(f"  Searched: {search_dirs}")


if __name__ == "__main__":
    app()
