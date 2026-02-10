"""Evaluation script for trained CLIP action model in ManiSkill simulation."""

from pathlib import Path
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import typer

from diffusion_policy.clip_action_model import create_clip_action_model

app = typer.Typer()


def load_model_from_checkpoint(
    checkpoint_path: Path,
    action_dim: int,
    clip_model: str = "ViT-B/32",
) -> torch.nn.Module:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        action_dim: Action dimension for the environment.
        clip_model: CLIP model variant.

    Returns:
        Loaded model.
    """
    model = create_clip_action_model(
        action_dim=action_dim,
        clip_model=clip_model,
        freeze_clip=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {checkpoint_path} (epoch {checkpoint['epoch']})")
    return model


@app.command()
def evaluate(
    env_id: str = typer.Option("PegInsertionSide-v1", "--env", "-e", help="Environment ID"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Path to model checkpoint"),
    task_description: str = typer.Option(
        "insert the peg into the hole", "--task", "-t", help="Task description for CLIP"
    ),
    num_episodes: int = typer.Option(10, "--num-episodes", "-n", help="Number of episodes to run"),
    max_steps: int = typer.Option(200, "--max-steps", "-s", help="Maximum steps per episode"),
    render: bool = typer.Option(True, "--render/--no-render", "-r", help="Render the environment"),
    clip_model: str = typer.Option("ViT-B/32", "--clip-model", "-m", help="CLIP model variant"),
    record: Optional[str] = typer.Option(None, "--record", help="Path to save video"),
) -> None:
    """Evaluate a trained CLIP action model in ManiSkill simulation."""
    print(f"Evaluating model on {env_id}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Task: {task_description}")
    print(f"  Episodes: {num_episodes}, Max steps: {max_steps}")

    render_mode = "human" if render else "rgb_array"
    if record:
        render_mode = "rgb_array"

    kwargs = dict(
        num_envs=1,
        obs_mode="rgbd",
        control_mode="pd_joint_delta_pos",
        render_mode=render_mode,
    )

    env = gym.make(env_id, **kwargs)
    action_dim = env.action_space.shape[-1]
    print(f"Environment created. Action dim: {action_dim}")

    model = load_model_from_checkpoint(
        Path(checkpoint),
        action_dim=action_dim,
        clip_model=clip_model,
    )
    print(f"Model loaded on device: {model.device}")

    if record:
        from mani_skill.utils.wrappers import RecordEpisode

        record_path = Path(record)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        env = RecordEpisode(env, output_dir=str(record_path.parent), save_video=True, video_fps=30)

    episode_rewards = []
    episode_successes = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        success = False

        for step in range(max_steps):
            if "sensor_data" in obs and "base_camera" in obs["sensor_data"]:
                rgb = obs["sensor_data"]["base_camera"]["rgb"]
            elif "sensor_data" in obs:
                camera_keys = list(obs["sensor_data"].keys())
                if camera_keys:
                    rgb = obs["sensor_data"][camera_keys[0]]["rgb"]
                else:
                    rgb = None
            else:
                rgb = None

            if rgb is None:
                action = env.action_space.sample()
            else:
                if hasattr(rgb, "cpu"):
                    rgb_np = rgb.cpu().numpy()
                else:
                    rgb_np = np.array(rgb)

                if rgb_np.ndim == 4:
                    rgb_np = rgb_np[0]
                if rgb_np.shape[-1] == 3 or rgb_np.shape[-1] == 4:
                    rgb_np = rgb_np[..., :3]
                    rgb_np = rgb_np.transpose(2, 0, 1)

                rgb_tensor = torch.from_numpy(rgb_np).float()
                if rgb_tensor.max() > 1.0:
                    rgb_tensor = rgb_tensor / 255.0

                rgb_tensor = torch.nn.functional.interpolate(
                    rgb_tensor.unsqueeze(0),
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                ).to(model.device)

                with torch.no_grad():
                    action = model(rgb_tensor, text=[task_description])
                action = action.cpu().numpy().squeeze()

            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            if render:
                env.render()

            reward_val = reward.item() if hasattr(reward, "item") else float(reward)
            episode_reward += reward_val

            if "success" in info:
                success_val = info["success"]
                if hasattr(success_val, "item"):
                    success_val = success_val.item()
                if success_val:
                    success = True

            term = terminated.any() if hasattr(terminated, "any") else terminated
            trunc = truncated.any() if hasattr(truncated, "any") else truncated
            if term or trunc:
                break

        episode_rewards.append(episode_reward)
        episode_successes.append(success)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Success = {success}, Steps = {step + 1}")

    env.close()

    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Success Rate: {np.mean(episode_successes) * 100:.1f}%")
    print(f"  Episodes: {num_episodes}")
    print("=" * 50)


@app.command()
def visualize_trained(
    env_id: str = typer.Option("PegInsertionSide-v1", "--env", "-e", help="Environment ID"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Path to model checkpoint"),
    task_description: str = typer.Option(
        "insert the peg into the hole", "--task", "-t", help="Task description for CLIP"
    ),
    max_steps: int = typer.Option(1000, "--max-steps", "-s", help="Maximum steps"),
    clip_model: str = typer.Option("ViT-B/32", "--clip-model", "-m", help="CLIP model variant"),
) -> None:
    """Visualize a trained model in the ManiSkill simulation (continuous loop)."""
    print(f"Visualizing trained model on {env_id}")

    kwargs = dict(
        num_envs=1,
        obs_mode="rgbd",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
    )

    env = gym.make(env_id, **kwargs)
    action_dim = env.action_space.shape[-1]
    print(f"Environment created. Action dim: {action_dim}")

    model = load_model_from_checkpoint(
        Path(checkpoint),
        action_dim=action_dim,
        clip_model=clip_model,
    )
    print("Model loaded. Running visualization...")

    obs, _ = env.reset()
    step = 0

    while step < max_steps:
        if "sensor_data" in obs:
            camera_keys = list(obs["sensor_data"].keys())
            if camera_keys:
                rgb = obs["sensor_data"][camera_keys[0]]["rgb"]
            else:
                rgb = None
        else:
            rgb = None

        if rgb is None:
            action = env.action_space.sample()
        else:
            if hasattr(rgb, "cpu"):
                rgb_np = rgb.cpu().numpy()
            else:
                rgb_np = np.array(rgb)

            if rgb_np.ndim == 4:
                rgb_np = rgb_np[0]
            if rgb_np.shape[-1] == 3 or rgb_np.shape[-1] == 4:
                rgb_np = rgb_np[..., :3]
                rgb_np = rgb_np.transpose(2, 0, 1)

            rgb_tensor = torch.from_numpy(rgb_np).float()
            if rgb_tensor.max() > 1.0:
                rgb_tensor = rgb_tensor / 255.0

            rgb_tensor = torch.nn.functional.interpolate(
                rgb_tensor.unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).to(model.device)

            with torch.no_grad():
                action = model(rgb_tensor, text=[task_description])
            action = action.cpu().numpy().squeeze()

        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        step += 1

        if step % 100 == 0:
            print(f"Step {step}, Reward: {reward}")

        term = terminated.any() if hasattr(terminated, "any") else terminated
        if term:
            print(f"Episode ended at step {step}. Resetting...")
            obs, _ = env.reset()

    env.close()
    print("Done!")


if __name__ == "__main__":
    app()
