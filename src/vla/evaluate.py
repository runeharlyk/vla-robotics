"""
Evaluate VLA policies on ManiSkill environments.

Supports evaluating trained RT-1 models and other VLA policies.

Usage:
    uv run python src/vla/evaluate.py rt1 --model models/rt1_pickcube_v1.pt
    uv run python src/vla/evaluate.py --help
"""
import time
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import typer
from PIL import Image

import mani_skill.envs

app = typer.Typer(no_args_is_help=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


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


def create_env(env_id: str, render: bool = False):
    """Create ManiSkill environment with consistent settings."""
    return gym.make(
        env_id,
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
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


@app.command()
def rt1(
    model_path: str = typer.Option(..., "--model", "-m", help="Path to trained model"),
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="Environment ID"),
    num_episodes: int = typer.Option(20, "--num-episodes", "-n", help="Number of test episodes"),
    max_steps: int = typer.Option(100, "--max-steps", "-s", help="Max steps per episode"),
    device: str = typer.Option("cuda", "--device", "-d", help="Device"),
    save_video: bool = typer.Option(False, "--save-video", help="Save video of episodes"),
    output_dir: str = typer.Option("outputs/eval", "--output-dir", "-o", help="Output directory"),
) -> None:
    """Evaluate trained RT-1 model on ManiSkill environment."""
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

    print(f"Creating environment: {env_id}")
    env = create_env(env_id)

    if save_video:
        from mani_skill.utils.wrappers import RecordEpisode
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        env = RecordEpisode(env, output_dir=str(output_path), save_video=True, video_fps=30)

    print(f"\nEvaluating RT-1 on {env_id}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps: {max_steps}")
    print(f"  Instruction: '{instruction}'")

    total_rewards = []
    successes = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        episode_reward = 0.0

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

            done = terminated.any() if hasattr(terminated, "any") else terminated
            if done:
                break

        success = info.get("success", False)
        if hasattr(success, "item"):
            success = success.item()

        total_rewards.append(episode_reward)
        successes.append(success)
        print(f"Episode {ep + 1}: Reward={episode_reward:.2f}, Success={success}")

    env.close()

    print("\n=== Evaluation Results ===")
    print(f"Episodes: {num_episodes}")
    print(f"Mean Reward: {np.mean(total_rewards):.2f} (+/- {np.std(total_rewards):.2f})")
    print(f"Success Rate: {np.mean(successes) * 100:.1f}%")

    if save_video:
        print(f"Videos saved to: {output_dir}")


@app.command()
def dummy(
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="Environment ID"),
    num_episodes: int = typer.Option(5, "--num-episodes", "-n", help="Number of test episodes"),
    max_steps: int = typer.Option(100, "--max-steps", "-s", help="Max steps per episode"),
) -> None:
    """Evaluate random policy (baseline) on ManiSkill environment."""
    print(f"Creating environment: {env_id}")
    env = create_env(env_id)

    print(f"\nEvaluating random policy on {env_id}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps: {max_steps}")

    total_rewards = []
    successes = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        episode_reward = 0.0

        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward.item() if hasattr(reward, "item") else reward)

            done = terminated.any() if hasattr(terminated, "any") else terminated
            if done:
                break

        success = info.get("success", False)
        if hasattr(success, "item"):
            success = success.item()

        total_rewards.append(episode_reward)
        successes.append(success)
        print(f"Episode {ep + 1}: Reward={episode_reward:.2f}, Success={success}")

    env.close()

    print("\n=== Evaluation Results ===")
    print(f"Episodes: {num_episodes}")
    print(f"Mean Reward: {np.mean(total_rewards):.2f} (+/- {np.std(total_rewards):.2f})")
    print(f"Success Rate: {np.mean(successes) * 100:.1f}%")


if __name__ == "__main__":
    app()
