"""
Evaluate VLA policies on ManiSkill environments.

Supports evaluating trained RT-1 models and other VLA policies.

Usage:
    uv run python src/vla/evaluate.py rt1 --model models/rt1_pickcube_v1.pt
    uv run python src/vla/evaluate.py --help
"""
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import typer
from PIL import Image

from vla.train import create_rt1_model

app = typer.Typer(no_args_is_help=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


DEFAULT_ACTION_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -1.0], dtype=np.float32)
DEFAULT_ACTION_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 1.0], dtype=np.float32)


def bins_to_continuous(bins: torch.Tensor, num_bins: int = 256) -> torch.Tensor:
    return (bins.float() / (num_bins - 1)) * 2 - 1


def denormalize_action(action_norm: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return (action_norm + 1.0) / 2.0 * (high - low) + low


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
    use_pretrained = config.get("pretrained", False)
    image_size = config.get("image_size", 256)
    sequence_length = config.get("sequence_length", 1)
    action_low = np.array(config.get("action_low", DEFAULT_ACTION_LOW), dtype=np.float32)
    action_high = np.array(config.get("action_high", DEFAULT_ACTION_HIGH), dtype=np.float32)

    model = create_rt1_model(action_dim, device, model_size, pretrained=use_pretrained)
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
    print(f"  Sequence length: {sequence_length}")
    print(f"  Image size: {image_size}")
    print(f"  Pretrained backbone: {use_pretrained}")
    print(f"  Instruction: '{instruction}'")

    total_rewards = []
    successes = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        episode_reward = 0.0
        frame_buffer: deque[torch.Tensor] = deque(maxlen=sequence_length)

        for step in range(max_steps):
            frame = get_frame(env)
            rgb = np.array(Image.fromarray(frame).resize((image_size, image_size)))

            img_tensor = torch.from_numpy(rgb).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)
            frame_buffer.append(img_tensor)

            while len(frame_buffer) < sequence_length:
                frame_buffer.appendleft(frame_buffer[0])

            frames = torch.stack(list(frame_buffer), dim=0)
            video = frames.unsqueeze(0).to(device)
            video = video.permute(0, 2, 1, 3, 4)

            with torch.no_grad():
                logits = model(video, texts=[instruction])
                action_bins = logits.argmax(dim=-1)[0, -1]

            action_norm = bins_to_continuous(action_bins).cpu().numpy()
            action = denormalize_action(action_norm, action_low, action_high)
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


@app.command()
def rt1_tf(
    checkpoint_path: str = typer.Option(..., "--checkpoint", "-c", help="Path to TF SavedModel checkpoint"),
    env_id: str = typer.Option("PickCube-v1", "--env", "-e", help="Environment ID"),
    instruction: str = typer.Option("pick up the cube", "--instruction", "-i", help="Task instruction"),
    num_episodes: int = typer.Option(10, "--num-episodes", "-n", help="Number of test episodes"),
    max_steps: int = typer.Option(100, "--max-steps", "-s", help="Max steps per episode"),
    policy_setup: str = typer.Option("google_robot", "--policy-setup", "-p", help="Policy setup: google_robot or widowx_bridge"),
    save_video: bool = typer.Option(False, "--save-video", help="Save video of episodes"),
    output_dir: str = typer.Option("outputs/rt1_tf_eval", "--output-dir", "-o", help="Output directory"),
) -> None:
    """Evaluate official Google RT-1 TensorFlow checkpoint."""
    from vla.rt1_tf_policy import RT1TFPolicy

    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        typer.echo(f"Checkpoint not found: {checkpoint_path}", err=True)
        typer.echo("Download with:")
        typer.echo("  gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/open_x_embodiment_and_rt_x_oss/rt_1_x_tf_trained_for_002272480_step.zip .")
        raise typer.Exit(1)

    print(f"Loading RT-1 TF policy from {checkpoint_path}")
    policy = RT1TFPolicy(
        checkpoint_path=str(checkpoint),
        policy_setup=policy_setup,
    )
    policy.load()
    policy.reset(instruction)

    print(f"Creating environment: {env_id}")
    env = create_env(env_id)

    if save_video:
        from mani_skill.utils.wrappers import RecordEpisode
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        env = RecordEpisode(env, output_dir=str(output_path), save_video=True, video_fps=30)

    print(f"\nEvaluating RT-1 TF on {env_id}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps: {max_steps}")
    print(f"  Instruction: '{instruction}'")
    print(f"  Policy setup: {policy_setup}")

    total_rewards = []
    successes = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        episode_reward = 0.0

        for step in range(max_steps):
            frame = get_frame(env)
            action = policy.predict_action(frame, instruction)
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


if __name__ == "__main__":
    app()
