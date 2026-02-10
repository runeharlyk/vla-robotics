"""Evaluation script for state-based action model."""

from pathlib import Path
from typing import Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import typer
from tqdm import tqdm

from diffusion_policy.state_action_model import StateActionConfig, StateActionModel

app = typer.Typer()


def extract_state_from_obs(obs: dict, env) -> np.ndarray:
    """Extract robot state from observation dictionary.

    This extracts joint positions and velocities from the environment state.
    """
    try:
        state_dict = env.unwrapped.get_state_dict()
        if "articulations" in state_dict:
            art_keys = list(state_dict["articulations"].keys())
            if art_keys:
                states = []
                for art_name in art_keys:
                    art_state = state_dict["articulations"][art_name]
                    if hasattr(art_state, "cpu"):
                        art_state = art_state[0].cpu().numpy()
                    states.append(art_state)
                state = np.concatenate(states) if len(states) > 1 else states[0]
                return state.astype(np.float32)
    except Exception:
        pass

    if isinstance(obs, dict) and "agent" in obs:
        agent_obs = obs["agent"]
        if isinstance(agent_obs, dict):
            qpos = agent_obs.get("qpos", np.zeros(9))
            qvel = agent_obs.get("qvel", np.zeros(9))
            if hasattr(qpos, "cpu"):
                qpos = qpos[0].cpu().numpy()
                qvel = qvel[0].cpu().numpy()
            state = np.concatenate([qpos, qvel])
            return state.astype(np.float32)

    return np.zeros(31, dtype=np.float32)


@app.command()
def evaluate(
    env_id: str = typer.Option("PegInsertionSide-v1", "--env", "-e", help="Environment ID"),
    checkpoint: str = typer.Option("checkpoints/state_model_PegInsertionSide_v1_best.pt", "--checkpoint", "-c"),
    num_episodes: int = typer.Option(10, "--num-episodes", "-n", help="Number of episodes"),
    max_steps: int = typer.Option(200, "--max-steps", help="Max steps per episode"),
    visualize: bool = typer.Option(False, "--vis", help="Visualize with viewer"),
) -> None:
    """Evaluate state-based action model in simulation."""
    print(f"Evaluating state model on {env_id}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Episodes: {num_episodes}")

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        raise typer.Exit(1)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    model = StateActionModel(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model loaded (val_loss={ckpt.get('val_loss', 'N/A'):.4f})")

    render_mode = "human" if visualize else None
    env = gym.make(env_id, num_envs=1, obs_mode="state", render_mode=render_mode)

    successes = []
    returns = []

    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        obs, _ = env.reset()
        episode_return = 0.0
        done = False
        step = 0

        while not done and step < max_steps:
            state = extract_state_from_obs(obs, env)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)

            with torch.no_grad():
                action = model(state_tensor)
                action = action[0].cpu().numpy()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if hasattr(reward, "item"):
                reward = reward.item()
            elif hasattr(reward, "__len__"):
                reward = reward[0]

            episode_return += reward
            step += 1

            if visualize:
                env.render()

        success = info.get("success", False)
        if hasattr(success, "item"):
            success = success.item()
        elif hasattr(success, "__len__"):
            success = success[0]

        successes.append(success)
        returns.append(episode_return)

    env.close()

    success_rate = np.mean(successes)
    mean_return = np.mean(returns)
    std_return = np.std(returns)

    print(f"\nResults:")
    print(f"  Success rate: {success_rate * 100:.1f}%")
    print(f"  Mean return: {mean_return:.2f} ± {std_return:.2f}")


if __name__ == "__main__":
    app()
