"""Reinforcement-learning fine-tuning loop (online rollouts).

This module provides a generic RL training skeleton that works with any
simulator backend via the :class:`SimEnv` protocol.  Policy gradient
details (loss, advantage estimation) are delegated to algorithm-specific
helpers so this loop can drive PPO, SRPO, etc.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import typer
from tqdm import tqdm

import wandb
from vla.constants import ACTION_DIM, MANISKILL_TASKS, MODELS_DIR
from vla.envs import make_env_factory

app = typer.Typer(no_args_is_help=True)


def _collect_trajectory(
    policy,
    env,
    max_steps: int,
    device: torch.device,
) -> dict:
    """Roll out one episode, collecting transitions."""
    obs_raw, info = env.reset(seed=int(time.time()) % 2**31)
    policy.reset()

    observations: list[dict] = []
    actions_list: list[torch.Tensor] = []
    rewards: list[float] = []
    dones: list[bool] = []

    for _step in range(max_steps):
        batch = env.obs_to_batch(obs_raw, device=device)
        observations.append(batch)

        with torch.inference_mode():
            action = policy.select_action(batch)

        action_np = action.to("cpu").numpy()
        if action_np.ndim == 2:
            action_np = action_np[0]

        actions_list.append(action.detach().cpu())
        obs_raw, reward, terminated, truncated, info = env.step(action_np)
        rewards.append(float(reward))
        done = terminated or truncated or env.is_success(info)
        dones.append(done)

        if done:
            break

    return {
        "observations": observations,
        "actions": actions_list,
        "rewards": rewards,
        "dones": dones,
        "success": env.is_success(info),
    }


@app.command()
def train_rl(
    checkpoint: str = typer.Option(..., "--checkpoint", "-c"),
    simulator: str = typer.Option("libero", "--simulator"),
    suite: str = typer.Option("long", "--suite", "-s"),
    env_id: str | None = typer.Option(None, "--env-id"),
    action_dim: int = typer.Option(ACTION_DIM, "--action-dim"),
    num_iterations: int = typer.Option(100, "--num-iterations"),
    trajectories_per_iter: int = typer.Option(16, "--trajectories-per-iter"),
    lr: float = typer.Option(1e-5, "--lr"),
    max_steps: int = typer.Option(300, "--max-steps"),
    device: str = typer.Option("cuda", "--device", "-d"),
    save_path: str | None = typer.Option(None, "--save"),
    wandb_project: str = typer.Option("vla-rl", "--wandb-project"),
    wandb_name: str | None = typer.Option(None, "--wandb-name"),
) -> None:
    """Online RL fine-tuning of a pre-trained policy."""
    device_obj = torch.device(device if torch.cuda.is_available() else "cpu")

    from vla.models import load_policy

    loaded = load_policy("smolvla", checkpoint, device)
    policy = loaded.policy
    policy.eval()

    sim = simulator.lower()
    factory_kwargs: dict = {}
    if sim == "libero":
        factory_kwargs["suite"] = suite
    elif sim == "maniskill":
        if not env_id:
            raise typer.BadParameter("--env-id is required for ManiSkill RL training")
        task_meta = MANISKILL_TASKS.get(env_id, {})
        factory_kwargs["env_id"] = env_id
        factory_kwargs["instruction"] = task_meta.get("instruction", env_id)
        factory_kwargs["max_episode_steps"] = task_meta.get("max_episode_steps")

    env_factory = make_env_factory(sim, **factory_kwargs)

    label = f"rl_{sim}_{env_factory.suite_name}"
    if save_path is None:
        save_path = str(MODELS_DIR / f"{label}.pt")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(  # noqa: F841 — used once PPO/SRPO loss is wired
        [p for p in policy.parameters() if p.requires_grad],
        lr=lr,
    )

    wandb.init(
        project=wandb_project,
        name=wandb_name or label,
        config={
            "simulator": sim,
            "checkpoint": checkpoint,
            "num_iterations": num_iterations,
            "trajectories_per_iter": trajectories_per_iter,
            "lr": lr,
            "max_steps": max_steps,
        },
    )

    best_success_rate = 0.0

    for iteration in tqdm(range(num_iterations), desc="RL iterations"):
        trajectories = []
        for task_id in range(env_factory.num_tasks):
            env = env_factory(task_id)
            for _ in range(trajectories_per_iter):
                traj = _collect_trajectory(policy, env, max_steps, device_obj)
                trajectories.append(traj)
            env.close()

        # ---- compute RL loss (placeholder — plug in PPO / SRPO here) ----
        total_reward = sum(sum(t["rewards"]) for t in trajectories)
        n_success = sum(1 for t in trajectories if t["success"])
        n_total = len(trajectories)
        success_rate = n_success / max(n_total, 1)

        # TODO: replace with actual policy gradient loss
        # loss = compute_policy_gradient_loss(policy, trajectories, ...)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()

        wandb.log(
            {
                "rl/iteration": iteration,
                "rl/total_reward": total_reward,
                "rl/success_rate": success_rate,
                "rl/n_trajectories": n_total,
            },
            step=iteration,
        )

        if success_rate > best_success_rate:
            best_success_rate = success_rate
            torch.save(
                {
                    "model_state_dict": policy.state_dict(),
                    "iteration": iteration,
                    "success_rate": success_rate,
                    "config": {
                        "simulator": sim,
                        "checkpoint": checkpoint,
                    },
                },
                save_path,
            )

        tqdm.write(
            f"  Iter {iteration}: reward={total_reward:.1f}, "
            f"SR={success_rate * 100:.0f}% (best={best_success_rate * 100:.0f}%)"
        )

    wandb.finish()
    print(f"\nRL training complete. Best SR: {best_success_rate * 100:.1f}%")
    print(f"Model: {save_path}")


if __name__ == "__main__":
    app()
