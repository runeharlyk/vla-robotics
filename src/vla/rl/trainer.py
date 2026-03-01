"""Training loops for SFT (behavior cloning) and SRPO RL."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from vla.data.dataset import FewDemoDataset
from vla.diagnostics.eval import evaluate, print_metrics
from vla.models.smolvla import SmolVLAPolicy
from vla.rl.rollout import ManiSkillRollout
from vla.rl.srpo_reward import compute_returns, compute_srpo_rewards

logger = logging.getLogger(__name__)


@dataclass
class SFTConfig:
    """Hyperparameters for supervised fine-tuning (behavior cloning)."""

    lr: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 50
    eval_every: int = 5
    eval_episodes: int = 50
    max_steps: int = 200
    save_dir: str = "checkpoints/sft"
    env_id: str = "PickCube-v1"
    seed: int = 42


@dataclass
class SRPOConfig:
    """Hyperparameters for SRPO RL training."""

    lr: float = 1e-5
    num_iterations: int = 100
    trajectories_per_iter: int = 16
    gamma: float = 0.99
    reward_scale: float = 1.0
    eval_every: int = 10
    eval_episodes: int = 50
    max_steps: int = 200
    save_dir: str = "checkpoints/srpo"
    env_id: str = "PickCube-v1"
    seed: int = 42
    mode: str = "srpo"


def train_sft(
    policy: SmolVLAPolicy,
    dataset: FewDemoDataset,
    config: SFTConfig,
    wandb_run: Any | None = None,
) -> SmolVLAPolicy:
    """Run supervised fine-tuning (behavior cloning) on a few-demo dataset.

    Args:
        policy: SmolVLA policy to fine-tune.
        dataset: Few-demo dataset.
        config: SFT hyperparameters.
        wandb_run: Optional wandb run for logging.

    Returns:
        The fine-tuned policy.
    """
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    instruction = dataset.instruction
    save_path = Path(config.save_dir)
    best_success = -1.0

    for epoch in range(1, config.num_epochs + 1):
        policy.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            images = batch["image"].to(policy.device)
            target_actions = batch["action"].to(policy.device)
            out = policy(images, instruction, target_actions)
            loss = out["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"SFT epoch {epoch}/{config.num_epochs}  loss={avg_loss:.6f}")
        if wandb_run is not None:
            wandb_run.log({"sft/loss": avg_loss, "sft/epoch": epoch})

        if epoch % config.eval_every == 0 or epoch == config.num_epochs:
            metrics = evaluate(
                policy_fn=policy.predict_action,
                instruction=instruction,
                env_id=config.env_id,
                num_episodes=config.eval_episodes,
                max_steps=config.max_steps,
                seed=config.seed + 10000,
            )
            print_metrics(metrics, tag=f"SFT epoch {epoch}")
            if wandb_run is not None:
                wandb_run.log({
                    "sft/success_rate": metrics.success_rate,
                    "sft/mean_reward": metrics.mean_reward,
                    "sft/mean_ep_len": metrics.mean_episode_length,
                    "sft/epoch": epoch,
                })
            if metrics.success_rate > best_success:
                best_success = metrics.success_rate
                policy.save_checkpoint(save_path / "best")
                logger.info(f"New best SFT checkpoint: {best_success:.2%}")

    policy.save_checkpoint(save_path / "last")
    return policy


def train_srpo(
    policy: SmolVLAPolicy,
    config: SRPOConfig,
    instruction: str,
    wandb_run: Any | None = None,
) -> SmolVLAPolicy:
    """Run SRPO (or sparse-RL baseline) on top of an SFT-initialised policy.

    When ``config.mode == "srpo"`` the SRPO Tier A shaped rewards are used.
    When ``config.mode == "sparse_rl"`` only the binary environment reward is used.

    Args:
        policy: SFT-initialised SmolVLA policy.
        config: SRPO/RL hyperparameters.
        instruction: Language instruction for the task.
        wandb_run: Optional wandb run for logging.

    Returns:
        The RL-tuned policy.
    """
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)
    rollout_engine = ManiSkillRollout(
        env_id=config.env_id,
        num_envs=1,
        max_steps=config.max_steps,
    )
    save_path = Path(config.save_dir)
    best_success = -1.0

    for iteration in range(1, config.num_iterations + 1):
        policy.eval()
        trajectories = rollout_engine.collect_batch(
            policy_fn=policy.predict_action,
            instruction=instruction,
            num_trajectories=config.trajectories_per_iter,
            seed=config.seed + iteration * 1000,
        )

        num_successes = sum(1 for t in trajectories if t.success)
        logger.info(
            f"Iter {iteration}: collected {len(trajectories)} trajs, "
            f"{num_successes} successes"
        )

        if config.mode == "srpo":
            shaped_rewards = compute_srpo_rewards(
                trajectories,
                gamma=config.gamma,
                reward_scale=config.reward_scale,
            )
        else:
            shaped_rewards = [t.rewards[: t.length].clone() for t in trajectories]

        policy.train()
        total_loss = 0.0
        num_steps_total = 0

        for traj, rewards in zip(trajectories, shaped_rewards):
            returns = compute_returns(rewards, gamma=config.gamma)
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            step_losses = []
            for t in range(traj.length):
                img = traj.images[t].unsqueeze(0).to(policy.device)
                target = traj.actions[t].unsqueeze(0).to(policy.device)
                out = policy(img, instruction, target)
                step_losses.append(out["loss"])

            step_losses_t = torch.stack(step_losses)
            returns_t = returns.to(policy.device)
            loss = (step_losses_t * returns_t).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            num_steps_total += traj.length

        avg_loss = total_loss / max(len(trajectories), 1)
        log_data = {
            f"{config.mode}/loss": avg_loss,
            f"{config.mode}/batch_successes": num_successes,
            f"{config.mode}/iteration": iteration,
        }
        logger.info(f"Iter {iteration}  loss={avg_loss:.6f}  successes={num_successes}")
        if wandb_run is not None:
            wandb_run.log(log_data)

        if iteration % config.eval_every == 0 or iteration == config.num_iterations:
            metrics = evaluate(
                policy_fn=policy.predict_action,
                instruction=instruction,
                env_id=config.env_id,
                num_episodes=config.eval_episodes,
                max_steps=config.max_steps,
                seed=config.seed + 20000,
            )
            print_metrics(metrics, tag=f"{config.mode} iter {iteration}")
            if wandb_run is not None:
                wandb_run.log({
                    f"{config.mode}/success_rate": metrics.success_rate,
                    f"{config.mode}/mean_reward": metrics.mean_reward,
                    f"{config.mode}/mean_ep_len": metrics.mean_episode_length,
                    f"{config.mode}/iteration": iteration,
                })
            if metrics.success_rate > best_success:
                best_success = metrics.success_rate
                policy.save_checkpoint(save_path / "best")
                logger.info(f"New best {config.mode} checkpoint: {best_success:.2%}")

    policy.save_checkpoint(save_path / "last")
    rollout_engine.close()
    return policy
