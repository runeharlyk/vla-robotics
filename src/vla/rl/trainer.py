"""SRPO reinforcement-learning training loop.

Implements the full SRPO algorithm (Section 3.3 of the paper):
  1. Collect M trajectories with π_θ_old.
  2. Compute world-progress trajectory rewards g_i via the world model.
  3. Compute trajectory-level advantages  = (g_i − μ_g) / σ_g.
  4. Cache per-step flow-matching losses under θ_old (fixed noise/time).
  5. For each PPO-epoch:
       - Recompute per-step FM losses under θ → importance ratio
       - Clipped surrogate loss + KL regularisation against π_ref.
  6. Update θ_old ← θ, add any new successes to reference set.

The rollout engine is **simulator-agnostic**: pass in any object that
implements :class:`~vla.rl.rollout.RolloutEngine` (ManiSkill or LIBERO).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vla.diagnostics.eval import evaluate_smolvla, print_metrics
from vla.models.smolvla import SmolVLAPolicy
from vla.models.world_model import WorldModelEncoder, build_world_model
from vla.rl.rollout import ManiSkillRollout, RolloutEngine, Trajectory
from vla.rl.srpo_reward import SRPORewardConfig, WorldProgressReward

logger = logging.getLogger(__name__)


@dataclass
class SRPOConfig:
    """Hyperparameters for SRPO RL training."""

    lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-10
    max_grad_norm: float = 10.0
    num_iterations: int = 100
    trajectories_per_iter: int = 16
    ppo_epochs: int = 4
    clip_epsilon: float = 0.2
    kl_coeff: float = 0.01
    eval_every: int = 10
    eval_episodes: int = 50
    max_steps: int = 200
    save_dir: str = "checkpoints/srpo"
    env_id: str = "PickCube-v1"
    seed: int = 42
    mode: str = "srpo"
    world_model_type: str = "dinov2"
    subsample_every: int = 5
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    num_fm_noise_samples: int = 4
    gamma: float = 0.99
    reward_scale: float = 1.0
    simulator: str = "maniskill"
    suite: str = "spatial"
    task_id: int = 0
    state_dim: int = 0
    num_rollout_envs: int = 1


# ---------------------------------------------------------------------------
# Rollout engine factory
# ---------------------------------------------------------------------------

def build_rollout_engine(config: SRPOConfig) -> RolloutEngine:
    """Create a rollout engine from the config's simulator setting."""
    sim = config.simulator.lower()

    if sim == "maniskill":
        return ManiSkillRollout(
            env_id=config.env_id,
            num_envs=config.num_rollout_envs,
            max_steps=config.max_steps,
        )

    if sim == "libero":
        from vla.rl.libero_rollout import LiberoRollout

        return LiberoRollout(
            suite_name=config.suite,
            task_id=config.task_id,
            num_envs=config.num_rollout_envs,
            max_steps=config.max_steps,
            state_dim=config.state_dim,
        )

    raise ValueError(f"Unknown simulator {config.simulator!r}. Available: maniskill, libero")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_fm_loss_per_step(
    policy: SmolVLAPolicy,
    traj: Trajectory,
    instruction: str,
    fixed_noise: list[torch.Tensor],
    fixed_time: list[torch.Tensor],
) -> torch.Tensor:
    """Compute per-timestep flow-matching loss for a trajectory.

    Uses pre-sampled noise/time so that the same random quantities are
    used under both θ_old and θ_new for valid importance-ratio estimation.
    Multiple noise/time samples are averaged for variance reduction.

    Returns:
        ``(T,)`` mean FM loss per timestep.
    """
    T = traj.length
    step_losses = []
    for t in range(T):
        img = traj.images[t].unsqueeze(0).to(policy.device)
        action = traj.actions[t].unsqueeze(0).to(policy.device)

        imgs_f = policy._to_float01(img).to(policy.device, dtype=policy.dtype)
        img_list, mask_list = policy._prepare_images(imgs_f)
        tokens, tmasks = policy._tokenize(instruction, batch_size=1)
        state_raw = traj.states[t].unsqueeze(0) if traj.states is not None else None
        state = policy._prepare_state_input(state_raw, batch_size=1)
        normalized_action = policy._normalize_action(action.to(policy.device, dtype=policy.dtype))
        action_padded = policy._prepare_action(normalized_action)
        action_padded = action_padded.unsqueeze(1).expand(-1, policy.chunk_size, -1)

        losses_per_sample = policy.model.forward(
            img_list,
            mask_list,
            tokens,
            tmasks,
            state,
            action_padded,
            noise=fixed_noise[t].to(policy.device, dtype=policy.dtype),
            time=fixed_time[t].to(policy.device, dtype=policy.dtype),
        )
        step_losses.append(losses_per_sample[:, :, : policy.max_action_dim].mean().detach())

    return torch.stack(step_losses)


def _sample_fixed_noise_time(
    traj: Trajectory,
    policy: SmolVLAPolicy,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Pre-sample noise and time tensors for each timestep in a trajectory."""
    T = traj.length
    noise_list = []
    time_list = []
    for _ in range(T):
        n = torch.randn(1, policy.chunk_size, policy.max_action_dim)
        beta = torch.distributions.Beta(1.5, 1.0)
        t = beta.sample((1,)) * 0.999 + 0.001
        noise_list.append(n)
        time_list.append(t)
    return noise_list, time_list


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_srpo(
    policy: SmolVLAPolicy,
    config: SRPOConfig,
    instruction: str,
    demo_trajectories: list[Trajectory] | None = None,
    wandb_run: Any | None = None,
    rollout_engine: RolloutEngine | None = None,
) -> SmolVLAPolicy:
    """Run SRPO training on top of an SFT-initialised policy.

    Implements the full SRPO algorithm from the paper:
      - World-progress reward shaping via world-model embeddings + DBSCAN
      - Trajectory-level advantages (GRPO-style)
      - Clipped surrogate objective with importance sampling
      - KL regularisation against the reference (SFT) policy

    When ``config.mode == "sparse_rl"`` the world-model rewards are
    disabled and only the binary environment reward is used (ablation).

    Args:
        policy: SFT-initialised SmolVLA policy (becomes π_θ *and* π_ref).
        config: SRPO hyperparameters.
        instruction: Language instruction for the task.
        demo_trajectories: Optional list of demo trajectories to seed the
            reference set.  Their images are used for world-model encoding.
        wandb_run: Optional wandb run for logging.
        rollout_engine: Pre-built rollout engine.  When ``None`` one is
            created from ``config`` via :func:`build_rollout_engine`.

    Returns:
        The RL-tuned policy.
    """
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.lr,
        betas=config.betas,
        weight_decay=config.weight_decay,
    )

    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    world_encoder: WorldModelEncoder | None = None
    reward_model: WorldProgressReward | None = None

    if config.mode == "srpo":
        world_encoder = build_world_model(
            model_type=config.world_model_type,
            device=str(policy.device),
        )
        reward_cfg = SRPORewardConfig(
            subsample_every=config.subsample_every,
            dbscan_eps=config.dbscan_eps,
            dbscan_min_samples=config.dbscan_min_samples,
        )
        reward_model = WorldProgressReward(world_encoder, reward_cfg)

        if demo_trajectories:
            demo_images = []
            for dt in demo_trajectories:
                    imgs = dt.images[: dt.length]
                    imgs = imgs.float() / 255.0 if imgs.dtype == torch.uint8 else imgs.float()
                    demo_images.append(imgs)
            reward_model.add_demo_trajectories(demo_images)

    if rollout_engine is None:
        rollout_engine = build_rollout_engine(config)

    use_vectorized = config.num_rollout_envs > 1

    save_path = Path(config.save_dir)
    best_success = -1.0

    _save_meta = dict(
        env_id=config.env_id,
        instruction=instruction,
        control_mode="pd_joint_delta_pos",
    )

    for iteration in range(1, config.num_iterations + 1):
        # ── 1. Collect trajectories with current policy ─────────────────
        policy.eval()
        trajectories = rollout_engine.collect_batch(
            policy_fn=policy.predict_action,
            instruction=instruction,
            num_trajectories=config.trajectories_per_iter,
            seed=config.seed + iteration * 1000,
            policy_batch_fn=policy.predict_action_batch if use_vectorized else None,
        )
        num_successes = sum(1 for t in trajectories if t.success)
        logger.info(f"Iter {iteration}: collected {len(trajectories)} trajs, {num_successes} successes")

        # ── 2. Compute trajectory-level rewards g_i ─────────────────────
        if config.mode == "srpo" and reward_model is not None:
            g_values = reward_model.compute_trajectory_rewards(trajectories)
            reward_model.add_successful_trajectories([t for t in trajectories if t.success])
        else:
            g_values = [1.0 if t.success else 0.0 for t in trajectories]

        # ── 3. Compute trajectory-level advantages  ──────────────────
        g_tensor = torch.tensor(g_values, dtype=torch.float32)
        g_mean = g_tensor.mean()
        g_std = g_tensor.std().clamp(min=1e-8)
        advantages = ((g_tensor - g_mean) / g_std).tolist()

        # ── 4. Cache per-step FM loss under θ_old (+ sample fixed noise/time)
        policy.eval()
        old_losses_per_traj: list[torch.Tensor] = []
        fixed_noise_per_traj: list[list[torch.Tensor]] = []
        fixed_time_per_traj: list[list[torch.Tensor]] = []

        with torch.no_grad():
            for traj in trajectories:
                noise_list, time_list = _sample_fixed_noise_time(traj, policy)
                fixed_noise_per_traj.append(noise_list)
                fixed_time_per_traj.append(time_list)
                old_loss = _compute_fm_loss_per_step(policy, traj, instruction, noise_list, time_list)
                old_losses_per_traj.append(old_loss)

        # ── 5. PPO epochs ───────────────────────────────────────────────
        policy.train()
        total_surrogate = 0.0
        total_kl = 0.0

        for _ppo_epoch in range(config.ppo_epochs):
            epoch_loss_sum = 0.0
            for i, traj in enumerate(trajectories):
                adv_i = advantages[i]
                old_losses = old_losses_per_traj[i]
                noise_list = fixed_noise_per_traj[i]
                time_list = fixed_time_per_traj[i]

                new_step_losses = []
                for t in range(traj.length):
                    img = traj.images[t].unsqueeze(0).to(policy.device)
                    action = traj.actions[t].unsqueeze(0).to(policy.device)

                    imgs_f = policy._to_float01(img).to(policy.device, dtype=policy.dtype)
                    img_list, mask_list = policy._prepare_images(imgs_f)
                    tokens, tmasks = policy._tokenize(instruction, batch_size=1)
                    state_raw = traj.states[t].unsqueeze(0) if traj.states is not None else None
                    state = policy._prepare_state_input(state_raw, batch_size=1)
                    normalized_action = policy._normalize_action(action.to(policy.device, dtype=policy.dtype))
                    action_padded = policy._prepare_action(normalized_action)
                    action_padded = action_padded.unsqueeze(1).expand(-1, policy.chunk_size, -1)

                    fm_loss = policy.model.forward(
                        img_list,
                        mask_list,
                        tokens,
                        tmasks,
                        state,
                        action_padded,
                        noise=noise_list[t].to(policy.device, dtype=policy.dtype),
                        time=time_list[t].to(policy.device, dtype=policy.dtype),
                    )
                    new_step_losses.append(fm_loss[:, :, : policy.max_action_dim].mean())

                if not new_step_losses:
                    continue

                new_losses_t = torch.stack(new_step_losses)
                old_losses_t = old_losses.to(policy.device)

                log_ratios = old_losses_t - new_losses_t
                ratios = torch.exp(log_ratios.clamp(-10.0, 10.0))

                adv_t = torch.full_like(ratios, adv_i)

                surr1 = ratios * adv_t
                surr2 = torch.clamp(ratios, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * adv_t
                clip_loss = -torch.min(surr1, surr2).mean()

                with torch.no_grad():
                    ref_losses = _compute_fm_loss_per_step(ref_policy, traj, instruction, noise_list, time_list).to(
                        policy.device
                    )
                kl_approx = (ref_losses - new_losses_t.detach()).mean()
                kl_penalty = config.kl_coeff * kl_approx

                loss = clip_loss + kl_penalty

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=config.max_grad_norm)
                optimizer.step()

                epoch_loss_sum += clip_loss.item()
                total_kl += kl_penalty.item()

            total_surrogate += epoch_loss_sum

        avg_surr = total_surrogate / max(config.ppo_epochs * len(trajectories), 1)
        avg_kl = total_kl / max(config.ppo_epochs * len(trajectories), 1)

        # ── 6. Update θ_old for next iteration's ratio computation ──────
        old_losses_per_traj.clear()
        fixed_noise_per_traj.clear()
        fixed_time_per_traj.clear()

        log_data = {
            f"{config.mode}/surrogate_loss": avg_surr,
            f"{config.mode}/kl_penalty": avg_kl,
            f"{config.mode}/batch_successes": num_successes,
            f"{config.mode}/mean_g": g_mean.item(),
            f"{config.mode}/iteration": iteration,
        }
        logger.info(
            f"Iter {iteration}  surr={avg_surr:.6f}  kl={avg_kl:.6f}  successes={num_successes}  g_mean={g_mean:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(log_data)

        if iteration % config.eval_every == 0 or iteration == config.num_iterations:
            metrics = evaluate_smolvla(
                policy,
                instruction=instruction,
                simulator=config.simulator,
                env_id=config.env_id,
                num_episodes=config.eval_episodes,
                max_steps=config.max_steps,
                seed=config.seed + 20000,
                suite=config.suite,
            )
            print_metrics(metrics, tag=f"{config.mode} iter {iteration}")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        f"{config.mode}/success_rate": metrics.success_rate,
                        f"{config.mode}/mean_reward": metrics.mean_reward,
                        f"{config.mode}/mean_ep_len": metrics.mean_episode_length,
                        f"{config.mode}/iteration": iteration,
                    }
                )
            if metrics.success_rate > best_success:
                best_success = metrics.success_rate
                policy.save_checkpoint(save_path / "best", **_save_meta)
                logger.info(f"New best {config.mode} checkpoint: {best_success:.2%}")

    policy.save_checkpoint(save_path / "last", **_save_meta)
    rollout_engine.close()
    return policy
