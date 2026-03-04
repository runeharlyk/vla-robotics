"""SRPO / sparse-RL training on top of an SFT-initialised SmolVLA policy.

Supports both ManiSkill and LIBERO simulators with vectorised rollouts.

Usage:
    # ManiSkill (default, vectorised with 16 parallel envs):
    uv run python scripts/train_srpo.py --sft-checkpoint checkpoints/sft/best --num-rollout-envs 16

    # LIBERO (subprocess-vectorised):
    uv run python scripts/train_srpo.py --sft-checkpoint checkpoints/sft/best --simulator libero --suite spatial --num-rollout-envs 8

    # Sparse RL ablation:
    uv run python scripts/train_srpo.py --sft-checkpoint ... --mode sparse_rl

    # SRPO with demo seeding:
    uv run python scripts/train_srpo.py --sft-checkpoint ... --data data/preprocessed/peginsertionside.pt --mode srpo
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

import wandb
from vla.data.dataset import FewDemoDataset
from vla.models.smolvla import SmolVLAPolicy
from vla.rl.rollout import Trajectory
from vla.rl.trainer import SRPOConfig, train_srpo
from vla.utils import get_device, seed_everything

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _discover_data(data_path: Path | None) -> Path:
    """Return a concrete .pt path (same logic as train_sft.py)."""
    if data_path is not None and data_path.is_file():
        return data_path
    search = data_path if data_path is not None and data_path.is_dir() else PREPROCESSED_DIR
    pts = sorted(search.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt files in {search}. Run preprocess_data.py first.")
    if len(pts) > 1:
        names = ", ".join(p.name for p in pts)
        logging.warning("Multiple .pt files: %s. Using %s.", names, pts[0].name)
    return pts[0]


def main(
    sft_checkpoint: Path = typer.Option(..., "--sft-checkpoint", "-s", path_type=Path),
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    data_path: Path = typer.Option(
        None, "--data", "-d", path_type=Path, help="Preprocessed .pt file (for demo seeding)"
    ),
    num_demos: int = typer.Option(5, "--num-demos", "-n"),
    mode: str = typer.Option("srpo", "--mode", "-m", help="srpo or sparse_rl"),
    simulator: str = typer.Option("maniskill", "--simulator", help="maniskill or libero"),
    suite: str = typer.Option("spatial", "--suite", help="LIBERO suite (spatial, object, goal, long)"),
    task_id: int = typer.Option(0, "--task-id", help="LIBERO task index within the suite"),
    num_rollout_envs: int = typer.Option(1, "--num-rollout-envs", help="Parallel envs for vectorised rollouts"),
    lr: float = typer.Option(1e-5, "--lr"),
    num_iterations: int = typer.Option(100, "--iterations"),
    trajectories_per_iter: int = typer.Option(16, "--trajs-per-iter"),
    ppo_epochs: int = typer.Option(4, "--ppo-epochs"),
    clip_epsilon: float = typer.Option(0.2, "--clip-epsilon"),
    kl_coeff: float = typer.Option(0.01, "--kl-coeff"),
    gamma: float = typer.Option(0.99, "--gamma"),
    reward_scale: float = typer.Option(1.0, "--reward-scale"),
    eval_every: int = typer.Option(10, "--eval-every"),
    eval_episodes: int = typer.Option(50, "--eval-episodes"),
    max_steps: int = typer.Option(None, "--max-steps", help="Override max steps (default: from checkpoint metadata)"),
    seed: int = typer.Option(42, "--seed"),
    env_id: str = typer.Option(None, "--env", help="Override env id (default: from checkpoint metadata)"),
    instruction: str = typer.Option(None, "--instruction", help="Override instruction (default: from checkpoint)"),
    world_model: str = typer.Option("dinov2", "--world-model", help="dinov2 or vjepa2"),
    subsample_every: int = typer.Option(5, "--subsample-every"),
    dbscan_eps: float = typer.Option(0.5, "--dbscan-eps"),
    dbscan_min_samples: int = typer.Option(2, "--dbscan-min-samples"),
    use_wandb: bool = typer.Option(True, "--wandb/--no-wandb"),
) -> None:
    """Run SRPO or sparse-RL training starting from an SFT checkpoint.

    ``env_id``, ``instruction``, and ``control_mode`` are loaded from the
    SFT checkpoint's saved metadata unless explicitly overridden via CLI.

    Use ``--num-rollout-envs N`` to enable vectorised parallel rollouts
    (recommended: set equal to ``--trajs-per-iter`` for full parallelism).
    """
    seed_everything(seed)
    device = get_device()

    # ── Load SFT policy and its saved metadata ───────────────────────────
    pt_path = _discover_data(data_path)
    dataset = FewDemoDataset(pt_path, num_demos=num_demos, seed=seed)

    policy = SmolVLAPolicy(
        checkpoint=checkpoint,
        action_dim=dataset.action_dim,
        state_dim=dataset.state_dim,
        device=str(device),
    )
    env_meta = policy.load_checkpoint(sft_checkpoint)
    logging.info("Loaded SFT checkpoint from %s (env_metadata=%s)", sft_checkpoint, env_meta)

    resolved_env_id = env_id or env_meta.get("env_id") or dataset.metadata.get("env_id", "PickCube-v1")
    resolved_instruction = (
        instruction
        or env_meta.get("instruction")
        or dataset.metadata.get("instruction", "complete the manipulation task")
    )
    resolved_max_steps = max_steps or 200

    logging.info(
        "RL training: mode=%s  simulator=%s  env_id=%s  instruction=%r  max_steps=%d  num_rollout_envs=%d",
        mode,
        simulator,
        resolved_env_id,
        resolved_instruction,
        resolved_max_steps,
        num_rollout_envs,
    )

    # ── Build demo trajectories for SRPO reward seeding ──────────────────
    demo_trajectories: list[Trajectory] | None = None
    if mode == "srpo":
        demo_trajectories = dataset.episodes_as_trajectories()
        logging.info("Built %d demo trajectories for reference seeding", len(demo_trajectories))

    task_tag = resolved_env_id.lower().replace("-", "_")

    config = SRPOConfig(
        lr=lr,
        num_iterations=num_iterations,
        trajectories_per_iter=trajectories_per_iter,
        ppo_epochs=ppo_epochs,
        clip_epsilon=clip_epsilon,
        kl_coeff=kl_coeff,
        gamma=gamma,
        reward_scale=reward_scale,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        max_steps=resolved_max_steps,
        save_dir=str(PROJECT_ROOT / "checkpoints" / mode / f"{task_tag}_seed{seed}"),
        env_id=resolved_env_id,
        seed=seed,
        mode=mode,
        world_model_type=world_model,
        subsample_every=subsample_every,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        simulator=simulator,
        suite=suite,
        task_id=task_id,
        state_dim=dataset.state_dim,
        num_rollout_envs=num_rollout_envs,
    )

    run = None
    if use_wandb:
        run = wandb.init(
            project="srpo-smolvla",
            name=f"{mode}_{task_tag}_seed{seed}",
            config={
                "method": mode,
                "task": resolved_env_id,
                "instruction": resolved_instruction,
                "seed": seed,
                "lr": lr,
                "num_iterations": num_iterations,
                "trajectories_per_iter": trajectories_per_iter,
                "ppo_epochs": ppo_epochs,
                "clip_epsilon": clip_epsilon,
                "kl_coeff": kl_coeff,
                "gamma": gamma,
                "reward_scale": reward_scale,
                "checkpoint": checkpoint,
                "sft_checkpoint": str(sft_checkpoint),
                "env_id": resolved_env_id,
                "world_model": world_model,
                "num_demos": num_demos,
                "simulator": simulator,
                "suite": suite,
                "num_rollout_envs": num_rollout_envs,
            },
        )

    train_srpo(policy, config, resolved_instruction, demo_trajectories=demo_trajectories, wandb_run=run)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    typer.run(main)
