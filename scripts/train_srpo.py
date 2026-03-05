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

    # Multi-task SRPO across an entire LIBERO suite:
    uv run python scripts/train_srpo.py --sft-checkpoint ... --simulator libero --suite spatial --multitask --data-dir data/preprocessed/spatial --mode srpo
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

import wandb
from vla.data.dataset import FewDemoDataset
from vla.models.smolvla import SmolVLAPolicy
from vla.constants import CHECKPOINTS_DIR, PREPROCESSED_DIR
from vla.rl.rollout import Trajectory
from vla.rl.trainer import SRPOConfig, TaskSpec, train_srpo, train_srpo_multitask
from vla.utils import get_device, run_id, seed_everything

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
    sft_checkpoint: Path = typer.Option(None, "--sft-checkpoint", "-s", path_type=Path),
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    data_path: Path = typer.Option(
        None, "--data", "-d", path_type=Path, help="Preprocessed .pt file (for demo seeding, single-task)"
    ),
    data_dir: Path = typer.Option(
        None, "--data-dir", path_type=Path, help="Directory of .pt files (one per task) for multi-task SRPO"
    ),
    num_demos: int = typer.Option(5, "--num-demos", "-n"),
    mode: str = typer.Option("srpo", "--mode", "-m", help="srpo or sparse_rl"),
    simulator: str = typer.Option("maniskill", "--simulator", help="maniskill or libero"),
    suite: str = typer.Option("spatial", "--suite", help="LIBERO suite (spatial, object, goal, long)"),
    task_id: int = typer.Option(0, "--task-id", help="LIBERO task index within the suite"),
    multitask: bool = typer.Option(False, "--multitask/--single-task", help="Enable multi-task SRPO training"),
    trajs_per_task: int = typer.Option(4, "--trajs-per-task", help="Trajectories per task per iteration (multi-task)"),
    num_rollout_envs: int = typer.Option(1, "--num-rollout-envs", help="Parallel envs per task for vectorised rollouts"),
    fm_batch_size: int = typer.Option(32, "--fm-batch-size", help="Timesteps per FM forward pass in PPO"),
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

    Use ``--multitask`` with ``--data-dir`` pointing to a directory of per-task
    ``.pt`` files to run multi-task SRPO with per-task reward models.
    """
    seed_everything(seed)
    device = get_device()

    from vla.constants import ACTION_DIM, MANISKILL_TASKS

    resolved_max_steps = max_steps or 200

    if multitask:
        _run_multitask(
            sft_checkpoint=sft_checkpoint,
            checkpoint=checkpoint,
            data_dir=data_dir,
            num_demos=num_demos,
            mode=mode,
            simulator=simulator,
            suite=suite,
            trajs_per_task=trajs_per_task,
            num_rollout_envs=num_rollout_envs,
            fm_batch_size=fm_batch_size,
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
            seed=seed,
            world_model=world_model,
            subsample_every=subsample_every,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            use_wandb=use_wandb,
            device=device,
        )
        return

    # ── Single-task path (unchanged) ─────────────────────────────────────
    pt_path = _discover_data(data_path)
    dataset = FewDemoDataset(pt_path, num_demos=num_demos, seed=seed)

    if simulator == "libero":
        resolved_action_dim = ACTION_DIM  # 7
    else:
        resolved_action_dim = dataset.action_dim

    policy = SmolVLAPolicy(
        checkpoint=checkpoint,
        action_dim=resolved_action_dim,
        state_dim=dataset.state_dim,
        device=str(device),
    )
    env_meta: dict = {}
    if sft_checkpoint is not None:
        env_meta = policy.load_checkpoint(sft_checkpoint)
        logging.info("Loaded SFT checkpoint from %s (env_metadata=%s)", sft_checkpoint, env_meta)
    else:
        logging.info("No SFT checkpoint provided - using pretrained %s weights directly", checkpoint)

    resolved_env_id = env_id or env_meta.get("env_id") or dataset.metadata.get("env_id", "PickCube-v1")
    resolved_instruction = (
        instruction
        or env_meta.get("instruction")
        or dataset.metadata.get("instruction", "complete the manipulation task")
    )

    logging.info(
        "RL training: mode=%s  simulator=%s  env_id=%s  instruction=%r  max_steps=%d  num_rollout_envs=%d",
        mode,
        simulator,
        resolved_env_id,
        resolved_instruction,
        resolved_max_steps,
        num_rollout_envs,
    )

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
        save_dir=str(CHECKPOINTS_DIR / mode / f"{task_tag}_seed{seed}_{run_id()}"),
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
        fm_batch_size=fm_batch_size,
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
                "fm_batch_size": fm_batch_size,
            },
        )

    train_srpo(policy, config, resolved_instruction, demo_trajectories=demo_trajectories, wandb_run=run)

    if run is not None:
        run.finish()


# ---------------------------------------------------------------------------
# Multi-task entry point
# ---------------------------------------------------------------------------


def _run_multitask(
    *,
    sft_checkpoint: Path | None,
    checkpoint: str,
    data_dir: Path | None,
    num_demos: int,
    mode: str,
    simulator: str,
    suite: str,
    trajs_per_task: int,
    num_rollout_envs: int,
    fm_batch_size: int,
    lr: float,
    num_iterations: int,
    trajectories_per_iter: int,
    ppo_epochs: int,
    clip_epsilon: float,
    kl_coeff: float,
    gamma: float,
    reward_scale: float,
    eval_every: int,
    eval_episodes: int,
    max_steps: int,
    seed: int,
    world_model: str,
    subsample_every: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    use_wandb: bool,
    device: str,
) -> None:
    """Build TaskSpecs from a directory of .pt files and launch multi-task training."""
    from vla.constants import ACTION_DIM

    if data_dir is None:
        data_dir = PREPROCESSED_DIR
    pt_files = sorted(Path(data_dir).glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    datasets: list[FewDemoDataset] = []
    task_specs: list[TaskSpec] = []
    for idx, pt_path in enumerate(pt_files):
        ds = FewDemoDataset(pt_path, num_demos=num_demos, seed=seed)
        datasets.append(ds)
        task_id_str = pt_path.stem
        task_instruction = ds.metadata.get("instruction", "complete the manipulation task")
        task_specs.append(
            TaskSpec(
                task_id=task_id_str,
                instruction=task_instruction,
                env_id=ds.metadata.get("env_id", ""),
                libero_task_idx=ds.metadata.get("libero_task_id", idx),
                data_path=str(pt_path),
            )
        )

    logging.info("Multi-task SRPO: %d tasks discovered from %s", len(task_specs), data_dir)
    for spec in task_specs:
        logging.info("  [%s] instruction=%r  libero_idx=%d", spec.task_id, spec.instruction, spec.libero_task_idx)

    if simulator == "libero":
        resolved_action_dim = ACTION_DIM
    else:
        resolved_action_dim = datasets[0].action_dim

    policy = SmolVLAPolicy(
        checkpoint=checkpoint,
        action_dim=resolved_action_dim,
        state_dim=datasets[0].state_dim,
        device=str(device),
    )
    if sft_checkpoint is not None:
        env_meta = policy.load_checkpoint(sft_checkpoint)
        logging.info("Loaded SFT checkpoint from %s (env_metadata=%s)", sft_checkpoint, env_meta)
    else:
        logging.info("No SFT checkpoint – using pretrained %s weights directly", checkpoint)

    demo_trajectories: dict[str, list[Trajectory]] | None = None
    if mode == "srpo":
        demo_trajectories = {}
        for spec, ds in zip(task_specs, datasets, strict=True):
            trajs = ds.episodes_as_trajectories()
            for t in trajs:
                t.task_id = spec.task_id
            demo_trajectories[spec.task_id] = trajs
            logging.info("  [%s] %d demo trajectories for reference seeding", spec.task_id, len(trajs))

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
        max_steps=max_steps,
        save_dir=str(CHECKPOINTS_DIR / mode / f"multitask_{suite}_seed{seed}"),
        seed=seed,
        mode=mode,
        world_model_type=world_model,
        subsample_every=subsample_every,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        simulator=simulator,
        suite=suite,
        state_dim=datasets[0].state_dim,
        num_rollout_envs=num_rollout_envs,
        fm_batch_size=fm_batch_size,
    )

    run = None
    if use_wandb:
        task_names = [s.task_id for s in task_specs]
        run = wandb.init(
            project="srpo-smolvla",
            name=f"{mode}_multitask_{suite}_seed{seed}",
            config={
                "method": mode,
                "multitask": True,
                "tasks": task_names,
                "num_tasks": len(task_specs),
                "suite": suite,
                "seed": seed,
                "lr": lr,
                "num_iterations": num_iterations,
                "trajs_per_task_per_iter": trajs_per_task,
                "ppo_epochs": ppo_epochs,
                "clip_epsilon": clip_epsilon,
                "kl_coeff": kl_coeff,
                "gamma": gamma,
                "reward_scale": reward_scale,
                "checkpoint": checkpoint,
                "sft_checkpoint": str(sft_checkpoint),
                "world_model": world_model,
                "num_demos": num_demos,
                "simulator": simulator,
                "num_rollout_envs": num_rollout_envs,
                "fm_batch_size": fm_batch_size,
            },
        )

    train_srpo_multitask(
        policy,
        config,
        task_specs,
        demo_trajectories=demo_trajectories,
        wandb_run=run,
        trajs_per_task_per_iter=trajs_per_task,
    )

    if run is not None:
        run.finish()


if __name__ == "__main__":
    typer.run(main)
