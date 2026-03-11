"""SRPO / sparse-RL training on top of an SFT-initialised SmolVLA policy.

Supports both ManiSkill and LIBERO simulators with vectorised rollouts.

Usage:
    # ManiSkill (default, vectorised with 16 parallel envs):
    uv run python scripts/train_srpo.py --sft-checkpoint checkpoints/sft/best --num-rollout-envs 16

    # LIBERO (subprocess-vectorised):
    uv run python scripts/train_srpo.py --sft-checkpoint checkpoints/sft/best \
        --simulator libero --suite spatial --num-rollout-envs 8

    # Sparse RL ablation:
    uv run python scripts/train_srpo.py --sft-checkpoint ... --mode sparse_rl

    # SRPO with demo seeding:
    uv run python scripts/train_srpo.py --sft-checkpoint ... --data data/preprocessed/peginsertionside.pt --mode srpo

    # SRPO on LIBERO loading demos directly from HuggingFace (no .pt needed):
    uv run python scripts/train_srpo.py --simulator libero --suite spatial \
        --task-id 0 --libero-suite spatial --mode srpo

    # Multi-task SRPO across an entire LIBERO suite:
    uv run python scripts/train_srpo.py --sft-checkpoint ... --simulator libero \
        --suite spatial --multitask --libero-suite spatial --mode srpo
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

import wandb
from vla.constants import CHECKPOINTS_DIR, PREPROCESSED_DIR
from vla.data.dataset import FewDemoDataset
from vla.models.smolvla import SmolVLAPolicy
from vla.rl.rollout import Trajectory
from vla.rl.trainer import SRPOConfig, TaskSpec, train_srpo
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


def _load_multitask_data(
    *,
    data_dir: Path | None,
    libero_suite: str | None,
    num_demos: int,
    seed: int,
    simulator: str,
    suite: str,
    include_demos: bool,
) -> tuple[list[TaskSpec], dict[str, list[Trajectory]] | None, int, int]:
    from vla.constants import ACTION_DIM

    demo_trajectories = {} if include_demos else None

    if simulator == "libero" and libero_suite is not None and data_dir is None:
        from vla.data.libero import LiberoSFTDataset

        catalog = LiberoSFTDataset(suite=libero_suite, num_demos=1, seed=seed)
        task_map = {
            int(task_idx): str(instruction)
            for task_idx, instruction in sorted(getattr(catalog, "_task_map", {}).items())
        }
        if not task_map:
            raise ValueError(f"No LIBERO tasks found for suite {libero_suite!r}")

        task_specs: list[TaskSpec] = []
        for task_idx, task_instruction in task_map.items():
            task_key = f"{suite}_task_{task_idx}"
            task_specs.append(
                TaskSpec(
                    task_id=task_key,
                    instruction=task_instruction or "complete the manipulation task",
                    env_id=f"libero_{libero_suite}",
                    libero_task_idx=task_idx,
                )
            )
            if demo_trajectories is not None:
                task_dataset = LiberoSFTDataset(
                    suite=libero_suite,
                    num_demos=num_demos,
                    seed=seed,
                    task_id=task_idx,
                )
                trajs = task_dataset.episodes_as_trajectories(task_id=task_idx)
                for traj in trajs:
                    traj.task_id = task_key
                demo_trajectories[task_key] = trajs

        return task_specs, demo_trajectories, catalog.state_dim, ACTION_DIM

    if data_dir is None:
        data_dir = PREPROCESSED_DIR
    pt_files = sorted(Path(data_dir).glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {data_dir}")

    datasets: list[FewDemoDataset] = []
    task_specs = []
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
        if demo_trajectories is not None:
            trajs = ds.episodes_as_trajectories()
            for traj in trajs:
                traj.task_id = task_id_str
            demo_trajectories[task_id_str] = trajs

    resolved_action_dim = ACTION_DIM if simulator == "libero" else datasets[0].action_dim
    return task_specs, demo_trajectories, datasets[0].state_dim, resolved_action_dim


def main(
    sft_checkpoint: Path = typer.Option(None, "--sft-checkpoint", "-s", path_type=Path),
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    data_path: Path = typer.Option(
        None, "--data", "-d", path_type=Path, help="Preprocessed .pt file (for demo seeding, single-task)"
    ),
    data_dir: Path = typer.Option(
        None, "--data-dir", path_type=Path, help="Directory of .pt files (one per task) for multi-task SRPO"
    ),
    libero_suite: str = typer.Option(
        None, "--libero-suite", help="Load demos from HuggingFace LeRobot dataset instead of .pt (e.g. spatial)"
    ),
    num_demos: int = typer.Option(5, "--num-demos", "-n"),
    mode: str = typer.Option("srpo", "--mode", "-m", help="srpo or sparse_rl"),
    simulator: str = typer.Option("maniskill", "--simulator", help="maniskill or libero"),
    suite: str = typer.Option("spatial", "--suite", help="LIBERO suite (spatial, object, goal, long)"),
    task_id: int = typer.Option(0, "--task-id", help="LIBERO task index within the suite"),
    multitask: bool = typer.Option(False, "--multitask/--single-task", help="Enable multi-task SRPO training"),
    trajs_per_task: int = typer.Option(4, "--trajs-per-task", help="Trajectories per task per iteration (multi-task)"),
    num_rollout_envs: int = typer.Option(
        1, "--num-rollout-envs", help="Parallel envs per task for vectorised rollouts"
    ),
    num_eval_envs: int = typer.Option(
        0, "--num-eval-envs", help="Parallel envs for vectorised eval (0 = same as num-rollout-envs)"
    ),
    fm_batch_size: int = typer.Option(32, "--fm-batch-size", help="Timesteps per FM forward pass"),
    lr: float = typer.Option(1e-5, "--lr"),
    max_grad_norm: float = typer.Option(10.0, "--max-grad-norm", help="Max gradient norm for clipping"),
    num_iterations: int = typer.Option(100, "--iterations"),
    trajectories_per_iter: int = typer.Option(16, "--trajs-per-iter"),
    update_method: str = typer.Option("awr", "--update-method", help="Policy update: awr or ppo"),
    ppo_epochs: int = typer.Option(4, "--ppo-epochs"),
    clip_epsilon: float = typer.Option(0.2, "--clip-epsilon"),
    awr_epochs: int = typer.Option(2, "--awr-epochs", help="Regression epochs per iteration (AWR)"),
    awr_temperature: float = typer.Option(5.0, "--awr-temperature", help="AWR weight sharpness (beta)"),
    awr_weight_clip: float = typer.Option(20.0, "--awr-weight-clip", help="Max AWR weight"),
    kl_coeff: float = typer.Option(0.01, "--kl-coeff"),
    eval_every: int = typer.Option(10, "--eval-every"),
    eval_episodes: int = typer.Option(50, "--eval-episodes"),
    max_steps: int = typer.Option(None, "--max-steps", help="Override max steps (default: from checkpoint metadata)"),
    seed: int = typer.Option(42, "--seed"),
    env_id: str = typer.Option(None, "--env", help="Override env id (default: from checkpoint metadata)"),
    instruction: str = typer.Option(None, "--instruction", help="Override instruction (default: from checkpoint)"),
    gradient_checkpointing: bool = typer.Option(
        False,
        "--gradient-checkpointing/--no-gradient-checkpointing",
        help="Enable gradient checkpointing to reduce VRAM",
    ),
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

    Use ``--multitask`` with either ``--data-dir`` or ``--libero-suite`` to run
    multi-task SRPO with per-task reward models.
    """
    seed_everything(seed)
    device = get_device()

    from vla.constants import ACTION_DIM

    resolved_max_steps = max_steps or 280
    resolved_eval_envs = num_eval_envs if num_eval_envs > 0 else num_rollout_envs

    if multitask:
        _run_multitask(
            sft_checkpoint=sft_checkpoint,
            checkpoint=checkpoint,
            data_dir=data_dir,
            libero_suite=libero_suite,
            num_demos=num_demos,
            mode=mode,
            simulator=simulator,
            suite=suite,
            trajs_per_task=trajs_per_task,
            num_rollout_envs=num_rollout_envs,
            num_eval_envs=resolved_eval_envs,
            fm_batch_size=fm_batch_size,
            gradient_checkpointing=gradient_checkpointing,
            lr=lr,
            max_grad_norm=max_grad_norm,
            num_iterations=num_iterations,
            trajectories_per_iter=trajectories_per_iter,
            update_method=update_method,
            ppo_epochs=ppo_epochs,
            clip_epsilon=clip_epsilon,
            awr_epochs=awr_epochs,
            awr_temperature=awr_temperature,
            awr_weight_clip=awr_weight_clip,
            kl_coeff=kl_coeff,
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

    # ── Single-task path ────────────────────────────────────────────────
    if libero_suite is not None:
        from vla.data.libero import LiberoSFTDataset

        dataset = LiberoSFTDataset(suite=libero_suite, num_demos=num_demos, seed=seed, task_id=task_id)
        resolved_action_dim = ACTION_DIM
    else:
        pt_path = _discover_data(data_path)
        dataset = FewDemoDataset(pt_path, num_demos=num_demos, seed=seed)
        resolved_action_dim = ACTION_DIM if simulator == "libero" else dataset.action_dim

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

    task_tag = resolved_env_id.lower().replace("-", "_")
    task_spec = TaskSpec(
        task_id=task_tag,
        instruction=resolved_instruction,
        env_id=resolved_env_id,
        libero_task_idx=task_id if simulator == "libero" else 0,
    )

    demo_dict: dict[str, list[Trajectory]] | None = None
    if mode == "srpo":
        if libero_suite is not None:
            raw_demos = dataset.episodes_as_trajectories(task_id=task_id)
        else:
            raw_demos = dataset.episodes_as_trajectories()
        for t in raw_demos:
            t.task_id = task_tag
        demo_dict = {task_tag: raw_demos}
        logging.info("Built %d demo trajectories for reference seeding", len(raw_demos))

    config = SRPOConfig(
        lr=lr,
        max_grad_norm=max_grad_norm,
        num_iterations=num_iterations,
        trajectories_per_iter=trajectories_per_iter,
        update_method=update_method,
        ppo_epochs=ppo_epochs,
        clip_epsilon=clip_epsilon,
        awr_epochs=awr_epochs,
        awr_temperature=awr_temperature,
        awr_weight_clip=awr_weight_clip,
        kl_coeff=kl_coeff,
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
        num_eval_envs=resolved_eval_envs,
        fm_batch_size=fm_batch_size,
        gradient_checkpointing=gradient_checkpointing,
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
                "max_grad_norm": max_grad_norm,
                "num_iterations": num_iterations,
                "trajectories_per_iter": trajectories_per_iter,
                "update_method": update_method,
                "ppo_epochs": ppo_epochs,
                "clip_epsilon": clip_epsilon,
                "awr_epochs": awr_epochs,
                "awr_temperature": awr_temperature,
                "awr_weight_clip": awr_weight_clip,
                "kl_coeff": kl_coeff,
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

    train_srpo(
        policy,
        config,
        [task_spec],
        demo_trajectories=demo_dict,
        wandb_run=run,
        trajs_per_task_per_iter=trajectories_per_iter,
    )

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
    libero_suite: str | None,
    num_demos: int,
    mode: str,
    simulator: str,
    suite: str,
    trajs_per_task: int,
    num_rollout_envs: int,
    num_eval_envs: int,
    fm_batch_size: int,
    gradient_checkpointing: bool,
    lr: float,
    max_grad_norm: float,
    num_iterations: int,
    trajectories_per_iter: int,
    update_method: str,
    ppo_epochs: int,
    clip_epsilon: float,
    awr_epochs: int,
    awr_temperature: float,
    awr_weight_clip: float,
    kl_coeff: float,
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
    """Build TaskSpecs and launch multi-task training."""
    task_specs, demo_trajectories, resolved_state_dim, resolved_action_dim = _load_multitask_data(
        data_dir=data_dir,
        libero_suite=libero_suite,
        num_demos=num_demos,
        seed=seed,
        simulator=simulator,
        suite=suite,
        include_demos=mode == "srpo",
    )

    use_libero_suite = simulator == "libero" and libero_suite is not None and data_dir is None
    source_desc = libero_suite if use_libero_suite else str(data_dir or PREPROCESSED_DIR)
    logging.info("Multi-task SRPO: %d tasks discovered from %s", len(task_specs), source_desc)
    for spec in task_specs:
        logging.info("  [%s] instruction=%r  libero_idx=%d", spec.task_id, spec.instruction, spec.libero_task_idx)

    policy = SmolVLAPolicy(
        checkpoint=checkpoint,
        action_dim=resolved_action_dim,
        state_dim=resolved_state_dim,
        device=str(device),
    )
    if sft_checkpoint is not None:
        env_meta = policy.load_checkpoint(sft_checkpoint)
        logging.info("Loaded SFT checkpoint from %s (env_metadata=%s)", sft_checkpoint, env_meta)
    else:
        logging.info("No SFT checkpoint – using pretrained %s weights directly", checkpoint)

    if demo_trajectories is not None:
        for spec in task_specs:
            logging.info(
                "  [%s] %d demo trajectories for reference seeding",
                spec.task_id,
                len(demo_trajectories.get(spec.task_id, [])),
            )

    config = SRPOConfig(
        lr=lr,
        max_grad_norm=max_grad_norm,
        num_iterations=num_iterations,
        trajectories_per_iter=trajectories_per_iter,
        update_method=update_method,
        ppo_epochs=ppo_epochs,
        clip_epsilon=clip_epsilon,
        awr_epochs=awr_epochs,
        awr_temperature=awr_temperature,
        awr_weight_clip=awr_weight_clip,
        kl_coeff=kl_coeff,
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
        state_dim=resolved_state_dim,
        num_rollout_envs=num_rollout_envs,
        num_eval_envs=num_eval_envs,
        fm_batch_size=fm_batch_size,
        gradient_checkpointing=gradient_checkpointing,
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
                "max_grad_norm": max_grad_norm,
                "num_iterations": num_iterations,
                "trajs_per_task_per_iter": trajs_per_task,
                "update_method": update_method,
                "ppo_epochs": ppo_epochs,
                "clip_epsilon": clip_epsilon,
                "awr_epochs": awr_epochs,
                "awr_temperature": awr_temperature,
                "awr_weight_clip": awr_weight_clip,
                "kl_coeff": kl_coeff,
                "checkpoint": checkpoint,
                "sft_checkpoint": str(sft_checkpoint),
                "world_model": world_model,
                "num_demos": num_demos,
                "simulator": simulator,
                "num_rollout_envs": num_rollout_envs,
                "fm_batch_size": fm_batch_size,
            },
        )

    train_srpo(
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
