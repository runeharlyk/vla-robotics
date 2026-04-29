"""SRPO / sparse-RL training on top of an SFT-initialised SmolVLA policy.

Supports both ManiSkill and LIBERO simulators with vectorised rollouts.

Usage:
    # ManiSkill (default, vectorised with 16 parallel envs):
    uv run python scripts/train_srpo.py --sft-checkpoint checkpoints/sft/best --rollout.num-envs 16

    # LIBERO single-task:
    uv run python scripts/train_srpo.py --simulator libero --suite spatial \
        --task-ids 0 --libero-suite spatial --mode srpo --rollout.num-envs 8

    # LIBERO 4 specific tasks:
    uv run python scripts/train_srpo.py --simulator libero --suite spatial \
        --task-ids 0,2,5,7 --libero-suite spatial --mode srpo

    # LIBERO all tasks in a suite:
    uv run python scripts/train_srpo.py --simulator libero --suite spatial \
        --task-ids all --libero-suite spatial --mode srpo

    # Sparse RL ablation:
    uv run python scripts/train_srpo.py --sft-checkpoint ... --mode sparse_rl
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import typer

from vla.constants import (
    CHECKPOINTS_DIR,
    PREPROCESSED_DIR,
    AdvantageMode,
    DistanceMetric,
    LiberoSuite,
    Mode,
    Simulator,
    UpdateMethod,
    WorldModelType,
)
from vla.results_registry import (
    get_git_info,
    get_scheduler_info,
    now_iso,
    summarize_metrics_jsonl,
    write_json,
    write_training_registry,
)
from vla.rl.config import (
    AWRConfig,
    AdvantageConfig,
    DynamicSamplingConfig,
    FPOConfig,
    KLConfig,
    PPOConfig,
    ReplayConfig,
    RolloutConfig,
    SRPOConfig,
    SuccessBCConfig,
    TaskSpec,
)
from vla.rl.demo_replay import replay_demo_rollouts
from vla.rl.rollout import Trajectory
from vla.utils import get_device, run_id, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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
        logger.warning("Multiple .pt files: %s. Using %s.", names, pts[0].name)
    return pts[0]


def _parse_task_ids(raw: str) -> list[int] | None:
    """Parse the ``--task-ids`` CLI value.

    Returns ``None`` when *all* tasks should be used, or a concrete list of
    integer indices otherwise.
    """
    stripped = raw.strip().lower()
    if stripped == "all":
        return None
    return [int(x) for x in stripped.split(",")]


def _build_tasks(
    *,
    data_path: Path | None,
    data_dir: Path | None,
    libero_suite: LiberoSuite | None,
    num_demos: int,
    seed: int,
    simulator: Simulator,
    suite: LiberoSuite,
    task_ids: list[int] | None,
    include_demos: bool,
    env_id_override: str | None,
    instruction_override: str | None,
) -> tuple[list[TaskSpec], dict[str, list[Trajectory]] | None, int, int]:
    """Build the task list and optional demo trajectories.

    Args:
        task_ids: Concrete indices to use, or ``None`` to discover all tasks
                  in the suite / data directory.

    Always returns a list of :class:`TaskSpec` (length >= 1) plus the
    state / action dimensions needed to construct the policy.
    """
    from vla.constants import ACTION_DIM

    demo_trajectories: dict[str, list[Trajectory]] | None = {} if include_demos else None

    if simulator is Simulator.LIBERO and libero_suite is not None and data_dir is None:
        from vla.data.libero import LiberoSFTDataset

        if task_ids is None:
            catalog = LiberoSFTDataset(suite=libero_suite, num_demos=1, seed=seed)
            task_map = {int(idx): str(instr) for idx, instr in sorted(getattr(catalog, "_task_map", {}).items())}
            if not task_map:
                raise ValueError(f"No LIBERO tasks found for suite {libero_suite!r}")
            state_dim = catalog.state_dim
        else:
            first_ds = LiberoSFTDataset(suite=libero_suite, num_demos=num_demos, seed=seed, task_id=task_ids[0])
            task_map = {}
            for tidx in task_ids:
                ds = (
                    first_ds
                    if tidx == task_ids[0]
                    else LiberoSFTDataset(suite=libero_suite, num_demos=num_demos, seed=seed, task_id=tidx)
                )
                task_map[tidx] = ds.instruction
            state_dim = first_ds.state_dim

        task_specs: list[TaskSpec] = []
        for tidx, task_instruction in task_map.items():
            if task_ids is not None:
                # LiberoSFTDataset.instruction is a suite-level default and
                # may point at the first task; use the per-task lookup here.
                task_instruction = (
                    first_ds._task_map.get(tidx, task_instruction)
                    if hasattr(first_ds, "_task_map")
                    else task_instruction
                )
            task_key = f"{suite}_task_{tidx}"
            task_specs.append(
                TaskSpec(
                    task_id=task_key,
                    instruction=instruction_override or task_instruction or "complete the manipulation task",
                    env_id=env_id_override or f"libero_{libero_suite}",
                    libero_task_idx=tidx,
                )
            )
            if demo_trajectories is not None:
                task_dataset = LiberoSFTDataset(
                    suite=libero_suite,
                    num_demos=num_demos,
                    seed=seed,
                    task_id=tidx,
                )
                trajs = task_dataset.episodes_as_trajectories(task_id=tidx)
                for traj in trajs:
                    traj.task_id = task_key
                demo_trajectories[task_key] = trajs

        return task_specs, demo_trajectories, state_dim, ACTION_DIM

    from vla.data.dataset import FewDemoDataset

    if task_ids is None:
        search_dir = data_dir or PREPROCESSED_DIR
        pt_files = sorted(Path(search_dir).glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {search_dir}")
    else:
        pt_files = [_discover_data(data_path)]

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
                instruction=instruction_override or task_instruction,
                env_id=env_id_override or ds.metadata.get("env_id", ""),
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
    sft_checkpoint: Path | None = typer.Option(
        None,
        "--sft-checkpoint",
        "-s",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    data_path: Path | None = typer.Option(
        None,
        "--data",
        "-d",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Preprocessed .pt file (for demo seeding, single-task)",
    ),
    data_dir: Path | None = typer.Option(
        None,
        "--data-dir",
        exists=True,
        dir_okay=True,
        readable=True,
        help="Directory of .pt files (one per task) for multi-task SRPO",
    ),
    libero_suite: LiberoSuite | None = typer.Option(
        None, "--libero-suite", help="Load demos from HuggingFace LeRobot dataset instead of .pt (e.g. spatial)"
    ),
    num_demos: int = typer.Option(5, "--num-demos", "-n"),
    mode: Mode = typer.Option("srpo", "--mode", "-m", help="srpo or sparse_rl"),
    simulator: Simulator = typer.Option("maniskill", "--simulator", help="maniskill or libero"),
    suite: LiberoSuite = typer.Option("spatial", "--suite", help="LIBERO suite (spatial, object, goal, long)"),
    task_ids: str = typer.Option("0", "--task-ids", help="Comma-separated task indices (e.g. '0,2,5,7') or 'all'"),
    trajs_per_task: int = typer.Option(4, "--trajs-per-task", help="Trajectories per task per iteration"),
    num_rollout_envs: int = typer.Option(
        1, "--rollout.num-envs", help="Parallel envs per task for vectorised rollouts"
    ),
    num_envs: int = typer.Option(
        0, "--rollout.eval-num-envs", help="Parallel envs for vectorised eval (0 = same as rollout.num-envs)"
    ),
    fm_batch_size: int = typer.Option(32, "--rollout.fm-batch-size", help="Timesteps per FM forward pass"),
    lr: float = typer.Option(1e-5, "--lr"),
    max_grad_norm: float = typer.Option(10.0, "--max-grad-norm", help="Max gradient norm for clipping"),
    num_iterations: int = typer.Option(100, "--iterations"),
    update_method: UpdateMethod = typer.Option(
        "awr",
        "--update-method",
        help="Policy update: awr, fpo, ppo, or success_bc",
    ),
    advantage_mode: AdvantageMode = typer.Option(
        AdvantageMode.LEAVE_ONE_OUT, "--adv.mode", help="Advantage method: zscore or leave-one-out"
    ),
    ppo_epochs: int = typer.Option(4, "--ppo.epochs"),
    ppo_minibatch_trajs: int = typer.Option(4, "--ppo.minibatch-trajs"),
    clip_epsilon: float = typer.Option(0.2, "--ppo.clip-epsilon"),
    clip_epsilon_high: float = typer.Option(
        0.28, "--ppo.clip-epsilon-high", help="Upper clip bound (asymmetric clipping)"
    ),
    num_fm_noise_samples: int = typer.Option(
        4, "--fpo.num-fm-noise-samples", help="FPO: noise/time samples per action for variance reduction"
    ),
    awr_epochs: int = typer.Option(2, "--awr.epochs", help="Regression epochs per iteration (AWR)"),
    awr_temperature: float = typer.Option(5.0, "--awr.temperature", help="AWR weight sharpness (beta)"),
    awr_weight_clip: float = typer.Option(20.0, "--awr.weight-clip", help="Max AWR weight"),
    success_bc_epochs: int = typer.Option(
        1,
        "--success-bc.epochs",
        help="BC/SFT epochs per iteration for --update-method success_bc.",
    ),
    success_bc_loss_reduction: str = typer.Option(
        "mean",
        "--success-bc.loss-reduction",
        help="Chunk-position reduction for success_bc: mean or sum.",
    ),
    kl_coeff: float = typer.Option(0.01, "--kl.coeff"),
    sft_kl_coeff: float = typer.Option(
        0.0,
        "--kl.sft-coeff",
        help="Additional KL penalty against the immutable initial policy checkpoint to limit cumulative drift.",
    ),
    adv_eps: float = typer.Option(1e-8, "--adv.eps"),
    adv_skip_threshold: float = typer.Option(1e-6, "--adv.skip-threshold"),
    eval_every: int = typer.Option(10, "--eval-every"),
    eval_episodes: int = typer.Option(50, "--eval-episodes"),
    max_steps: int = typer.Option(280, "--max-steps", help="Override max steps (default: from checkpoint metadata)"),
    seed: int = typer.Option(42, "--seed"),
    env_id: str = typer.Option(None, "--env", help="Override env id (default: from checkpoint metadata)"),
    instruction: str = typer.Option(None, "--instruction", help="Override instruction (default: from checkpoint)"),
    gradient_checkpointing: bool = typer.Option(
        False,
        "--rollout.gradient-checkpointing/--no-rollout.gradient-checkpointing",
        help="Enable gradient checkpointing to reduce VRAM",
    ),
    world_model: WorldModelType = typer.Option("vjepa2", "--world-model", help="dinov2 or vjepa2"),
    subsample_every: int = typer.Option(1, "--subsample-every"),
    dbscan_eps: float = typer.Option(0.5, "--dbscan-eps"),
    dbscan_min_samples: int = typer.Option(2, "--dbscan-min-samples"),
    distance_metric: DistanceMetric = typer.Option(
        "normalized_l2", "--distance-metric", help="normalized_l2 or cosine or l2"
    ),
    dbscan_auto_eps: bool = typer.Option(False, "--dbscan-auto-eps", help="Auto-tune DBSCAN eps"),
    use_failure_rewards: bool = typer.Option(
        True,
        "--failure-rewards/--no-failure-rewards",
        help="Use distance-based failure rewards (SRPO). Disable for sparse-only rewards.",
    ),
    use_standard_scaler: bool = typer.Option(
        False,
        "--standard-scaler/--no-standard-scaler",
        help="Apply StandardScaler before DBSCAN (matches siiRL production code).",
    ),
    use_wandb: bool = typer.Option(True, "--wandb/--no-wandb"),
    wandb_name: str = typer.Option(None, "--wandb-name", help="Optional prefix for the wandb run name"),
    fpo_full_chunk_target: bool = typer.Option(
        True,
        "--fpo.full-chunk-target/--no-fpo.full-chunk-target",
        help=(
            "For chunked rollouts, train on v28-style sliding windows reconstructed from executed actions. "
            "Disable to train only on directly executed positions inside each sampled chunk."
        ),
    ),
    fpo_loss_reduction: str = typer.Option("sum", "--fpo.loss-reduction"),
    fpo_positive_adv_only: bool = typer.Option(False, "--fpo.positive-adv-only/--no-fpo.positive-adv-only"),
    fpo_negative_adv_scale: float = typer.Option(0.25, "--fpo.negative-adv-scale"),
    fpo_log_ratio_clip: float = typer.Option(5.0, "--fpo.log-ratio-clip"),
    fpo_use_ref_policy_kl: bool = typer.Option(
        False,
        "--fpo.use-ref-policy-kl/--no-fpo.use-ref-policy-kl",
        help="Anchor KL penalty to a reference policy instead of cached old_fm losses.",
    ),
    eval_zero_sample: bool = typer.Option(True, "--rollout.eval-zero-sample/--no-rollout.eval-zero-sample"),
    adaptive_kl: bool = typer.Option(
        False,
        "--kl.adaptive/--no-kl.adaptive",
        help="Adaptively adjust kl_coeff each iteration to track kl_target",
    ),
    kl_target: float = typer.Option(0.01, "--kl.target", help="Target KL for adaptive adjustment"),
    kl_adapt_factor: float = typer.Option(1.5, "--kl.adapt-factor", help="Multiplicative factor for adaptive KL"),
    include_demos_in_update: bool = typer.Option(
        False,
        "--replay.include-demos/--no-replay.include-demos",
        help="Include demonstration trajectories in every policy update iteration (online SFT).",
    ),
    success_replay_buffer_size: int = typer.Option(
        0, "--replay.success-buffer-size", help="Replay successful trajectories from previous iterations."
    ),
    success_replay_total_size: int = typer.Option(
        0,
        "--replay.success-total-size",
        help="Global capacity for balanced success replay across tasks. Overrides --replay.success-buffer-size.",
    ),
    success_replay_alpha: float = typer.Option(
        1.0,
        "--replay.success-alpha",
        help="Inverse-success weighting strength for balanced replay (0 disables balancing, 1 is linear inverse).",
    ),
    success_replay_ema_decay: float = typer.Option(
        0.8,
        "--replay.success-ema-decay",
        help="EMA decay for per-task success-rate estimates used by balanced replay.",
    ),
    success_replay_max_ratio: float = typer.Option(
        1.0,
        "--replay.success-max-ratio",
        help="Maximum replayed trajectories per iteration as a multiple of fresh rollout trajectories.",
    ),
    dynamic_sampling: bool = typer.Option(
        False,
        "--sampling.dynamic/--no-sampling.dynamic",
        help="DAPO-style replacement sampling: re-collect rollouts for tasks whose reward std is below --adv.skip-threshold.",
    ),
    dynamic_sampling_max_retries: int = typer.Option(
        2,
        "--sampling.dynamic-max-retries",
        help="Max number of replacement draws per uniform-reward task before giving up.",
    ),
    n_action_steps: int = typer.Option(
        1,
        "--rollout.n-action-steps",
        help="Number of actions to execute from each sampled policy chunk before re-planning. "
        "1 (default) preserves the legacy single-step rollout behaviour; H>1 reduces rollout "
        "compute by ~H×. By default policy updates reconstruct v28-style sliding-window targets "
        "from the executed action stream.",
    ),
) -> None:
    import wandb
    from vla.models.smolvla import SmolVLAPolicy
    from vla.rl.trainer import train_srpo
    from vla.training.metrics_logger import MetricsLogger

    """Run SRPO or sparse-RL training starting from an SFT checkpoint."""
    seed_everything(seed)
    device = get_device()

    resolved_max_steps = max_steps or 280
    resolved_eval_envs = num_envs if num_envs > 0 else num_rollout_envs
    resolved_task_ids = _parse_task_ids(task_ids)

    include_demos_internal = (mode == Mode.SRPO) or include_demos_in_update
    demo_seeding = include_demos_internal

    task_specs, demo_trajectories, resolved_state_dim, resolved_action_dim = _build_tasks(
        data_path=data_path,
        data_dir=data_dir,
        libero_suite=libero_suite,
        num_demos=num_demos,
        seed=seed,
        simulator=simulator,
        suite=suite,
        task_ids=resolved_task_ids,
        include_demos=include_demos_internal,
        env_id_override=env_id,
        instruction_override=instruction,
    )

    policy = SmolVLAPolicy(
        checkpoint=checkpoint,
        action_dim=resolved_action_dim,
        state_dim=resolved_state_dim,
        device=str(device),
    )
    if sft_checkpoint is not None:
        env_meta = policy.load_checkpoint(sft_checkpoint)
        logger.info("Loaded SFT checkpoint from %s (env_metadata=%s)", sft_checkpoint, env_meta)
        for spec in task_specs:
            if not spec.env_id and env_meta.env_id:
                spec.env_id = env_meta.env_id
            if spec.instruction == "complete the manipulation task" and env_meta.instruction:
                spec.instruction = env_meta.instruction
    else:
        logger.info("No SFT checkpoint - using pretrained %s weights directly", checkpoint)

    if demo_trajectories:
        logger.info("Replacing raw demo trajectories with simulator-replayed trajectories before training uses them.")
        demo_trajectories = replay_demo_rollouts(
            task_specs=task_specs,
            demo_trajectories=demo_trajectories,
            simulator=simulator,
            suite=suite,
            max_steps=resolved_max_steps,
            seed=seed,
            state_dim=resolved_state_dim,
        )

    if len(task_specs) > 1:
        task_tag = f"{len(task_specs)}tasks_{suite}"
        run_tag = f"{task_tag}_seed{seed}_{run_id()}"
    else:
        run_tag = f"{task_specs[0].task_id}_seed{seed}_{run_id()}"
    save_dir = str(CHECKPOINTS_DIR / mode / run_tag)

    config = SRPOConfig(
        lr=lr,
        max_grad_norm=max_grad_norm,
        num_iterations=num_iterations,
        update_method=update_method,
        advantage=AdvantageConfig(
            mode=advantage_mode,
            eps=adv_eps,
            skip_threshold=adv_skip_threshold,
        ),
        ppo=PPOConfig(
            epochs=ppo_epochs,
            minibatch_trajs=ppo_minibatch_trajs,
            clip_epsilon=clip_epsilon,
            clip_epsilon_high=clip_epsilon_high,
        ),
        awr=AWRConfig(
            epochs=awr_epochs,
            temperature=awr_temperature,
            weight_clip=awr_weight_clip,
        ),
        fpo=FPOConfig(
            num_fm_noise_samples=num_fm_noise_samples,
            full_chunk_target=fpo_full_chunk_target,
            loss_reduction=fpo_loss_reduction,
            positive_adv_only=fpo_positive_adv_only,
            negative_adv_scale=fpo_negative_adv_scale,
            log_ratio_clip=fpo_log_ratio_clip,
            use_ref_policy_kl=fpo_use_ref_policy_kl,
        ),
        success_bc=SuccessBCConfig(
            epochs=success_bc_epochs,
            loss_reduction=success_bc_loss_reduction,
        ),
        kl=KLConfig(
            coeff=kl_coeff,
            sft_coeff=sft_kl_coeff,
            adaptive=adaptive_kl,
            target=kl_target,
            adapt_factor=kl_adapt_factor,
        ),
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        max_steps=resolved_max_steps,
        save_dir=save_dir,
        env_id=task_specs[0].env_id,
        seed=seed,
        mode=mode,
        world_model_type=world_model,
        subsample_every=subsample_every,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        distance_metric=distance_metric,
        dbscan_auto_eps=dbscan_auto_eps,
        use_failure_rewards=use_failure_rewards,
        use_standard_scaler=use_standard_scaler,
        simulator=simulator,
        suite=suite,
        task_id=task_specs[0].libero_task_idx,
        state_dim=resolved_state_dim,
        replay=ReplayConfig(
            include_demos_in_update=include_demos_in_update,
            success_buffer_size=success_replay_buffer_size,
            success_total_size=success_replay_total_size,
            success_alpha=success_replay_alpha,
            success_ema_decay=success_replay_ema_decay,
            success_max_ratio=success_replay_max_ratio,
        ),
        sampling=DynamicSamplingConfig(
            enabled=dynamic_sampling,
            max_retries=dynamic_sampling_max_retries,
        ),
        rollout=RolloutConfig(
            num_envs=num_rollout_envs,
            eval_num_envs=resolved_eval_envs,
            fm_batch_size=fm_batch_size,
            gradient_checkpointing=gradient_checkpointing,
            eval_zero_sample=eval_zero_sample,
            n_action_steps=n_action_steps,
        ),
    )

    logger.info(
        "RL training: mode=%s  simulator=%s  tasks=%d  max_steps=%d  num_rollout_envs=%d",
        mode,
        simulator,
        len(task_specs),
        resolved_max_steps,
        num_rollout_envs,
    )
    for spec in task_specs:
        n_demos = len(demo_trajectories.get(spec.task_id, [])) if demo_trajectories else 0
        logger.info(
            "  [%s] instruction=%r  libero_idx=%d  demos=%d",
            spec.task_id,
            spec.instruction,
            spec.libero_task_idx,
            n_demos,
        )

    run = None
    final_name: str | None = None
    if use_wandb:
        wb_config = config.to_dict()
        wb_config.update(
            method=mode,
            tasks=[s.task_id for s in task_specs],
            num_tasks=len(task_specs),
            checkpoint=checkpoint,
            sft_checkpoint=str(sft_checkpoint),
            num_demos=num_demos,
            trajs_per_task_per_iter=trajs_per_task,
        )
        final_name = f"{wandb_name}_{mode}_{run_tag}" if wandb_name else f"{mode}_{run_tag}"
        run = wandb.init(
            project="srpo-smolvla",
            name=final_name,
            config=wb_config,
        )

    save_dir_path = Path(config.save_dir)
    metrics_jsonl_path = save_dir_path / "metrics.jsonl"
    training_record = {
        "record_type": "training",
        "recorded_at": now_iso(),
        "completed_at": None,
        "method": str(mode),
        "update_method": str(update_method),
        "save_dir": str(save_dir_path),
        "best_checkpoint_dir": str(save_dir_path / "best"),
        "best_rollout_checkpoint_dir": str(save_dir_path / "best_rollout"),
        "last_checkpoint_dir": str(save_dir_path / "last"),
        "checkpoint": checkpoint,
        "sft_checkpoint": str(sft_checkpoint) if sft_checkpoint is not None else "",
        "simulator": str(simulator),
        "suite": str(suite),
        "seed": seed,
        "num_tasks": len(task_specs),
        "task_ids": [spec.task_id for spec in task_specs],
        "libero_task_indices": [spec.libero_task_idx for spec in task_specs],
        "task_instructions": {spec.task_id: spec.instruction for spec in task_specs},
        "wandb_run_name": final_name or "",
        "demo_seeding": demo_seeding,
        "demo_trajectory_source": "replayed_env_rollouts" if demo_trajectories else "none",
        "include_demos_in_update": include_demos_in_update,
        "success_replay_total_size": success_replay_total_size,
        "trajs_per_task_per_iter": trajs_per_task,
        "config": config.to_dict(),
        "task_specs": [asdict(spec) for spec in task_specs],
        "metrics_jsonl": str(metrics_jsonl_path),
        **get_git_info(),
        **get_scheduler_info(),
    }
    write_json(save_dir_path / "training_run.json", training_record)

    ml = MetricsLogger(
        jsonl_path=metrics_jsonl_path,
        wandb_run=run,
    )

    train_srpo(
        policy,
        config,
        task_specs,
        demo_trajectories=demo_trajectories,
        metrics_logger=ml,
        trajs_per_task_per_iter=trajs_per_task,
    )

    training_record["completed_at"] = now_iso()
    training_record.update(
        summarize_metrics_jsonl(
            metrics_jsonl_path,
            eval_key_suffixes=[f"{mode}/eval/success_rate"],
        )
    )
    write_json(save_dir_path / "training_run.json", training_record)
    write_training_registry(training_record)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    typer.run(main)
