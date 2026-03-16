"""SRPO / sparse-RL training on top of an SFT-initialised SmolVLA policy.

Supports both ManiSkill and LIBERO simulators with vectorised rollouts.

Usage:
    # ManiSkill (default, vectorised with 16 parallel envs):
    uv run python scripts/train_srpo.py --sft-checkpoint checkpoints/sft/best --num-rollout-envs 16

    # LIBERO single-task:
    uv run python scripts/train_srpo.py --simulator libero --suite spatial \
        --task-ids 0 --libero-suite spatial --mode srpo --num-rollout-envs 8

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
from pathlib import Path

import typer

import wandb
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
from vla.data.dataset import FewDemoDataset
from vla.models.smolvla import SmolVLAPolicy
from vla.rl.rollout import Trajectory
from vla.rl.trainer import SRPOConfig, TaskSpec, train_srpo
from vla.training.metrics_logger import MetricsLogger
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
    sft_checkpoint: Path = typer.Option(None, "--sft-checkpoint", "-s", path_type=Path),
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    data_path: Path = typer.Option(
        None, "--data", "-d", path_type=Path, help="Preprocessed .pt file (for demo seeding, single-task)"
    ),
    data_dir: Path = typer.Option(
        None, "--data-dir", path_type=Path, help="Directory of .pt files (one per task) for multi-task SRPO"
    ),
    libero_suite: LiberoSuite = typer.Option(
        None, "--libero-suite", help="Load demos from HuggingFace LeRobot dataset instead of .pt (e.g. spatial)"
    ),
    num_demos: int = typer.Option(5, "--num-demos", "-n"),
    mode: Mode = typer.Option("srpo", "--mode", "-m", help="srpo or sparse_rl"),
    simulator: Simulator = typer.Option("maniskill", "--simulator", help="maniskill or libero"),
    suite: LiberoSuite = typer.Option("spatial", "--suite", help="LIBERO suite (spatial, object, goal, long)"),
    task_ids: str = typer.Option("0", "--task-ids", help="Comma-separated task indices (e.g. '0,2,5,7') or 'all'"),
    trajs_per_task: int = typer.Option(4, "--trajs-per-task", help="Trajectories per task per iteration"),
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
    update_method: UpdateMethod = typer.Option("awr", "--update-method", help="Policy update: awr or ppo"),
    advantage_mode: AdvantageMode = typer.Option(
        AdvantageMode, "--advantage-mode", help="Advantage method: zscore or leave-one-out"
    ),
    ppo_epochs: int = typer.Option(4, "--ppo-epochs"),
    clip_epsilon: float = typer.Option(0.2, "--clip-epsilon"),
    clip_epsilon_high: float = typer.Option(0.28, "--clip-epsilon-high", help="Upper clip bound (asymmetric clipping)"),
    num_fm_noise_samples: int = typer.Option(
        4, "--num-fm-noise-samples", help="FPO: noise/time samples per action for variance reduction"
    ),
    awr_epochs: int = typer.Option(2, "--awr-epochs", help="Regression epochs per iteration (AWR)"),
    awr_temperature: float = typer.Option(5.0, "--awr-temperature", help="AWR weight sharpness (beta)"),
    awr_weight_clip: float = typer.Option(20.0, "--awr-weight-clip", help="Max AWR weight"),
    kl_coeff: float = typer.Option(0.01, "--kl-coeff"),
    adv_eps: float = typer.Option(1e-8, "--adv-eps"),
    adv_skip_threshold: float = typer.Option(1e-6, "--adv-skip-threshold"),
    eval_every: int = typer.Option(10, "--eval-every"),
    eval_episodes: int = typer.Option(50, "--eval-episodes"),
    max_steps: int = typer.Option(280, "--max-steps", help="Override max steps (default: from checkpoint metadata)"),
    seed: int = typer.Option(42, "--seed"),
    env_id: str = typer.Option(None, "--env", help="Override env id (default: from checkpoint metadata)"),
    instruction: str = typer.Option(None, "--instruction", help="Override instruction (default: from checkpoint)"),
    gradient_checkpointing: bool = typer.Option(
        False,
        "--gradient-checkpointing/--no-gradient-checkpointing",
        help="Enable gradient checkpointing to reduce VRAM",
    ),
    world_model: WorldModelType = typer.Option("vjepa2", "--world-model", help="dinov2 or vjepa2"),
    subsample_every: int = typer.Option(5, "--subsample-every"),
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
    use_wandb: bool = typer.Option(True, "--wandb/--no-wandb"),
) -> None:
    """Run SRPO or sparse-RL training starting from an SFT checkpoint."""
    seed_everything(seed)
    device = get_device()

    resolved_max_steps = max_steps or 280
    resolved_eval_envs = num_eval_envs if num_eval_envs > 0 else num_rollout_envs
    resolved_task_ids = _parse_task_ids(task_ids)

    task_specs, demo_trajectories, resolved_state_dim, resolved_action_dim = _build_tasks(
        data_path=data_path,
        data_dir=data_dir,
        libero_suite=libero_suite,
        num_demos=num_demos,
        seed=seed,
        simulator=simulator,
        suite=suite,
        task_ids=resolved_task_ids,
        include_demos=mode == "srpo",
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

    if len(task_specs) > 1:
        task_tag = f"{len(task_specs)}tasks_{suite}"
        run_tag = f"{task_tag}_seed{seed}"
    else:
        run_tag = f"{task_specs[0].task_id}_seed{seed}_{run_id()}"
    save_dir = str(CHECKPOINTS_DIR / mode / run_tag)

    config = SRPOConfig(
        lr=lr,
        max_grad_norm=max_grad_norm,
        num_iterations=num_iterations,
        trajectories_per_iter=trajectories_per_iter,
        update_method=update_method,
        ppo_epochs=ppo_epochs,
        clip_epsilon=clip_epsilon,
        clip_epsilon_high=clip_epsilon_high,
        num_fm_noise_samples=num_fm_noise_samples,
        awr_epochs=awr_epochs,
        awr_temperature=awr_temperature,
        awr_weight_clip=awr_weight_clip,
        advantage_mode=advantage_mode,
        kl_coeff=kl_coeff,
        adv_eps=adv_eps,
        adv_skip_threshold=adv_skip_threshold,
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
        simulator=simulator,
        suite=suite,
        task_id=task_specs[0].libero_task_idx,
        state_dim=resolved_state_dim,
        num_rollout_envs=num_rollout_envs,
        num_eval_envs=resolved_eval_envs,
        fm_batch_size=fm_batch_size,
        gradient_checkpointing=gradient_checkpointing,
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
        run = wandb.init(
            project="srpo-smolvla",
            name=f"{mode}_{run_tag}",
            config=wb_config,
        )

    ml = MetricsLogger(
        jsonl_path=Path(config.save_dir) / "metrics.jsonl",
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

    if run is not None:
        run.finish()


if __name__ == "__main__":
    typer.run(main)
