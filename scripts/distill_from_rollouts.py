"""Self-distill SmolVLA from collected RL rollouts.

Wraps the standard SFT loop (``vla.training.sft_smolvla.train_sft``) with
distillation-friendly defaults so the typical "RL teacher -> SFT-init
student" workflow is one command:

    uv run python scripts/distill_from_rollouts.py \\
        --rollouts data/collected/libero_spatial_distill.pt \\
        --student-checkpoint HuggingFaceVLA/smolvla_libero \\
        --num-epochs 10

What it does:
    1. Loads the collected ``.pt`` rollouts via
       :func:`vla.data.rollout_distill.build_rollout_distill_dataset`,
       which transparently applies the standard label-preserving
       perturbation block (mild brightness / contrast / noise /
       random crop, episodes seen twice per epoch).
    2. Initialises a fresh ``SmolVLAPolicy`` from the chosen student
       checkpoint (default: the public LeRobot SFT base; pass the
       Phase-2 WiSE-FT-merged ckpt for "anchor on the merge" distill).
    3. Runs the existing SFT training loop with hyperparameters tuned
       for distillation rather than fresh SFT (much lower LR, much
       fewer epochs, fixed-decay schedule).

Why the LR / epoch defaults differ from train_sft.py:
    * train_sft.py defaults to ``lr=1e-4`` and ``num-epochs=50``,
      because it trains a SmolVLA from the public LeRobot SFT base on
      a fresh task with maybe 50-200 demos.
    * Distillation has a fundamentally different setup: the student
      already approximates the teacher closely (it *is* either the SFT
      base or the WiSE-FT merge of SFT + RL teacher). We only need to
      polish away the teacher's residual mistakes without forgetting
      the SFT base, so ``lr=2e-5`` (5x lower) and ``num-epochs=10``
      (5x fewer) are appropriate. This recipe is documented in
      ``docs/thesis_research_plan.md`` Phase 4 and was derived from
      RIPT-VLA + WiSE-FT distill recipes adapted to SmolVLA scale.

For a no-augmentation A/B baseline, pass ``--no-augment``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys

import torch
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    rollouts: list[Path] = typer.Option(
        ...,
        "--rollouts",
        "-r",
        help="One or more ``.pt`` rollout files produced by scripts/collect_success_dataset.py.",
    ),
    student_checkpoint: str = typer.Option(
        "HuggingFaceVLA/smolvla_libero",
        "--student-checkpoint",
        "-c",
        help=(
            "HF id or local dir for the student's starting weights. "
            "Use the SFT base for vanilla distillation, or the WiSE-FT-merged "
            "checkpoint for 'anchor on merge' distillation."
        ),
    ),
    save_dir: Path | None = typer.Option(
        None,
        "--save-dir",
        help="Directory to write checkpoints into. Defaults to checkpoints/distill/<run-id>.",
    ),
    num_demos: int | None = typer.Option(
        None,
        "--num-demos",
        help="Optional cap on episodes per .pt file (None = use all).",
    ),
    seed: int = typer.Option(42, "--seed"),
    lr: float = typer.Option(2e-5, "--lr"),
    decay_lr: float = typer.Option(1e-6, "--decay-lr"),
    warmup_steps: int = typer.Option(200, "--warmup-steps"),
    decay_steps: int = typer.Option(5000, "--decay-steps"),
    batch_size: int = typer.Option(32, "--batch-size"),
    micro_batch_size: int = typer.Option(4, "--micro-batch-size"),
    num_epochs: int = typer.Option(10, "--epochs", "-e"),
    max_grad_norm: float = typer.Option(10.0, "--grad-clip-norm"),
    eval_every: int = typer.Option(2, "--eval-every"),
    eval_episodes: int = typer.Option(20, "--eval-episodes"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    simulator: str = typer.Option("libero", "--simulator"),
    eval_suite: str = typer.Option("spatial", "--eval-suite"),
    action_chunk_size: int = typer.Option(50, "--action-chunk-size"),
    augment: bool = typer.Option(
        True,
        "--augment/--no-augment",
        help="Toggle loader-time perturbations (color / contrast / noise / crop). Defaults on.",
    ),
    augment_repeats: int = typer.Option(2, "--augment-repeats"),
    augment_brightness: float = typer.Option(0.10, "--augment-brightness"),
    augment_contrast: float = typer.Option(0.10, "--augment-contrast"),
    augment_noise_std: float = typer.Option(0.02, "--augment-noise-std"),
    augment_crop_scale: float = typer.Option(0.92, "--augment-crop-scale"),
    instruction_variants_path: Path | None = typer.Option(
        None,
        "--instruction-variants",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="JSON mapping instruction -> paraphrase list; supports '*' wildcard default.",
    ),
    use_wandb: bool = typer.Option(True, "--wandb/--no-wandb"),
    wandb_project: str = typer.Option("srpo-smolvla", "--wandb-project"),
    wandb_name: str | None = typer.Option(None, "--wandb-name"),
    resume: str | None = typer.Option(None, "--resume"),
) -> None:
    """Self-distill SmolVLA from collected RL rollouts."""
    import wandb

    from vla.constants import CHECKPOINTS_DIR
    from vla.data.rollout_distill import build_rollout_distill_dataset
    from vla.models.smolvla import SmolVLAPolicy
    from vla.results_registry import (
        get_git_info,
        get_scheduler_info,
        now_iso,
        summarize_metrics_jsonl,
        write_json,
        write_training_registry,
    )
    from vla.training.metrics_logger import MetricsLogger
    from vla.training.sft_smolvla import SFTConfig, train_sft
    from vla.utils import get_device, run_id, seed_everything

    seed_everything(seed)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    variants: dict[str, list[str]] | None = None
    if instruction_variants_path is not None:
        loaded = json.loads(instruction_variants_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            variants = {"*": [str(v) for v in loaded]}
        elif isinstance(loaded, dict):
            variants = {str(k): [str(v) for v in values] for k, values in loaded.items() if isinstance(values, list)}
        else:
            raise typer.BadParameter("--instruction-variants must be a JSON list or object")

    dataset = build_rollout_distill_dataset(
        rollouts,
        num_demos=num_demos,
        seed=seed,
        action_chunk_size=action_chunk_size,
        repeats=augment_repeats,
        brightness=augment_brightness,
        contrast=augment_contrast,
        noise_std=augment_noise_std,
        random_crop_scale=augment_crop_scale,
        instruction_variants=variants,
        enable_augmentation=augment,
    )

    meta = dataset.metadata
    resolved_env_id = meta.get("env_id", f"libero_{eval_suite}")
    resolved_instruction = meta.get("instruction", "complete the manipulation task")
    resolved_max_steps = max_steps or meta.get("max_episode_steps") or 220
    resolved_control_mode = meta.get("control_mode", "libero_default")

    logger.info(
        "Distillation dataset: %d episodes / %d timesteps  source_checkpoint=%s",
        getattr(dataset, "num_episodes", "?"),
        len(dataset),
        meta.get("source_checkpoint", ""),
    )
    logger.info(
        "Student init: %s  augment=%s repeats=%d br=%.2f ct=%.2f noise=%.3f crop=%.2f",
        student_checkpoint,
        augment,
        augment_repeats,
        augment_brightness,
        augment_contrast,
        augment_noise_std,
        augment_crop_scale,
    )

    policy = SmolVLAPolicy(
        checkpoint=student_checkpoint,
        action_dim=dataset.action_dim,
        state_dim=dataset.state_dim,
        device=str(device),
    )
    if resume:
        logger.info("Resuming policy weights from %s", resume)
        policy.load_checkpoint(resume)

    if save_dir is None:
        save_dir = CHECKPOINTS_DIR / "distill" / f"libero_{eval_suite}_seed{seed}_{run_id()}"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = SFTConfig(
        lr=lr,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_lr=decay_lr,
        max_grad_norm=max_grad_norm,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        max_steps=resolved_max_steps,
        save_dir=str(save_dir),
        env_id=resolved_env_id,
        seed=seed,
        simulator=simulator,
        eval_suite=eval_suite,
        control_mode=resolved_control_mode,
        resume_from=resume,
    )

    run = None
    final_name = wandb_name or f"distill_libero_{eval_suite}_seed{seed}"
    if use_wandb:
        wb_config = config.to_dict()
        wb_config.update(
            method="distill",
            student_checkpoint=student_checkpoint,
            rollout_paths=[str(p) for p in rollouts],
            num_demos=num_demos,
            data_source="pt",
            augment=augment,
            augment_repeats=augment_repeats,
            augment_brightness=augment_brightness,
            augment_contrast=augment_contrast,
            augment_noise_std=augment_noise_std,
            augment_crop_scale=augment_crop_scale,
            source_teacher_checkpoint=meta.get("source_checkpoint", ""),
        )
        run = wandb.init(
            project=wandb_project,
            name=final_name,
            config=wb_config,
            resume="allow" if resume else None,
        )

    metrics_jsonl_path = save_dir / "metrics.jsonl"
    training_record = {
        "record_type": "training",
        "recorded_at": now_iso(),
        "completed_at": None,
        "method": "distill",
        "save_dir": str(save_dir),
        "best_checkpoint_dir": str(save_dir / "best"),
        "last_checkpoint_dir": str(save_dir / "last"),
        "checkpoint": student_checkpoint,
        "resume_from": resume or "",
        "simulator": simulator,
        "suite": eval_suite,
        "env_id": resolved_env_id,
        "instruction": resolved_instruction,
        "seed": seed,
        "num_demos": num_demos,
        "wandb_run_name": final_name if use_wandb else "",
        "data_source": "pt",
        "data_tag": "rollout_distill",
        "rollout_paths": [str(p) for p in rollouts],
        "source_teacher_checkpoint": meta.get("source_checkpoint", ""),
        "config": config.to_dict(),
        "metrics_jsonl": str(metrics_jsonl_path),
        **get_git_info(),
        **get_scheduler_info(),
    }
    write_json(save_dir / "training_run.json", training_record)

    ml = MetricsLogger(jsonl_path=metrics_jsonl_path, wandb_run=run)
    train_sft(policy, dataset, config, metrics_logger=ml, instruction=resolved_instruction)

    training_record["completed_at"] = now_iso()
    training_record.update(
        summarize_metrics_jsonl(
            metrics_jsonl_path,
            eval_key_suffixes=["sft/success_rate"],
        )
    )
    write_json(save_dir / "training_run.json", training_record)
    write_training_registry(training_record)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    app()
