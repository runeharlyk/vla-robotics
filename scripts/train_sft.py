"""Supervised fine-tuning (behavior cloning) of SmolVLA.

Supports two data sources:
  1. Preprocessed .pt files (ManiSkill or any simulator):
       uv run python scripts/train_sft.py --data data/preprocessed/pickcube.pt

  2. Libero datasets loaded directly from HuggingFace (no preprocessing):
       uv run python scripts/train_sft.py --libero-suite spatial

Other examples:
    # Combine multiple .pt files:
    uv run python scripts/train_sft.py --data pickcube.pt --data stackcube.pt

    # Train on ManiSkill data, evaluate in Libero:
    uv run python scripts/train_sft.py --data pickcube.pt --simulator libero --eval-suite spatial

    # Resume a crashed job:
    uv run python scripts/train_sft.py --data pickcube.pt --resume checkpoints/sft/.../last
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import typer

import wandb
from vla.constants import CHECKPOINTS_DIR, PREPROCESSED_DIR
from vla.data.dataset import ConcatFewDemoDataset, FewDemoDataset
from vla.models.smolvla import SmolVLAPolicy
from vla.training.sft_smolvla import SFTConfig, train_sft
from vla.utils import get_device, run_id, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _discover_pt(data_paths: list[Path] | None) -> list[Path]:
    """Return a list of concrete .pt paths.

    If *data_paths* is empty / ``None``, auto-discover in ``data/preprocessed/``.
    Each entry can be a file or a directory (in which case all ``.pt`` files
    inside are collected).
    """
    if not data_paths:
        pts = sorted(PREPROCESSED_DIR.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(
                f"No .pt files in {PREPROCESSED_DIR}. Run `python scripts/preprocess_data.py` first."
            )
        if len(pts) > 1:
            names = ", ".join(p.name for p in pts)
            logger.warning(
                "Multiple .pt files found (%s). Using %s. Pass --data explicitly to choose.",
                names,
                pts[0].name,
            )
            return [pts[0]]
        return pts

    result: list[Path] = []
    for p in data_paths:
        if p.is_file():
            result.append(p)
        elif p.is_dir():
            result.extend(sorted(p.glob("*.pt")))
        else:
            raise FileNotFoundError(f"Data path does not exist: {p}")
    if not result:
        raise FileNotFoundError(f"No .pt files found from: {data_paths}")
    return result


def main(
    data_path: list[Path] = typer.Option(
        None, "--data", "-d", help="Preprocessed .pt file(s) or directory. Repeat for multi-dataset."
    ),
    libero_suite: str = typer.Option(
        None,
        "--libero-suite",
        "-l",
        help="Load Libero data directly from HF (spatial, object, goal, long). Mutually exclusive with --data.",
    ),
    num_demos: int = typer.Option(None, "--num-demos", "-n", help="Episodes to use (default: all)"),
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint", "-c"),
    lr: float = typer.Option(1e-4, "--lr"),
    batch_size: int = typer.Option(32, "--batch-size"),
    micro_batch_size: int = typer.Option(4, "--micro-batch-size"),
    num_epochs: int = typer.Option(50, "--epochs"),
    warmup_steps: int = typer.Option(1000, "--warmup-steps"),
    decay_steps: int = typer.Option(30000, "--decay-steps"),
    decay_lr: float = typer.Option(2.5e-6, "--decay-lr"),
    grad_clip_norm: float = typer.Option(10.0, "--grad-clip-norm"),
    eval_every: int = typer.Option(5, "--eval-every"),
    eval_episodes: int = typer.Option(50, "--eval-episodes"),
    max_steps: int = typer.Option(None, "--max-steps", help="Override max episode steps (default: from metadata)"),
    seed: int = typer.Option(42, "--seed"),
    env_id: str = typer.Option(None, "--env", help="Override env id (default: from data metadata)"),
    simulator: str = typer.Option(
        None,
        "--simulator",
        "-s",
        help="Simulator for periodic eval: maniskill or libero (default: auto from data source)",
    ),
    eval_suite: str = typer.Option("all", "--eval-suite", help="Libero suite for eval"),
    resume: str = typer.Option(
        None,
        "--resume",
        "-r",
        help="Resume from checkpoint dir (e.g. checkpoints/sft/.../last)",
    ),
    use_wandb: bool = typer.Option(True, "--wandb/--no-wandb"),
) -> None:
    """Fine-tune SmolVLA via behaviour cloning.

    Data can come from preprocessed ``.pt`` files (``--data``) or directly
    from Libero HuggingFace datasets (``--libero-suite``).

    Supports checkpoint resume (``--resume``) to recover from crashes.
    """
    if data_path and libero_suite:
        raise typer.BadParameter("Specify either --data or --libero-suite, not both.")

    seed_everything(seed)
    device = get_device()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if libero_suite:
        from vla.data.libero import LiberoSFTDataset

        dataset = LiberoSFTDataset(libero_suite, num_demos=num_demos, seed=seed)
        data_tag = f"libero_{libero_suite}"
        default_simulator = "libero"
    else:
        pt_paths = _discover_pt(data_path)
        if len(pt_paths) == 1:
            dataset: FewDemoDataset | ConcatFewDemoDataset = FewDemoDataset(pt_paths[0], num_demos=num_demos, seed=seed)
        else:
            logger.info("Combining %d datasets: %s", len(pt_paths), [p.name for p in pt_paths])
            dataset = ConcatFewDemoDataset(pt_paths, num_demos=num_demos, seed=seed)
        data_tag = "multi" if len(pt_paths) > 1 else pt_paths[0].stem
        default_simulator = "maniskill"

    meta = dataset.metadata

    resolved_simulator = simulator or meta.get("simulator", default_simulator)
    resolved_env_id = env_id or meta.get("env_id", meta.get("skill", "PickCube-v1"))
    resolved_instruction = meta.get("instruction", "complete the manipulation task")
    resolved_max_steps = max_steps or meta.get("max_episode_steps") or 200
    resolved_control_mode = meta.get("control_mode", "pd_joint_delta_pos")

    logger.info(
        "Dataset: %d episodes, %d timesteps\n"
        "  env_id=%s  instruction=%r\n"
        "  action_dim=%d  state_dim=%d  control_mode=%s\n"
        "  eval_simulator=%s  eval_suite=%s",
        dataset.num_episodes,
        len(dataset),
        resolved_env_id,
        resolved_instruction,
        dataset.action_dim,
        dataset.state_dim,
        resolved_control_mode,
        resolved_simulator,
        eval_suite,
    )

    policy = SmolVLAPolicy(
        checkpoint=checkpoint,
        action_dim=dataset.action_dim,
        state_dim=dataset.state_dim,
        device=str(device),
    )

    if resume:
        logger.info("Loading policy weights from %s", resume)
        policy.load_checkpoint(resume)

    task_tag = resolved_env_id.lower().replace("-", "_")
    demos_tag = f"demos{num_demos}" if num_demos is not None else "all"
    save_dir = str(CHECKPOINTS_DIR / "sft" / f"{task_tag}_{demos_tag}_seed{seed}_{run_id()}")

    config = SFTConfig(
        lr=lr,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        num_epochs=num_epochs,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_lr=decay_lr,
        grad_clip_norm=grad_clip_norm,
        eval_every=eval_every,
        eval_episodes=eval_episodes,
        max_steps=resolved_max_steps,
        save_dir=save_dir,
        env_id=resolved_env_id,
        seed=seed,
        simulator=resolved_simulator,
        eval_suite=eval_suite,
        control_mode=resolved_control_mode,
        resume_from=resume,
    )

    run = None
    if use_wandb:
        run = wandb.init(
            project="srpo-smolvla",
            name=f"sft_{data_tag}_{demos_tag}_seed{seed}",
            config={
                "method": "sft",
                "task": resolved_env_id,
                "instruction": resolved_instruction,
                "num_demos": num_demos,
                "seed": seed,
                "lr": lr,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "checkpoint": checkpoint,
                "env_id": resolved_env_id,
                "control_mode": resolved_control_mode,
                "action_dim": dataset.action_dim,
                "state_dim": dataset.state_dim,
                "simulator": resolved_simulator,
                "eval_suite": eval_suite,
                "data_source": "libero" if libero_suite else "pt",
                "resume": resume or "",
            },
            resume="allow" if resume else None,
        )

    train_sft(policy, dataset, config, wandb_run=run, instruction=resolved_instruction)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    typer.run(main)
