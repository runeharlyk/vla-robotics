"""Export LIBERO suites from HuggingFace into the repo's SFT ``.pt`` schema.

The replay-perturbation tool (``scripts/replay_perturbed_dataset.py``) and the
``FewDemoDataset`` SFT loader both consume the ``.pt`` format produced by
:func:`vla.rl.trajectory_io.save_trajectories_as_sft_pt`.  This script
materialises that schema directly from ``LiberoSFTDataset`` for any of the four
LIBERO suites (spatial, object, goal, long), preserving the per-episode
``init_state_id`` so downstream sim replay can land in the exact recorded
starting configuration.

Example:

    # All 4 suites in one go:
    uv run python scripts/export_libero_to_pt.py --suite all

    # One specific suite, capping demos per task:
    uv run python scripts/export_libero_to_pt.py --suite goal --num-demos-per-task 50
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import typer

from vla.constants import LIBERO_SUITES, PREPROCESSED_DIR, SUITE_MAP
from vla.data.libero import LiberoSFTDataset
from vla.rl.trajectory_io import save_trajectories_as_sft_pt


def _images_to_uint8(images: torch.Tensor) -> torch.Tensor:
    """Cast image tensor to uint8 in the [0, 255] range that FewDemoDataset expects.

    LiberoSFTDataset returns images as float32 in [0, 1]; SFT/replay loaders
    keep images as uint8 in [0, 255] to avoid the 4x memory blow-up.
    """
    if images.dtype == torch.uint8:
        return images
    arr = images.detach().cpu()
    max_val = float(arr.max().item()) if arr.numel() else 0.0
    if max_val <= 2.0:
        arr = arr * 255.0
    return arr.clamp(0, 255).round().to(torch.uint8)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _resolve_suites(suite_arg: str) -> list[str]:
    if suite_arg.lower() == "all":
        return list(LIBERO_SUITES.keys())
    parts = [s.strip().lower() for s in suite_arg.split(",") if s.strip()]
    unknown = [s for s in parts if s not in LIBERO_SUITES]
    if unknown:
        raise typer.BadParameter(f"Unknown LIBERO suite(s): {unknown}. Available: {list(LIBERO_SUITES)}")
    return parts


def _export_one_suite(
    suite: str,
    output_dir: Path,
    num_demos_per_task: int | None,
    seed: int,
    action_chunk_size: int,
    num_tasks: int,
) -> Path:
    """Export every task of one LIBERO suite into a single ``.pt`` file."""
    all_trajectories = []
    instructions_by_task: dict[str, str] = {}

    for task_id in range(num_tasks):
        logger.info("Loading %s task %d", suite, task_id)
        dataset = LiberoSFTDataset(
            suite=suite,
            num_demos=num_demos_per_task,
            seed=seed,
            task_id=task_id,
            action_chunk_size=action_chunk_size,
        )
        if dataset.num_episodes == 0:
            logger.warning("No episodes found for %s task %d; skipping", suite, task_id)
            continue

        trajs = dataset.episodes_as_trajectories(task_id=task_id)
        task_key = f"{SUITE_MAP[suite]}_task_{task_id}"
        for traj in trajs:
            traj.task_id = task_key
            traj.is_demo = True
            traj.images = _images_to_uint8(traj.images)
        instruction = dataset._task_map.get(task_id, dataset.instruction)
        instructions_by_task[task_key] = instruction
        all_trajectories.extend(trajs)
        logger.info(
            "%s task %d: kept %d trajectories (instruction=%r)",
            suite,
            task_id,
            len(trajs),
            instruction,
        )

    if not all_trajectories:
        raise RuntimeError(f"No trajectories collected for suite {suite!r}")

    output_path = output_dir / f"{suite}.pt"
    metadata = {
        "simulator": "libero",
        "suite": suite,
        "env_id": f"libero_{suite}",
        "source_repo": LIBERO_SUITES[suite],
        "control_mode": "libero_default",
        "image_size": 256,
        "max_episode_steps": 220,
    }

    save_trajectories_as_sft_pt(
        trajectories=all_trajectories,
        path=output_path,
        metadata=metadata,
        default_instruction=next(iter(instructions_by_task.values()), "complete the manipulation task"),
        instructions_by_task=instructions_by_task,
        only_successful=False,
        action_chunk_size=action_chunk_size,
    )

    instructions_path = output_path.with_suffix(".instructions.json")
    instructions_path.write_text(
        json.dumps(instructions_by_task, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Wrote %d episodes to %s", len(all_trajectories), output_path)
    logger.info("Wrote per-task instructions to %s", instructions_path)
    return output_path


@app.command()
def main(
    suite: str = typer.Option(
        "all",
        "--suite",
        help="LIBERO suite name ('spatial', 'object', 'goal', 'long'), 'all', or comma-separated list.",
    ),
    output_dir: Path = typer.Option(
        PREPROCESSED_DIR,
        "--output-dir",
        help="Directory to write <suite>.pt files into. Defaults to data/preprocessed/.",
    ),
    num_demos_per_task: int | None = typer.Option(
        None,
        "--num-demos-per-task",
        help="Cap on episodes kept per task (None = use all available demos).",
    ),
    seed: int = typer.Option(42, "--seed", help="Subsampling seed when --num-demos-per-task is set."),
    action_chunk_size: int = typer.Option(50, "--action-chunk-size", min=1),
    num_tasks: int = typer.Option(
        10,
        "--num-tasks",
        help="Number of tasks per suite (LIBERO has 10 per suite by convention).",
    ),
) -> None:
    """Materialise the LIBERO suites into ``.pt`` files compatible with the SFT loader."""
    suites = _resolve_suites(suite)
    output_dir.mkdir(parents=True, exist_ok=True)

    for s in suites:
        _export_one_suite(
            suite=s,
            output_dir=output_dir,
            num_demos_per_task=num_demos_per_task,
            seed=seed,
            action_chunk_size=action_chunk_size,
            num_tasks=num_tasks,
        )


if __name__ == "__main__":
    app()
