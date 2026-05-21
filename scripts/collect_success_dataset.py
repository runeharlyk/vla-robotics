"""Collect successful LIBERO rollouts and export them for SFT/LeRobot-style BC.

Example:
    uv run python scripts/collect_success_dataset.py \
        --checkpoint HuggingFaceVLA/smolvla_libero \
        --suite spatial \
        --task-ids all \
        --successes-per-task 100 \
        --num-envs 8 \
        --n-action-steps 5 \
        --output data/collected/libero_spatial_success100_chunk5.pt
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from vla.constants import ACTION_DIM, DATA_DIR, SUITE_MAP
from vla.models.smolvla import SmolVLAPolicy
from vla.rl.libero_rollout import LiberoRollout
from vla.rl.rollout import Trajectory
from vla.rl.trajectory_io import save_trajectories_as_sft_pt
from vla.utils import get_device, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _parse_task_ids(value: str) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(10))
    task_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not task_ids:
        raise typer.BadParameter("--task-ids must be 'all' or a comma-separated list")
    return task_ids


@app.command()
def main(
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", "--checkpoint"),
    checkpoint_dir: Path | None = typer.Option(
        None,
        "--checkpoint-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help=(
            "Optional local checkpoint directory (must contain policy.pt). "
            "When given, the SmolVLA architecture is built from --checkpoint (HF id) "
            "and then the local weights are loaded on top -- same as scripts/evaluate.py."
        ),
    ),
    output: Path = typer.Option(DATA_DIR / "collected" / "libero_success.pt", "--output"),
    suite: str = typer.Option("spatial", "--suite", help="LIBERO suite: spatial, object, goal, long."),
    task_ids: str = typer.Option("all", "--task-ids", help="'all' or comma-separated task indices, e.g. 0,3,5."),
    successes_per_task: int = typer.Option(100, "--successes-per-task", min=1),
    max_attempts_per_task: int = typer.Option(1000, "--max-attempts-per-task", min=1),
    num_envs: int = typer.Option(8, "--num-envs", min=1),
    attempts_per_batch: int | None = typer.Option(
        None,
        "--attempts-per-batch",
        help="Rollout attempts per collection call. Defaults to num_envs.",
    ),
    n_action_steps: int = typer.Option(5, "--n-action-steps", min=1),
    action_chunk_size: int = typer.Option(50, "--action-chunk-size", min=1),
    max_steps: int = typer.Option(300, "--max-steps", min=1),
    seed: int = typer.Option(42, "--seed"),
    state_dim: int = typer.Option(8, "--state-dim", min=1),
    image_size: int = typer.Option(256, "--image-size", min=1),
) -> None:
    """Collect successful trajectories from a checkpoint into one ``.pt`` file."""
    seed_everything(seed)
    device = get_device()
    suite_key = suite.lower()
    if suite_key not in SUITE_MAP:
        raise typer.BadParameter(f"Unknown suite {suite!r}; available: {list(SUITE_MAP)}")
    parsed_task_ids = _parse_task_ids(task_ids)
    batch_attempts = attempts_per_batch or num_envs

    policy = SmolVLAPolicy(
        checkpoint=checkpoint,
        action_dim=ACTION_DIM,
        state_dim=state_dim,
        device=str(device),
    )
    if checkpoint_dir is not None:
        logger.info("Loading local checkpoint weights from %s", checkpoint_dir)
        policy.load_checkpoint(checkpoint_dir)

    all_successes: list[Trajectory] = []
    instructions_by_task: dict[str, str] = {}

    rollout: LiberoRollout | None = None
    try:
        for task_id in parsed_task_ids:
            if rollout is None:
                rollout = LiberoRollout(
                    suite_name=suite_key,
                    task_id=task_id,
                    num_envs=num_envs,
                    max_steps=max_steps,
                    image_size=image_size,
                    state_dim=state_dim,
                )
            else:
                rollout.reconfigure(suite_key, task_id)

            task_key = f"{suite_key}_task_{task_id}"
            instructions_by_task[task_key] = rollout.task_description
            task_successes: list[Trajectory] = []
            attempts = 0

            while len(task_successes) < successes_per_task and attempts < max_attempts_per_task:
                remaining_attempts = max_attempts_per_task - attempts
                n_attempts = min(batch_attempts, remaining_attempts)
                batch_seed = seed + task_id * 100_000 + attempts
                trajs = rollout.collect_batch(
                    policy_fn=policy.predict_action,
                    instruction=rollout.task_description,
                    num_trajectories=n_attempts,
                    seed=batch_seed,
                    policy_batch_fn=policy.predict_action_batch,
                    n_action_steps=n_action_steps,
                    policy_chunk_fn=policy.predict_action_chunk if n_action_steps > 1 else None,
                    policy_chunk_batch_fn=policy.predict_action_chunk_batch if n_action_steps > 1 else None,
                )
                for traj_idx, traj in enumerate(trajs):
                    traj.reset_seed = batch_seed + traj_idx
                attempts += len(trajs)
                new_successes = [traj for traj in trajs if traj.success]
                for traj in new_successes:
                    traj.task_id = task_key
                task_successes.extend(new_successes)
                logger.info(
                    "suite=%s task=%d successes=%d/%d attempts=%d/%d",
                    suite_key,
                    task_id,
                    len(task_successes),
                    successes_per_task,
                    attempts,
                    max_attempts_per_task,
                )

            selected = task_successes[:successes_per_task]
            all_successes.extend(selected)
            if len(selected) < successes_per_task:
                logger.warning(
                    "Task %d only reached %d/%d successes within %d attempts",
                    task_id,
                    len(selected),
                    successes_per_task,
                    max_attempts_per_task,
                )
    finally:
        if rollout is not None:
            rollout.close()

    saved = save_trajectories_as_sft_pt(
        all_successes,
        output,
        metadata={
            "env_id": f"libero_{suite_key}",
            "suite": suite_key,
            "simulator": "libero",
            "source_checkpoint": checkpoint,
            "source_checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else "",
            "n_action_steps": n_action_steps,
            "max_steps": max_steps,
            "seed": seed,
            "num_envs": num_envs,
            "successes_per_task_target": successes_per_task,
        },
        default_instruction="complete the LIBERO task",
        instructions_by_task=instructions_by_task,
        only_successful=True,
        action_chunk_size=action_chunk_size,
    )
    logger.info("Saved %d successful trajectories to %s", len(all_successes), saved)


if __name__ == "__main__":
    app()
