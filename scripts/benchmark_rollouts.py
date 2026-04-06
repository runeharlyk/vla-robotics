"""Benchmark rollout collection scaling for ManiSkill and LIBERO.

This measures **rollout collection only** using a deterministic zero-action
policy so results reflect simulator/vectorization overhead rather than policy
quality, reward-model cost, or update-time cost.

Examples:
    # Local ManiSkill quick check
    ./.venv/bin/python scripts/benchmark_rollouts.py \
        --simulator maniskill \
        --maniskill-env PickCube-v1 \
        --maniskill-env-counts 1,2,4,8,16 \
        --repeats 2 \
        --warmup-runs 1

    # Compare both backends
    ./.venv/bin/python scripts/benchmark_rollouts.py \
        --simulator all \
        --maniskill-env-counts 1,2,4,8,16,32,64,128 \
        --libero-env-counts 1,2,4,8,16
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import socket
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import typer

from vla.constants import ACTION_DIM, MANISKILL_TASKS, OUTPUTS_DIR
from vla.rl.libero_rollout import LiberoRollout
from vla.rl.maniskill_rollout import ManiSkillRollout

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRow:
    simulator: str
    task: str
    env_count: int
    num_trajectories: int
    max_steps: int
    repeat_idx: int
    init_sec: float | None
    collect_sec: float | None
    trajs_per_sec: float | None
    mean_episode_length: float | None
    success_rate: float | None
    peak_cuda_mem_mb: float | None
    status: str
    error: str = ""


def _parse_counts(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _ordered_counts(simulator: str, counts: list[int]) -> list[int]:
    if simulator == "maniskill":
        # GPU PhysX must be enabled before any CPU PhysX-backed env is created.
        # Run vectorized counts first and benchmark the 1-env CPU fallback last.
        return sorted(counts, key=lambda n: (n == 1, n))
    return counts


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_cuda_mem_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024**2)


def _zero_policy(action_dim: int):
    def _single(_image: torch.Tensor, _instruction: str, state: torch.Tensor | None = None) -> torch.Tensor:
        device = state.device if state is not None else torch.device("cpu")
        return torch.zeros(action_dim, dtype=torch.float32, device=device)

    def _batch(images: torch.Tensor, _instruction: str, states: torch.Tensor | None = None) -> torch.Tensor:
        batch_size = images.shape[0]
        device = states.device if states is not None else images.device
        return torch.zeros(batch_size, action_dim, dtype=torch.float32, device=device)

    return _single, _batch


def _make_maniskill_rollout(env_id: str, env_count: int, max_steps: int) -> tuple[ManiSkillRollout, str, int]:
    meta = MANISKILL_TASKS.get(env_id, {})
    instruction = meta.get("instruction", "complete the manipulation task")
    action_dim = int(meta.get("action_dim", 8))
    num_cameras = int(meta.get("num_cameras", 2))
    rollout = ManiSkillRollout(
        env_id=env_id,
        num_envs=env_count,
        max_steps=max_steps,
        num_cameras=num_cameras,
    )
    return rollout, instruction, action_dim


def _make_libero_rollout(
    suite: str,
    task_id: int,
    env_count: int,
    max_steps: int,
    state_dim: int,
) -> tuple[LiberoRollout, str, int]:
    rollout = LiberoRollout(
        suite_name=suite,
        task_id=task_id,
        num_envs=env_count,
        max_steps=max_steps,
        state_dim=state_dim,
    )
    return rollout, rollout.task_description, ACTION_DIM


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[BenchmarkRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[BenchmarkRow]] = defaultdict(list)
    for row in rows:
        if row.status == "ok":
            grouped[(row.simulator, row.task, row.env_count)].append(row)

    baseline_tps: dict[tuple[str, str], float] = {}
    env_counts_by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for simulator, task, env_count in grouped:
        env_counts_by_key[(simulator, task)].append(env_count)
    for key, env_counts in env_counts_by_key.items():
        baseline_env_count = min(env_counts)
        baseline_group = grouped[(key[0], key[1], baseline_env_count)]
        baseline_tps[key] = statistics.fmean(r.trajs_per_sec for r in baseline_group if r.trajs_per_sec is not None)

    summary: list[dict[str, Any]] = []
    for (simulator, task, env_count), group in sorted(grouped.items()):
        collect_secs = [r.collect_sec for r in group if r.collect_sec is not None]
        tps = [r.trajs_per_sec for r in group if r.trajs_per_sec is not None]
        ep_len = [r.mean_episode_length for r in group if r.mean_episode_length is not None]
        mem = [r.peak_cuda_mem_mb for r in group if r.peak_cuda_mem_mb is not None]
        mean_tps = statistics.fmean(tps)
        baseline = baseline_tps[(simulator, task)]
        summary.append(
            {
                "simulator": simulator,
                "task": task,
                "env_count": env_count,
                "num_trajectories": group[0].num_trajectories,
                "repeats": len(group),
                "init_sec": round(group[0].init_sec or 0.0, 4),
                "collect_sec_mean": round(statistics.fmean(collect_secs), 4),
                "collect_sec_std": round(statistics.pstdev(collect_secs), 4) if len(collect_secs) > 1 else 0.0,
                "trajs_per_sec_mean": round(mean_tps, 4),
                "trajs_per_sec_std": round(statistics.pstdev(tps), 4) if len(tps) > 1 else 0.0,
                "speedup_vs_1env": round(mean_tps / max(baseline, 1e-8), 4),
                "mean_episode_length": round(statistics.fmean(ep_len), 2),
                "peak_cuda_mem_mb_mean": round(statistics.fmean(mem), 2) if mem else "",
            }
        )
    return summary


def _print_summary(summary_rows: list[dict[str, Any]]) -> None:
    by_sim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_sim[row["simulator"]].append(row)

    for simulator, rows in by_sim.items():
        print(f"\n=== {simulator.upper()} ===")
        print("envs | trajs | collect_s | trajs/s | speedup_vs_1env | peak_cuda_mem_mb")
        for row in rows:
            print(
                f"{row['env_count']:>4} | "
                f"{row['num_trajectories']:>5} | "
                f"{row['collect_sec_mean']:>9} | "
                f"{row['trajs_per_sec_mean']:>7} | "
                f"{row['speedup_vs_1env']:>15} | "
                f"{row['peak_cuda_mem_mb_mean']}"
            )


def main(
    simulator: str = typer.Option("all", help="maniskill, libero, or all"),
    maniskill_env: str = typer.Option("PickCube-v1", help="ManiSkill env id"),
    libero_suite: str = typer.Option("spatial", help="LIBERO suite name"),
    libero_task_id: int = typer.Option(0, help="LIBERO task id"),
    max_steps: int = typer.Option(100, help="Episode horizon used for both simulators"),
    maniskill_env_counts: str = typer.Option("1,2,4,8,16,32,64,128"),
    libero_env_counts: str = typer.Option("1,2,4,8,16"),
    num_trajectories: int = typer.Option(
        0, help="Trajectories to collect per run. 0 means match env count for a single wave."
    ),
    warmup_runs: int = typer.Option(1, help="Discarded warmup runs per env count"),
    repeats: int = typer.Option(3, help="Measured repeats per env count"),
    seed: int = typer.Option(42),
    libero_state_dim: int = typer.Option(8, help="LIBERO robot state dim"),
    continue_on_error: bool = typer.Option(True, "--continue-on-error/--fail-fast"),
    output_dir: Path | None = typer.Option(None, path_type=Path),
) -> None:
    sim = simulator.lower()
    selected = ["maniskill", "libero"] if sim == "all" else [sim]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = (output_dir or (OUTPUTS_DIR / "benchmarks" / f"rollouts_{timestamp}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    system_info = {
        "timestamp": timestamp,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "gpu_total_mem_mb": (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2) if torch.cuda.is_available() else ""
        ),
        "max_steps": max_steps,
        "warmup_runs": warmup_runs,
        "repeats": repeats,
        "seed": seed,
    }

    rows: list[BenchmarkRow] = []
    logger.info("Saving benchmark outputs to %s", out_dir)
    logger.info("System info: %s", system_info)

    for current_sim in selected:
        counts = _ordered_counts(
            current_sim,
            _parse_counts(maniskill_env_counts if current_sim == "maniskill" else libero_env_counts),
        )
        task_label = maniskill_env if current_sim == "maniskill" else f"libero_{libero_suite}_task_{libero_task_id}"

        for env_count in counts:
            measured_trajs = num_trajectories if num_trajectories > 0 else env_count
            logger.info(
                "[%s] env_count=%d num_trajectories=%d max_steps=%d",
                current_sim,
                env_count,
                measured_trajs,
                max_steps,
            )

            engine = None
            init_sec: float | None = None
            try:
                t0 = time.monotonic()
                if current_sim == "maniskill":
                    engine, instruction, action_dim = _make_maniskill_rollout(maniskill_env, env_count, max_steps)
                else:
                    engine, instruction, action_dim = _make_libero_rollout(
                        libero_suite,
                        libero_task_id,
                        env_count,
                        max_steps,
                        libero_state_dim,
                    )
                init_sec = time.monotonic() - t0
                policy_fn, policy_batch_fn = _zero_policy(action_dim)

                for warmup_idx in range(warmup_runs):
                    logger.info("[%s] warmup %d/%d @ env_count=%d", current_sim, warmup_idx + 1, warmup_runs, env_count)
                    _sync_cuda()
                    engine.collect_batch(
                        policy_fn=policy_fn,
                        instruction=instruction,
                        num_trajectories=measured_trajs,
                        seed=seed + warmup_idx,
                        policy_batch_fn=policy_batch_fn,
                    )
                    _sync_cuda()

                for repeat_idx in range(repeats):
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    _sync_cuda()
                    t1 = time.monotonic()
                    trajectories = engine.collect_batch(
                        policy_fn=policy_fn,
                        instruction=instruction,
                        num_trajectories=measured_trajs,
                        seed=seed + 10_000 * (repeat_idx + 1),
                        policy_batch_fn=policy_batch_fn,
                    )
                    _sync_cuda()
                    collect_sec = time.monotonic() - t1
                    mean_ep_len = statistics.fmean(t.length for t in trajectories)
                    success_rate = sum(1 for t in trajectories if t.success) / max(len(trajectories), 1)

                    rows.append(
                        BenchmarkRow(
                            simulator=current_sim,
                            task=task_label,
                            env_count=env_count,
                            num_trajectories=measured_trajs,
                            max_steps=max_steps,
                            repeat_idx=repeat_idx,
                            init_sec=init_sec,
                            collect_sec=collect_sec,
                            trajs_per_sec=measured_trajs / max(collect_sec, 1e-8),
                            mean_episode_length=mean_ep_len,
                            success_rate=success_rate,
                            peak_cuda_mem_mb=_peak_cuda_mem_mb(),
                            status="ok",
                        )
                    )
                    logger.info(
                        "[%s] envs=%d repeat=%d collect=%.3fs trajs/s=%.3f",
                        current_sim,
                        env_count,
                        repeat_idx + 1,
                        collect_sec,
                        measured_trajs / max(collect_sec, 1e-8),
                    )
            except Exception as exc:
                rows.append(
                    BenchmarkRow(
                        simulator=current_sim,
                        task=task_label,
                        env_count=env_count,
                        num_trajectories=measured_trajs,
                        max_steps=max_steps,
                        repeat_idx=-1,
                        init_sec=init_sec,
                        collect_sec=None,
                        trajs_per_sec=None,
                        mean_episode_length=None,
                        success_rate=None,
                        peak_cuda_mem_mb=None,
                        status="error",
                        error=repr(exc),
                    )
                )
                logger.exception("[%s] benchmark failed for env_count=%d", current_sim, env_count)
                if not continue_on_error:
                    raise
            finally:
                if engine is not None:
                    engine.close()

    raw_path = out_dir / "raw_results.csv"
    summary_path = out_dir / "summary.csv"
    meta_path = out_dir / "metadata.json"
    summary_rows = _aggregate(rows)

    _write_csv(raw_path, [asdict(r) for r in rows])
    _write_csv(summary_path, summary_rows)
    meta_path.write_text(json.dumps(system_info, indent=2))

    logger.info("Wrote raw results to %s", raw_path)
    logger.info("Wrote summary to %s", summary_path)
    logger.info("Wrote metadata to %s", meta_path)
    _print_summary(summary_rows)


if __name__ == "__main__":
    typer.run(main)
