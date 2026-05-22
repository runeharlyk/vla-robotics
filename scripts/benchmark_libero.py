"""Benchmark LIBERO rollout throughput and batched SmolVLA inference.

This script measures where time goes when scaling LIBERO subprocess count:
environment startup, reset, observation packing, policy inference, env.step,
and overall throughput.

Typical usage:

    uv run python scripts/benchmark_libero.py \
      --checkpoint HuggingFaceVLA/smolvla_libero \
      --suite spatial \
      --task-id 5 \
      --env-counts 8,16,24,32 \
      --benchmark-steps 40

For a CPU-only environment benchmark without model inference:

    uv run python scripts/benchmark_libero.py \
      --policy-mode zero \
      --suite spatial \
      --task-id 5 \
      --env-counts 8,16,24,32
"""

from __future__ import annotations

import contextlib
import json
import logging
import multiprocessing
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vla.models.smolvla import SmolVLAPolicy
from vla.rl.libero_rollout import LiberoRollout
from vla.utils import get_device, seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _parse_env_counts(raw: str) -> list[int]:
    counts: list[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        counts.append(int(chunk))
    unique = sorted(set(counts))
    if not unique:
        raise typer.BadParameter("env-counts must contain at least one positive integer")
    for count in unique:
        if count <= 0:
            raise typer.BadParameter("env-counts must be positive")
    return unique


def _safe_float_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _maybe_set_torch_threads(num_threads: int | None) -> None:
    if num_threads is None or num_threads <= 0:
        return
    torch.set_num_threads(num_threads)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(max(1, num_threads))


class PolicyLike(Protocol):
    action_dim: int

    def predict_action_batch(
        self,
        images: torch.Tensor,
        instruction: str,
        states: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class ZeroPolicy:
    def __init__(self, action_dim: int, device: torch.device) -> None:
        self.action_dim = action_dim
        self.device = device

    def predict_action_batch(
        self,
        images: torch.Tensor,
        instruction: str,
        states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = images.shape[0]
        return torch.zeros(batch_size, self.action_dim, device=self.device, dtype=torch.float32)


@dataclass
class BenchmarkResult:
    env_count: int
    startup_seconds: float
    initial_reset_seconds: float
    benchmark_steps: int
    resets_during_benchmark: int
    total_seconds: float
    obs_extract_seconds: float
    inference_seconds: float
    env_step_seconds: float
    reset_seconds: float
    steps_per_second: float
    env_steps_per_second: float
    obs_extract_pct: float
    inference_pct: float
    env_step_pct: float
    reset_pct: float
    peak_gpu_mem_gb: float | None


def _build_policy(
    *,
    policy_mode: str,
    checkpoint: str,
    checkpoint_dir: Path | None,
    action_dim: int,
    state_dim: int,
    device: torch.device,
) -> PolicyLike:
    if policy_mode == "zero":
        return ZeroPolicy(action_dim=action_dim, device=device)

    policy = SmolVLAPolicy(
        checkpoint=checkpoint,
        action_dim=action_dim,
        state_dim=state_dim,
        device=str(device),
    )
    if checkpoint_dir is not None:
        policy.load_checkpoint(checkpoint_dir)
    policy.eval()
    return policy


def _cpu_summary() -> dict[str, int | None]:
    return {
        "os_cpu_count": os.cpu_count(),
        "mp_cpu_count": multiprocessing.cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _print_system_summary(device: torch.device, env_counts: list[int]) -> None:
    typer.echo("System")
    typer.echo("------")
    cpu = _cpu_summary()
    typer.echo(f"device={device}")
    typer.echo(f"os_cpu_count={cpu['os_cpu_count']} mp_cpu_count={cpu['mp_cpu_count']}")
    typer.echo(
        f"torch_threads={cpu['torch_num_threads']} torch_interop_threads={cpu['torch_num_interop_threads']}"
    )
    typer.echo(f"requested_env_counts={env_counts}")
    typer.echo(
        "note: LIBERO uses one subprocess per env, so env_count > allocated CPU cores will oversubscribe the node."
    )


def _benchmark_env_count(
    *,
    env_count: int,
    policy: PolicyLike,
    suite: str,
    task_id: int,
    seed: int,
    benchmark_steps: int,
    warmup_steps: int,
    max_steps: int,
    image_size: int,
    state_dim: int,
    num_cameras: int,
) -> BenchmarkResult:
    startup_t0 = time.perf_counter()
    rollout = LiberoRollout(
        suite_name=suite,
        task_id=task_id,
        num_envs=env_count,
        max_steps=max_steps,
        image_size=image_size,
        state_dim=state_dim,
        num_cameras=num_cameras,
    )
    startup_seconds = time.perf_counter() - startup_t0
    instruction = rollout.task_description

    device = getattr(policy, "device", torch.device("cpu"))
    if isinstance(device, str):
        device = torch.device(device)

    try:
        reset_t0 = time.perf_counter()
        obs = rollout.vec_env.reset([(seed + i) for i in range(env_count)])
        initial_reset_seconds = time.perf_counter() - reset_t0

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        obs_extract_seconds = 0.0
        inference_seconds = 0.0
        env_step_seconds = 0.0
        reset_seconds = 0.0
        resets_during_benchmark = 0

        total_t0 = time.perf_counter()
        total_loop_steps = warmup_steps + benchmark_steps

        for step_idx in range(total_loop_steps):
            extract_t0 = time.perf_counter()
            batch_images = []
            batch_states = []
            for packed in obs:
                img_t, state_t = rollout._obs_to_tensors(packed)
                batch_images.append(img_t)
                batch_states.append(state_t)
            images_batch = torch.stack(batch_images, dim=0)
            states_batch = torch.stack(batch_states, dim=0)
            obs_extract_elapsed = time.perf_counter() - extract_t0

            infer_t0 = time.perf_counter()
            _sync_device(device)
            actions_t = policy.predict_action_batch(images_batch, instruction, states_batch)
            _sync_device(device)
            inference_elapsed = time.perf_counter() - infer_t0
            if isinstance(actions_t, torch.Tensor):
                actions = actions_t.detach().cpu().numpy()
            else:
                actions = np.asarray(actions_t, dtype=np.float32)

            step_t0 = time.perf_counter()
            obs, _rewards, terminateds, truncateds, _infos = rollout.vec_env.step(actions)
            step_elapsed = time.perf_counter() - step_t0

            if step_idx >= warmup_steps:
                obs_extract_seconds += obs_extract_elapsed
                inference_seconds += inference_elapsed
                env_step_seconds += step_elapsed

            if any(terminateds) or any(truncateds):
                reset_again_t0 = time.perf_counter()
                obs = rollout.vec_env.reset([(seed + env_count * (step_idx + 1) + i) for i in range(env_count)])
                reset_again_elapsed = time.perf_counter() - reset_again_t0
                if step_idx >= warmup_steps:
                    reset_seconds += reset_again_elapsed
                    resets_during_benchmark += 1

        total_seconds = time.perf_counter() - total_t0
        if warmup_steps > 0:
            warmup_fraction = warmup_steps / total_loop_steps
            total_seconds *= max(0.0, 1.0 - warmup_fraction)

        timed_total = obs_extract_seconds + inference_seconds + env_step_seconds + reset_seconds
        peak_gpu_mem_gb = None
        if device.type == "cuda":
            peak_gpu_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

        return BenchmarkResult(
            env_count=env_count,
            startup_seconds=startup_seconds,
            initial_reset_seconds=initial_reset_seconds,
            benchmark_steps=benchmark_steps,
            resets_during_benchmark=resets_during_benchmark,
            total_seconds=timed_total if timed_total > 0 else total_seconds,
            obs_extract_seconds=obs_extract_seconds,
            inference_seconds=inference_seconds,
            env_step_seconds=env_step_seconds,
            reset_seconds=reset_seconds,
            steps_per_second=_safe_float_div(benchmark_steps, timed_total if timed_total > 0 else total_seconds),
            env_steps_per_second=_safe_float_div(env_count * benchmark_steps, timed_total if timed_total > 0 else total_seconds),
            obs_extract_pct=_safe_float_div(obs_extract_seconds, timed_total) * 100.0 if timed_total > 0 else 0.0,
            inference_pct=_safe_float_div(inference_seconds, timed_total) * 100.0 if timed_total > 0 else 0.0,
            env_step_pct=_safe_float_div(env_step_seconds, timed_total) * 100.0 if timed_total > 0 else 0.0,
            reset_pct=_safe_float_div(reset_seconds, timed_total) * 100.0 if timed_total > 0 else 0.0,
            peak_gpu_mem_gb=peak_gpu_mem_gb,
        )
    finally:
        rollout.close()


def _print_result_table(results: list[BenchmarkResult]) -> None:
    typer.echo("")
    typer.echo("Results")
    typer.echo("-------")
    for result in results:
        typer.echo(
            f"envs={result.env_count:>2} "
            f"startup={result.startup_seconds:>6.1f}s "
            f"reset0={result.initial_reset_seconds:>6.1f}s "
            f"steps/s={result.steps_per_second:>6.2f} "
            f"env_steps/s={result.env_steps_per_second:>8.2f} "
            f"infer={result.inference_seconds:>6.1f}s ({result.inference_pct:>5.1f}%) "
            f"step={result.env_step_seconds:>6.1f}s ({result.env_step_pct:>5.1f}%) "
            f"obs={result.obs_extract_seconds:>6.1f}s ({result.obs_extract_pct:>5.1f}%) "
            f"reset={result.reset_seconds:>6.1f}s ({result.reset_pct:>5.1f}%) "
            f"resets={result.resets_during_benchmark}"
        )


@app.command()
def main(
    checkpoint: str = typer.Option("HuggingFaceVLA/smolvla_libero", help="Base HuggingFace SmolVLA checkpoint."),
    checkpoint_dir: Path | None = typer.Option(None, path_type=Path, help="Optional local fine-tuned checkpoint dir."),
    policy_mode: str = typer.Option("smolvla", help="smolvla or zero."),
    suite: str = typer.Option("spatial", help="LIBERO suite key: spatial/object/goal/long/all."),
    task_id: int = typer.Option(0, help="LIBERO task id to benchmark."),
    env_counts: str = typer.Option("8,16,24,32", help="Comma-separated env counts to benchmark."),
    benchmark_steps: int = typer.Option(40, help="Number of timed vector steps per env-count run."),
    warmup_steps: int = typer.Option(5, help="Warmup vector steps excluded from timed totals."),
    max_steps: int = typer.Option(220, help="Episode horizon used by the env."),
    image_size: int = typer.Option(256, help="Image size passed to the rollout packer."),
    num_cameras: int = typer.Option(2, help="Expected number of camera views."),
    action_dim: int = typer.Option(7, help="Policy action dimension."),
    state_dim: int = typer.Option(8, help="Policy state dimension."),
    seed: int = typer.Option(0, help="Base seed."),
    device_name: str = typer.Option("cuda", help="cuda, cpu, or mps."),
    torch_threads: int | None = typer.Option(
        1,
        help="Set torch intra-op threads. Use 1 on HPC to avoid thread explosion across many env subprocesses.",
    ),
    output_json: Path | None = typer.Option(None, path_type=Path, help="Optional JSON output path."),
) -> None:
    if policy_mode not in {"smolvla", "zero"}:
        raise typer.BadParameter("policy-mode must be 'smolvla' or 'zero'")

    if benchmark_steps <= 0:
        raise typer.BadParameter("benchmark-steps must be positive")
    if warmup_steps < 0:
        raise typer.BadParameter("warmup-steps must be non-negative")

    _maybe_set_torch_threads(torch_threads)
    seed_everything(seed)
    device = get_device(device_name)
    counts = _parse_env_counts(env_counts)
    _print_system_summary(device, counts)

    policy = _build_policy(
        policy_mode=policy_mode,
        checkpoint=checkpoint,
        checkpoint_dir=checkpoint_dir,
        action_dim=action_dim,
        state_dim=state_dim,
        device=device,
    )

    results: list[BenchmarkResult] = []
    for env_count in counts:
        typer.echo("")
        typer.echo(f"Benchmarking env_count={env_count}")
        typer.echo("-------------------------")
        result = _benchmark_env_count(
            env_count=env_count,
            policy=policy,
            suite=suite,
            task_id=task_id,
            seed=seed,
            benchmark_steps=benchmark_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            image_size=image_size,
            state_dim=state_dim,
            num_cameras=num_cameras,
        )
        results.append(result)
        typer.echo(
            f"envs={result.env_count} env_steps/s={result.env_steps_per_second:.2f} "
            f"steps/s={result.steps_per_second:.2f} "
            f"infer_pct={result.inference_pct:.1f} step_pct={result.env_step_pct:.1f}"
        )

    _print_result_table(results)

    if output_json is not None:
        payload = {
            "checkpoint": checkpoint,
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else "",
            "policy_mode": policy_mode,
            "suite": suite,
            "task_id": task_id,
            "device": str(device),
            "system": _cpu_summary(),
            "results": [asdict(result) for result in results],
        }
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        typer.echo(f"\nWrote {output_json}")


if __name__ == "__main__":
    app()
