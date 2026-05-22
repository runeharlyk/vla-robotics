"""Training log analysis built on top of local result artifacts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from statistics import mean, median
from typing import Any

DEFAULT_HPARAM_KEYS = (
    "lr",
    "clip_epsilon",
    "clip_epsilon_high",
    "ppo_epochs",
    "trajs_per_task_per_iter",
    "eval_every",
    "eval_episodes",
    "sft_kl_coeff",
    "include_demos_in_update",
    "success_replay_total_size",
    "success_replay_buffer_size",
    "success_replay_max_ratio",
    "fpo_negative_adv_scale",
    "kl_coeff",
)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _normalize_save_dir(path: str | None) -> str:
    if not path:
        return ""
    pure = PurePosixPath(path)
    if pure.name in {"best", "last"}:
        pure = pure.parent
    return pure.as_posix()


def _normalize_method(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_update_method(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_suite(value: Any) -> str:
    return str(value or "").strip().lower()


def _task_ids(record: dict[str, Any], config: dict[str, Any]) -> tuple[str, ...]:
    raw = record.get("task_ids")
    if isinstance(raw, list) and raw:
        return tuple(str(v) for v in raw)
    raw = config.get("tasks")
    if isinstance(raw, list) and raw:
        return tuple(str(v) for v in raw)
    specs = record.get("task_specs")
    if isinstance(specs, list):
        resolved = [str(spec.get("task_id")) for spec in specs if isinstance(spec, dict) and spec.get("task_id")]
        if resolved:
            return tuple(resolved)
    task_id = config.get("task_id")
    if task_id is not None:
        return (str(task_id),)
    return ()


def _num_tasks(record: dict[str, Any], config: dict[str, Any], task_ids: tuple[str, ...]) -> int:
    raw = record.get("num_tasks", config.get("num_tasks"))
    if isinstance(raw, int) and raw > 0:
        return raw
    return max(len(task_ids), 1)


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _stringify_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_stringify_value(item) for item in value) + "]"
    return str(value)


@dataclass
class TrainingRun:
    name: str
    record_path: Path
    curve_path: Path | None
    record: dict[str, Any]
    config: dict[str, Any]
    curve_rows: list[dict[str, Any]]
    method: str
    update_method: str
    suite: str
    save_dir: str
    task_ids: tuple[str, ...]
    num_tasks: int
    metric_prefix: str


def _infer_metric_prefix(record: dict[str, Any], config: dict[str, Any], curve_rows: list[dict[str, Any]]) -> str:
    if curve_rows:
        counts: dict[str, int] = defaultdict(int)
        for row in curve_rows[:10]:
            for key in row:
                if key.startswith("_") or "/" not in key:
                    continue
                prefix = key.split("/", 1)[0].strip().lower()
                if prefix:
                    counts[prefix] += 1
        if counts:
            return max(counts, key=counts.get)

    method = _normalize_method(record.get("method", config.get("mode", config.get("method"))))
    return method or "training"


def load_training_runs(results_root: Path | str = Path("results")) -> list[TrainingRun]:
    results_path = Path(results_root)
    training_dir = results_path / "training"
    curves_dir = results_path / "training_curves"

    runs: list[TrainingRun] = []
    for record_path in sorted(training_dir.glob("*.json")):
        record = _load_json(record_path)
        if not record:
            continue
        config = record.get("config", {})

        curve_path: Path | None = None
        metrics_jsonl = record.get("metrics_jsonl")
        if isinstance(metrics_jsonl, str) and metrics_jsonl:
            candidate = Path(metrics_jsonl)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if candidate.exists():
                curve_path = candidate
        if curve_path is None:
            candidate = curves_dir / f"{record_path.stem}.jsonl"
            if candidate.exists():
                curve_path = candidate

        curve_rows = _load_jsonl(curve_path) if curve_path is not None else []
        task_ids = _task_ids(record, config)

        runs.append(
            TrainingRun(
                name=str(record.get("wandb_run_name") or record_path.stem),
                record_path=record_path,
                curve_path=curve_path,
                record=record,
                config=config,
                curve_rows=curve_rows,
                method=_normalize_method(record.get("method", config.get("mode", config.get("method")))),
                update_method=_normalize_update_method(config.get("update_method")),
                suite=_normalize_suite(record.get("suite", config.get("suite"))),
                save_dir=_normalize_save_dir(str(record.get("save_dir") or config.get("save_dir") or "")),
                task_ids=task_ids,
                num_tasks=_num_tasks(record, config, task_ids),
                metric_prefix=_infer_metric_prefix(record, config, curve_rows),
            )
        )

    return runs


def _candidate_iteration_keys(run: TrainingRun) -> tuple[str, ...]:
    if run.metric_prefix == "sft":
        return ("sft/epoch", "sft/step")
    return (f"{run.metric_prefix}/iteration",)


def _candidate_eval_keys(run: TrainingRun) -> tuple[str, ...]:
    if run.metric_prefix == "sft":
        return ("sft/success_rate", "sft/eval/success_rate")
    return (f"{run.metric_prefix}/eval/success_rate",)


def _candidate_loss_keys(run: TrainingRun) -> tuple[str, ...]:
    if run.metric_prefix == "sft":
        return ("sft/loss",)
    if run.update_method == "awr":
        return (f"{run.metric_prefix}/awr_loss",)
    if run.update_method == "ppo":
        return (f"{run.metric_prefix}/ppo_loss",)
    return (f"{run.metric_prefix}/fpo_loss", f"{run.metric_prefix}/ppo_loss", f"{run.metric_prefix}/awr_loss")


def _row_x_value(run: TrainingRun, row: dict[str, Any]) -> int | None:
    for key in _candidate_iteration_keys(run):
        resolved = _as_int(row.get(key))
        if resolved is not None:
            return resolved
    return _as_int(row.get("_step"))


def metric_series(run: TrainingRun, key: str, *, dedupe_x: bool = False) -> list[tuple[int, float]]:
    series: list[tuple[int, float]] = []
    for row in run.curve_rows:
        x_value = _row_x_value(run, row)
        y_value = _as_float(row.get(key))
        if x_value is None or y_value is None:
            continue
        series.append((x_value, y_value))

    if not dedupe_x:
        return series

    deduped: dict[int, float] = {}
    for x_value, y_value in series:
        deduped[x_value] = y_value
    return [(x_value, deduped[x_value]) for x_value in sorted(deduped)]


def _first_existing_metric_series(run: TrainingRun, keys: tuple[str, ...], *, dedupe_x: bool = False) -> tuple[str | None, list[tuple[int, float]]]:
    for key in keys:
        series = metric_series(run, key, dedupe_x=dedupe_x)
        if series:
            return key, series
    return None, []


def _cohort_id(run: TrainingRun) -> str:
    task_part = ",".join(run.task_ids) if run.task_ids else "none"
    return f"{run.method}|{run.update_method}|{run.suite}|{run.num_tasks}|{task_part}"


def cohort_label(summary: dict[str, Any]) -> str:
    tasks = summary.get("task_ids") or ()
    task_part = ", ".join(tasks) if tasks else "none"
    return (
        f"{summary.get('method', '')}/{summary.get('update_method', '')} "
        f"suite={summary.get('suite', '')} tasks={summary.get('num_tasks', 0)} [{task_part}]"
    )


def summarize_run(
    run: TrainingRun,
    *,
    degrade_threshold: float = 0.05,
    stable_tolerance: float = 0.02,
) -> dict[str, Any]:
    eval_metric_key, eval_series = _first_existing_metric_series(run, _candidate_eval_keys(run), dedupe_x=True)
    loss_metric_key, loss_series = _first_existing_metric_series(run, _candidate_loss_keys(run))

    initial_eval = eval_series[0][1] if eval_series else None
    best_eval = _as_float(run.record.get("best_eval_metric_value"))
    final_eval = _as_float(run.record.get("final_eval_metric_value"))
    best_iteration = _as_int(run.record.get("best_eval_iteration"))
    final_iteration = _as_int(run.record.get("final_eval_iteration"))

    if eval_series:
        if best_eval is None:
            best_iteration, best_eval = max(eval_series, key=lambda item: item[1])
        if final_eval is None:
            final_iteration, final_eval = eval_series[-1]

    success_drop = None
    status = "unknown"
    if best_eval is not None and final_eval is not None:
        success_drop = best_eval - final_eval
        if success_drop >= degrade_threshold:
            status = "degrading"
        elif abs(success_drop) <= stable_tolerance:
            status = "stable"
        elif final_eval > best_eval:
            status = "improving"
        else:
            status = "slightly_worse"

    raw_kl_key = f"{run.metric_prefix}/raw_kl"
    clip_key = f"{run.metric_prefix}/clip_frac"
    skipped_key = f"{run.metric_prefix}/skipped_tasks"
    rollout_success_key = f"{run.metric_prefix}/rollout_successes"

    raw_kl_values = [value for _, value in metric_series(run, raw_kl_key)]
    clip_values = [value for _, value in metric_series(run, clip_key)]
    skipped_values = [value for _, value in metric_series(run, skipped_key)]
    rollout_successes = [value for _, value in metric_series(run, rollout_success_key)]
    loss_values = [value for _, value in loss_series]
    runtime_values = [_as_float(row.get("_runtime")) for row in run.curve_rows]
    runtime_values = [value for value in runtime_values if value is not None]

    trajs_per_task = _as_int(run.config.get("trajs_per_task_per_iter")) or _as_int(run.record.get("trajs_per_task_per_iter"))
    rollout_denom = None
    if trajs_per_task is not None and trajs_per_task > 0 and run.num_tasks > 0:
        rollout_denom = float(trajs_per_task * run.num_tasks)
    rollout_success_rates = [value / rollout_denom for value in rollout_successes] if rollout_denom else []
    runtime_seconds = max(runtime_values) if runtime_values else None
    max_iteration = max((x_value for x_value, _ in metric_series(run, f"{run.metric_prefix}/rollout_successes")), default=None)
    hours_per_iteration = None
    if runtime_seconds is not None and isinstance(max_iteration, int) and max_iteration > 0:
        hours_per_iteration = (runtime_seconds / 3600.0) / max_iteration

    shared_replay = run.config.get("success_replay_total_size")
    if not isinstance(shared_replay, int):
        shared_replay = run.config.get("success_replay_buffer_size")

    summary = {
        "name": run.name,
        "record_path": str(run.record_path),
        "curve_path": str(run.curve_path) if run.curve_path is not None else "",
        "config": run.config,
        "method": run.method,
        "update_method": run.update_method,
        "suite": run.suite,
        "simulator": _normalize_method(run.record.get("simulator", run.config.get("simulator"))),
        "env_id": str(run.record.get("env_id", run.config.get("env_id", ""))),
        "save_dir": run.save_dir,
        "task_ids": run.task_ids,
        "num_tasks": run.num_tasks,
        "cohort_id": _cohort_id(run),
        "metric_prefix": run.metric_prefix,
        "eval_metric_key": eval_metric_key or "",
        "loss_metric_key": loss_metric_key or "",
        "initial_eval": initial_eval,
        "best_eval": best_eval,
        "best_iteration": best_iteration,
        "final_eval": final_eval,
        "final_iteration": final_iteration,
        "mean_eval": _mean_or_none([value for _, value in eval_series]),
        "eval_points": len(eval_series),
        "history_points": len(run.curve_rows),
        "runtime_seconds": runtime_seconds,
        "runtime_hours": (runtime_seconds / 3600.0) if runtime_seconds is not None else None,
        "train_points": len(rollout_successes),
        "max_iteration": max_iteration,
        "hours_per_iteration": hours_per_iteration,
        "success_drop": success_drop,
        "status": status,
        "stable": status in {"stable", "improving"},
        "degrading": status == "degrading",
        "mean_loss": _mean_or_none(loss_values),
        "max_loss": _max_or_none(loss_values),
        "positive_loss_fraction": (
            sum(value > 0 for value in loss_values) / len(loss_values) if loss_values else None
        ),
        "mean_raw_kl": _mean_or_none(raw_kl_values),
        "max_raw_kl": _max_or_none(raw_kl_values),
        "mean_clip_frac": _mean_or_none(clip_values),
        "max_clip_frac": _max_or_none(clip_values),
        "mean_skipped_tasks": _mean_or_none(skipped_values),
        "max_skipped_tasks": _max_or_none(skipped_values),
        "max_rollout_success_rate": _max_or_none(rollout_success_rates),
        "mean_rollout_success_rate": _mean_or_none(rollout_success_rates),
        "final_rollout_success_rate": rollout_success_rates[-1] if rollout_success_rates else None,
        "trajs_per_task_per_iter": trajs_per_task,
        "lr": run.config.get("lr"),
        "clip_epsilon": run.config.get("clip_epsilon"),
        "clip_epsilon_high": run.config.get("clip_epsilon_high"),
        "ppo_epochs": run.config.get("ppo_epochs"),
        "sft_kl_coeff": run.config.get("sft_kl_coeff", 0.0),
        "include_demos_in_update": bool(run.config.get("include_demos_in_update", False)),
        "success_replay_total_size": shared_replay if isinstance(shared_replay, int) else 0,
        "success_replay_max_ratio": run.config.get("success_replay_max_ratio"),
    }
    return summary


def build_run_summaries(
    runs: list[TrainingRun],
    *,
    degrade_threshold: float = 0.05,
    stable_tolerance: float = 0.02,
) -> list[dict[str, Any]]:
    return [
        summarize_run(run, degrade_threshold=degrade_threshold, stable_tolerance=stable_tolerance)
        for run in runs
    ]


def group_summaries_by_cohort(summaries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[str(summary.get("cohort_id", ""))].append(summary)
    return grouped


def annotate_cohort_health(
    summaries: list[dict[str, Any]],
    *,
    stable_tolerance: float = 0.02,
) -> list[dict[str, Any]]:
    grouped = group_summaries_by_cohort(summaries)
    for cohort_summaries in grouped.values():
        if len(cohort_summaries) < 2:
            for summary in cohort_summaries:
                summary["healthy"] = False
            continue

        finals = [summary["final_eval"] for summary in cohort_summaries if isinstance(summary.get("final_eval"), (int, float))]
        if not finals:
            for summary in cohort_summaries:
                summary["healthy"] = False
            continue

        if len(finals) >= 4:
            final_threshold = _quantile([float(value) for value in finals], 0.75)
        else:
            final_threshold = max(float(value) for value in finals) - stable_tolerance

        for summary in cohort_summaries:
            final_eval = summary.get("final_eval")
            drop = summary.get("success_drop")
            stable = bool(summary.get("stable"))
            summary["healthy"] = (
                isinstance(final_eval, (int, float))
                and final_eval >= final_threshold
                and stable
                and (drop is None or drop <= stable_tolerance)
            )

    return summaries


def rank_run_summaries(
    summaries: list[dict[str, Any]],
    *,
    multitask_only: bool = False,
    healthy_only: bool = False,
) -> list[dict[str, Any]]:
    filtered = [
        summary
        for summary in summaries
        if (not multitask_only or int(summary.get("num_tasks", 1)) > 1)
        and (not healthy_only or bool(summary.get("healthy")))
    ]
    return sorted(
        filtered,
        key=lambda summary: (
            bool(summary.get("healthy")),
            _as_float(summary.get("final_eval")) or float("-inf"),
            -(_as_float(summary.get("success_drop")) or 0.0),
            _as_float(summary.get("best_eval")) or float("-inf"),
        ),
        reverse=True,
    )


def build_runs_table(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for summary in summaries:
        row = dict(summary)
        row["task_ids"] = ", ".join(summary.get("task_ids", ()))
        table.append(row)
    return table


def _get_hparam_value(summary: dict[str, Any], key: str) -> Any:
    if key in summary:
        return summary[key]
    config = summary.get("config")
    if isinstance(config, dict):
        return config.get(key)
    return None


def summarize_hyperparameter_effects(
    summaries: list[dict[str, Any]],
    *,
    hyperparameter_keys: tuple[str, ...] = DEFAULT_HPARAM_KEYS,
    min_distinct_values: int = 2,
) -> list[dict[str, Any]]:
    grouped = group_summaries_by_cohort(summaries)
    rows: list[dict[str, Any]] = []

    for cohort_id, cohort_summaries in grouped.items():
        if len(cohort_summaries) < 2:
            continue
        for key in hyperparameter_keys:
            value_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for summary in cohort_summaries:
                value = summary.get(key)
                if value is None:
                    value = summary.get("config", {}).get(key) if isinstance(summary.get("config"), dict) else None
                value_groups[_stringify_value(value)].append(summary)

            if len(value_groups) < min_distinct_values:
                continue

            for value_label, value_runs in value_groups.items():
                finals = [float(summary["final_eval"]) for summary in value_runs if isinstance(summary.get("final_eval"), (int, float))]
                bests = [float(summary["best_eval"]) for summary in value_runs if isinstance(summary.get("best_eval"), (int, float))]
                drops = [float(summary["success_drop"]) for summary in value_runs if isinstance(summary.get("success_drop"), (int, float))]
                rows.append(
                    {
                        "cohort_id": cohort_id,
                        "cohort_label": cohort_label(value_runs[0]),
                        "hyperparameter": key,
                        "value": value_label,
                        "count": len(value_runs),
                        "healthy_rate": sum(bool(summary.get("healthy")) for summary in value_runs) / len(value_runs),
                        "stable_rate": sum(bool(summary.get("stable")) for summary in value_runs) / len(value_runs),
                        "mean_final_eval": _mean_or_none(finals),
                        "mean_best_eval": _mean_or_none(bests),
                        "mean_success_drop": _mean_or_none(drops),
                    }
                )

    return sorted(
        rows,
        key=lambda row: (
            row["cohort_id"],
            row["hyperparameter"],
            -(row["mean_final_eval"] if row["mean_final_eval"] is not None else -1.0),
            -row["count"],
        ),
    )


def filter_summaries(
    summaries: list[dict[str, Any]],
    *,
    method: str | None = None,
    update_method: str | None = None,
    suite: str | None = None,
    simulator: str | None = None,
    num_tasks: int | None = None,
) -> list[dict[str, Any]]:
    filtered = summaries
    if method is not None:
        method = _normalize_method(method)
        filtered = [summary for summary in filtered if _normalize_method(summary.get("method")) == method]
    if update_method is not None:
        update_method = _normalize_update_method(update_method)
        filtered = [
            summary for summary in filtered if _normalize_update_method(summary.get("update_method")) == update_method
        ]
    if suite is not None:
        suite = _normalize_suite(suite)
        filtered = [summary for summary in filtered if _normalize_suite(summary.get("suite")) == suite]
    if simulator is not None:
        simulator = _normalize_method(simulator)
        filtered = [
            summary for summary in filtered if _normalize_method(summary.get("simulator")) == simulator
        ]
    if num_tasks is not None:
        filtered = [summary for summary in filtered if int(summary.get("num_tasks", 0)) == num_tasks]
    return filtered


def build_long_metric_table(
    runs: list[TrainingRun],
    metric_key: str,
    *,
    dedupe_x: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for x_value, y_value in metric_series(run, metric_key, dedupe_x=dedupe_x):
            rows.append(
                {
                    "name": run.name,
                    "cohort_id": _cohort_id(run),
                    "method": run.method,
                    "update_method": run.update_method,
                    "suite": run.suite,
                    "num_tasks": run.num_tasks,
                    "task_ids": ", ".join(run.task_ids),
                    "x": x_value,
                    "metric_key": metric_key,
                    "value": y_value,
                }
            )
    return rows


def _run_order_key(run: TrainingRun) -> tuple[float, str]:
    for field in ("recorded_at", "completed_at"):
        value = run.record.get(field)
        if isinstance(value, str) and value:
            try:
                from datetime import datetime

                return (datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp(), run.name)
            except Exception:
                pass

    timestamps = [_as_float(row.get("_timestamp")) for row in run.curve_rows]
    finite_timestamps = [value for value in timestamps if value is not None]
    if finite_timestamps:
        return (min(finite_timestamps), run.name)

    return (float("inf"), run.name)


def build_rollout_success_percent_table(
    runs: list[TrainingRun],
    *,
    alpha_min: float = 0.12,
    alpha_max: float = 1.0,
) -> list[dict[str, Any]]:
    ordered_runs = sorted(runs, key=_run_order_key)
    total_runs = len(ordered_runs)
    rows: list[dict[str, Any]] = []

    for index, run in enumerate(ordered_runs, start=1):
        trajs_per_task = _as_int(run.config.get("trajs_per_task_per_iter")) or _as_int(run.record.get("trajs_per_task_per_iter"))
        if trajs_per_task is None or trajs_per_task <= 0 or run.num_tasks <= 0:
            continue

        denom = float(trajs_per_task * run.num_tasks)
        metric_key = f"{run.metric_prefix}/rollout_successes"
        for iteration, successes in metric_series(run, metric_key, dedupe_x=True):
            success_pct = 100.0 * successes / denom
            if total_runs <= 1:
                alpha = alpha_max
            else:
                alpha = alpha_min + (alpha_max - alpha_min) * (index - 1) / (total_runs - 1)

            rows.append(
                {
                    "name": run.name,
                    "cohort_id": _cohort_id(run),
                    "method": run.method,
                    "update_method": run.update_method,
                    "suite": run.suite,
                    "num_tasks": run.num_tasks,
                    "task_ids": ", ".join(run.task_ids),
                    "iteration": iteration,
                    "success_pct": success_pct,
                    "run_order": index,
                    "run_order_alpha": alpha,
                    "trajs_per_task_per_iter": trajs_per_task,
                }
            )

    return rows


def to_dataframe(rows: list[dict[str, Any]]):
    import pandas as pd

    return pd.DataFrame(rows)
