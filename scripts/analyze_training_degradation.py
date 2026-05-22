"""Analyze training health and hyperparameter effects from local logs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import median
import sys
from typing import Any

import typer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vla.analysis import (
    annotate_cohort_health,
    build_run_summaries,
    cohort_label,
    filter_summaries,
    group_summaries_by_cohort,
    load_training_runs,
    rank_run_summaries,
    summarize_hyperparameter_effects,
)

app = typer.Typer(add_completion=False)


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return "-"
    return str(value)


def _print_section(title: str) -> None:
    typer.echo("")
    typer.echo(title)
    typer.echo("-" * len(title))


def _valid(values: list[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]


def _print_run_rows(rows: list[dict[str, Any]], *, top_n: int) -> None:
    for row in rows[:top_n]:
        typer.echo(
            f"{row['name']}: final={_fmt(row.get('final_eval'))} "
            f"best={_fmt(row.get('best_eval'))} drop={_fmt(row.get('success_drop'))} "
            f"healthy={bool(row.get('healthy'))} tasks={row.get('num_tasks')} "
            f"runtime_h={_fmt(row.get('runtime_hours'))} "
            f"h_per_iter={_fmt(row.get('hours_per_iteration'))} "
            f"lr={_fmt(row.get('lr'), 6)} sft_kl={_fmt(row.get('sft_kl_coeff'), 4)} "
            f"demos={row.get('include_demos_in_update')} replay={row.get('success_replay_total_size')}"
        )


def _print_hparam_effects(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["cohort_id"], {}).setdefault(row["hyperparameter"], []).append(row)

    for cohort_id, by_hparam in grouped.items():
        sample = next(iter(next(iter(by_hparam.values()))))
        typer.echo(sample["cohort_label"])
        for hparam, values in sorted(by_hparam.items()):
            typer.echo(f"  {hparam}:")
            ordered = sorted(
                values,
                key=lambda item: (
                    item["mean_final_eval"] if item["mean_final_eval"] is not None else -1.0,
                    -(item["mean_success_drop"] if item["mean_success_drop"] is not None else 0.0),
                ),
                reverse=True,
            )
            for item in ordered:
                typer.echo(
                    f"    {item['value']}: n={item['count']} "
                    f"final={_fmt(item['mean_final_eval'])} "
                    f"best={_fmt(item['mean_best_eval'])} "
                    f"drop={_fmt(item['mean_success_drop'])} "
                    f"healthy_rate={_fmt(item['healthy_rate'])}"
                )
        typer.echo("")


@app.command()
def main(
    results_root: Path = typer.Option(Path("results"), exists=True, file_okay=False, dir_okay=True),
    top_n: int = typer.Option(10, help="Number of top runs to print per section."),
    cohort_min_runs: int = typer.Option(2, help="Minimum runs in a cohort before comparing hyperparameters."),
    method: str | None = typer.Option(None, help="Optional method filter: sparse_rl, srpo, sft."),
    update_method: str | None = typer.Option(None, help="Optional update-method filter: fpo, awr, ppo."),
    suite: str | None = typer.Option(None, help="Optional suite filter."),
    simulator: str | None = typer.Option(None, help="Optional simulator filter: libero, maniskill."),
) -> None:
    runs = load_training_runs(results_root)
    summaries = annotate_cohort_health(build_run_summaries(runs))
    summaries = filter_summaries(
        summaries,
        method=method,
        update_method=update_method,
        suite=suite,
        simulator=simulator,
    )

    grouped = group_summaries_by_cohort(summaries)
    major_cohorts = {key: value for key, value in grouped.items() if len(value) >= cohort_min_runs}
    hyperparameter_rows = [
        row for row in summarize_hyperparameter_effects(summaries) if len(grouped.get(row["cohort_id"], [])) >= cohort_min_runs
    ]

    typer.echo("Training Health Analysis")
    typer.echo("=======================")
    typer.echo(f"Runs loaded: {len(summaries)}")
    typer.echo(f"Cohorts: {len(grouped)}")
    typer.echo(f"Comparable cohorts (n>={cohort_min_runs}): {len(major_cohorts)}")
    typer.echo(f"Methods: {dict(Counter(summary['method'] for summary in summaries))}")
    typer.echo(f"Statuses: {dict(Counter(summary['status'] for summary in summaries))}")

    healthy_rows = rank_run_summaries(summaries, healthy_only=True)
    multitask_healthy_rows = rank_run_summaries(summaries, multitask_only=True, healthy_only=True)
    degrading_rows = sorted(
        [summary for summary in summaries if isinstance(summary.get("success_drop"), (int, float))],
        key=lambda summary: float(summary["success_drop"]),
        reverse=True,
    )

    _print_section("Healthiest Runs")
    if healthy_rows:
        _print_run_rows(healthy_rows, top_n=top_n)
    else:
        typer.echo("No healthy runs found under the current filters.")

    _print_section("Healthiest Multitask Runs")
    if multitask_healthy_rows:
        _print_run_rows(multitask_healthy_rows, top_n=top_n)
    else:
        typer.echo("No healthy multitask runs found under the current filters.")

    _print_section("Largest Drops")
    _print_run_rows(degrading_rows, top_n=top_n)

    _print_section("Healthy vs Unhealthy Signals")
    healthy = [summary for summary in summaries if bool(summary.get("healthy"))]
    unhealthy = [summary for summary in summaries if not bool(summary.get("healthy"))]
    if healthy and unhealthy:
        for key in (
            "final_eval",
            "success_drop",
            "mean_raw_kl",
            "mean_clip_frac",
            "positive_loss_fraction",
            "mean_rollout_success_rate",
        ):
            healthy_values = _valid([summary.get(key) for summary in healthy])
            unhealthy_values = _valid([summary.get(key) for summary in unhealthy])
            typer.echo(
                f"{key}: healthy_median={_fmt(median(healthy_values) if healthy_values else None)} "
                f"unhealthy_median={_fmt(median(unhealthy_values) if unhealthy_values else None)}"
            )
    else:
        typer.echo("Not enough separation between healthy and unhealthy runs for a log-signal comparison.")

    _print_section("Major Cohorts")
    for cohort_id, cohort_summaries in sorted(major_cohorts.items(), key=lambda item: len(item[1]), reverse=True):
        ranked = rank_run_summaries(cohort_summaries)
        healthy_count = sum(bool(summary.get("healthy")) for summary in cohort_summaries)
        stable_count = sum(bool(summary.get("stable")) for summary in cohort_summaries)
        typer.echo(f"{cohort_label(cohort_summaries[0])}")
        typer.echo(
            f"  runs={len(cohort_summaries)} healthy={healthy_count} stable={stable_count} "
            f"best_final={_fmt(ranked[0].get('final_eval') if ranked else None)}"
        )
        if ranked:
            typer.echo(
                f"  top_run={ranked[0]['name']} "
                f"final={_fmt(ranked[0].get('final_eval'))} "
                f"drop={_fmt(ranked[0].get('success_drop'))}"
            )
        worse = [summary for summary in ranked if summary.get("status") == "degrading"]
        if worse:
            typer.echo(
                f"  degrading_run={worse[0]['name']} "
                f"final={_fmt(worse[0].get('final_eval'))} "
                f"drop={_fmt(worse[0].get('success_drop'))}"
            )

    _print_section("Hyperparameter Effects By Cohort")
    if hyperparameter_rows:
        _print_hparam_effects(hyperparameter_rows)
    else:
        typer.echo("No cohorts with enough variation to compare hyperparameters.")


if __name__ == "__main__":
    app()
