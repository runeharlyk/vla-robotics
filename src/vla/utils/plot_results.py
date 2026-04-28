"""Generate comparison plots from eval result JSON files.

Reads all ``*.json`` files produced by ``scripts/evaluate.py`` from a results
directory, groups them by suite, and generates grouped bar charts comparing
per-task success rates across different training methods (SFT, Sparse RL, etc.).

Usage::

    uv run python -m vla.utils.plot_results --results-dir results/evals --suite spatial
    uv run python -m vla.utils.plot_results --results-dir results/evals \\
        --suite spatial --output assets/libero_spatial_comparison.png
    uv run python -m vla.utils.plot_results --results-dir results/evals \\
        --suite spatial --filter-training-job-id 28161033
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "assets"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SEEDING_FIX_COMMIT = "ef79c3b5dce7212faf1b7c5b3b8bc81ff3d0cf16"


_METHOD_LABELS: dict[str, str] = {
    "sft": "SFT",
    "sparse_rl": "Sparse RL",
    "srpo": "SRPO",
}


def _label_for_method(record: dict[str, Any]) -> str:
    """Build a human-readable label from the eval record."""
    method = record.get("training_method", "sft") or "sft"
    base = _METHOD_LABELS.get(method, method.replace("_", " ").title())

    # For RL methods, extract the training task from the save dir.
    if method in ("sparse_rl", "srpo"):
        save_dir = record.get("training_save_dir", "")
        match = re.search(r"task_(\d+)", save_dir)
        if match:
            base = f"{base} (Task {match.group(1)})"

    return base


def _job_ids_from_name(name: str) -> list[int]:
    """Extract scheduler-style numeric ids embedded as underscore-delimited tokens."""
    ids: list[int] = []
    for match in re.finditer(r"(?:^|_)(\d{6,})(?=_|$)", name):
        try:
            ids.append(int(match.group(1)))
        except ValueError:
            continue
    return ids


def _extract_eval_run_id(record: dict[str, Any]) -> int | None:
    """Extract the evaluation run id from an eval record name."""
    name = str(record.get("eval_name") or record.get("wandb_run_name") or record.get("_source", "")).strip()
    name_job_ids = _job_ids_from_name(Path(name).stem)
    if not name_job_ids:
        return None
    return name_job_ids[-1]


def _extract_training_run_id(record: dict[str, Any]) -> int | None:
    """Extract the training run id used for filtering an eval record.

    RL eval names are generated as ``eval_rl_..._<training_job_id>_<eval_job_id>``.
    """
    name = str(record.get("eval_name") or record.get("wandb_run_name") or record.get("_source", "")).strip()
    name = Path(name).stem
    name_job_ids = _job_ids_from_name(name)
    if len(name_job_ids) >= 2:
        return name_job_ids[-2]
    if len(name_job_ids) == 1 and (record.get("training_method", "") or "").lower() != "sft":
        return name_job_ids[0]

    save_dir = str(record.get("training_save_dir", "")).strip()
    save_dir_job_ids = _job_ids_from_name(save_dir.replace("/", "_").replace("\\", "_"))
    if save_dir_job_ids:
        return save_dir_job_ids[-1]

    if name_job_ids:
        return name_job_ids[-1]

    return None


def _extract_filter_run_id(record: dict[str, Any]) -> int | None:
    """Backward-compatible alias for training-run filtering."""
    return _extract_training_run_id(record)


def _commit_contains_ancestor(commit: str, ancestor: str) -> bool:
    """Return whether ``ancestor`` is reachable from ``commit``."""
    if not commit or not ancestor:
        return False

    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, commit],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def load_eval_records(
    results_dir: Path,
    suite: str | None = None,
    *,
    min_eval_run_id: int | None = None,
    min_training_run_id: int | None = None,
    min_filter_run_id: int | None = None,
    required_eval_ancestor: str | None = None,
) -> list[dict[str, Any]]:
    """Load all eval JSON files, optionally filtering by suite."""
    if min_training_run_id is None:
        min_training_run_id = min_filter_run_id

    records: list[dict[str, Any]] = []
    for json_path in sorted(results_dir.glob("*.json")):
        try:
            record = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Skipping invalid JSON: %s", json_path)
            continue

        if record.get("record_type") != "evaluation":
            continue
        if suite and record.get("suite") != suite:
            continue
        if not record.get("task_metrics"):
            continue
        record["_source"] = json_path.name
        if min_eval_run_id is not None:
            eval_run_id = _extract_eval_run_id(record)
            if eval_run_id is None or eval_run_id <= min_eval_run_id:
                continue
        if min_training_run_id is not None:
            training_run_id = _extract_training_run_id(record)
            if training_run_id is None or training_run_id <= min_training_run_id:
                continue
        if required_eval_ancestor:
            eval_commit = str(record.get("git_commit", ""))
            if not _commit_contains_ancestor(eval_commit, required_eval_ancestor):
                continue

        records.append(record)

    return records


# Canonical display order for training methods.
_METHOD_ORDER = ["sft", "sparse_rl", "srpo"]


def select_best_per_method(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the best eval (highest success_rate) per training_method."""
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        method = record.get("training_method", "sft") or "sft"
        prev = best.get(method)
        if prev is None or record.get("success_rate", 0) > prev.get("success_rate", 0):
            best[method] = record

    # Sort by canonical order, unknown methods go last.
    def _sort_key(method: str) -> int:
        try:
            return _METHOD_ORDER.index(method)
        except ValueError:
            return len(_METHOD_ORDER)

    return [best[m] for m in sorted(best, key=_sort_key)]


def build_comparison_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a long-form DataFrame from eval records for seaborn plotting."""
    rows: list[dict[str, Any]] = []
    for record in records:
        label = _label_for_method(record)
        for task in record["task_metrics"]:
            rows.append(
                {
                    "Task": f"Task {task['task_id']}",
                    "Method": label,
                    "Success Rate": task["success_rate"] * 100,
                }
            )
        # Add an "Overall" entry.
        rows.append(
            {
                "Task": "Overall",
                "Method": label,
                "Success Rate": record.get("success_rate", 0) * 100,
            }
        )
    return pd.DataFrame(rows)


def plot_comparison(
    df: pd.DataFrame,
    suite: str,
    output: Path,
) -> Path:
    """Generate a grouped bar chart comparing methods across tasks."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    palette = ["#4878A8", "#C0504D", "#6DA86D"]  # Steel blue, muted red, muted green

    fig, ax = plt.subplots(figsize=(14, 5.5))

    sns.barplot(
        data=df,
        x="Task",
        y="Success Rate",
        hue="Method",
        palette=palette[: df["Method"].nunique()],
        ax=ax,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_title(
        "LIBERO Evaluation Comparison",
        fontsize=14,
        fontweight="bold",
        pad=30,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=30)

    # Value labels on each bar.
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=7, padding=2)

    ax.legend(
        title="",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=df["Method"].nunique(),
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot to %s", output)
    return output


def main(
    results_dir: Path = typer.Option("results/evals", "--results-dir", "-r", help="Directory with eval JSON files"),  # noqa: B008
    suite: str = typer.Option("spatial", "--suite", "-s", help="Filter by suite name"),  # noqa: B008
    filter_job_id: int | None = typer.Option(
        None,
        "--filter-job-id",
        help="Only include evals whose evaluation job id is newer than this cutoff.",
    ),
    filter_training_job_id: int | None = typer.Option(
        None,
        "--filter-training-job-id",
        help="Only include evals whose training job id is newer than this cutoff.",
    ),
    require_eval_commit: str | None = typer.Option(
        SEEDING_FIX_COMMIT,
        "--require-eval-commit",
        help="Only include evals whose recorded git_commit contains this ancestor commit.",
    ),
    output: Path = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Output PNG path (default: assets/libero_<suite>_comparison.png)"
    ),
) -> None:
    """Generate comparison plots from eval result JSON files."""
    results_path = Path(results_dir)
    if not results_path.is_absolute():
        # Resolve relative to project root
        results_path = PROJECT_ROOT / results_path

    records = load_eval_records(
        results_path,
        suite=suite,
        min_eval_run_id=filter_job_id,
        min_training_run_id=filter_training_job_id,
        required_eval_ancestor=require_eval_commit,
    )
    if not records:
        logger.error("No eval records found in %s for suite=%s", results_path, suite)
        raise typer.Exit(1)

    logger.info("Found %d eval records for suite=%s", len(records), suite)
    if filter_job_id is not None:
        logger.info("Applied eval-run cutoff: keeping only run ids > %d", filter_job_id)
    if filter_training_job_id is not None:
        logger.info("Applied training-run cutoff: keeping only run ids > %d", filter_training_job_id)
    if require_eval_commit:
        logger.info("Applied eval commit filter: keeping commits containing %s", require_eval_commit)
    for r in records:
        logger.info("  %s — %s (%.1f%%)", r["_source"], r.get("training_method", "unknown"), r["success_rate"] * 100)

    # Keep only the best eval per training method.
    records = select_best_per_method(records)
    logger.info("Selected %d best records (one per method)", len(records))

    df = build_comparison_dataframe(records)

    if output is None:
        output = ASSETS_DIR / f"libero_{suite}_comparison.png"

    plot_comparison(df, suite, output)


if __name__ == "__main__":
    typer.run(main)
