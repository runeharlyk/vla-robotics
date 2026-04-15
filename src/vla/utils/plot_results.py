"""Generate comparison plots from eval result JSON files.

Reads all ``*.json`` files produced by ``scripts/evaluate.py`` from a results
directory, groups them by suite, and generates grouped bar charts comparing
per-task success rates across different training methods (SFT, Sparse RL, etc.).

Usage::

    uv run python -m vla.utils.plot_results --results-dir results/evals --suite spatial
    uv run python -m vla.utils.plot_results --results-dir results/evals \\
        --suite spatial --output assets/libero_spatial_comparison.png
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "assets"


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


def load_eval_records(results_dir: Path, suite: str | None = None) -> list[dict[str, Any]]:
    """Load all eval JSON files, optionally filtering by suite."""
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
    output: Path = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Output PNG path (default: assets/libero_<suite>_comparison.png)"
    ),
) -> None:
    """Generate comparison plots from eval result JSON files."""
    results_path = Path(results_dir)
    if not results_path.is_absolute():
        # Resolve relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        results_path = project_root / results_path

    records = load_eval_records(results_path, suite=suite)
    if not records:
        logger.error("No eval records found in %s for suite=%s", results_path, suite)
        raise typer.Exit(1)

    logger.info("Found %d eval records for suite=%s", len(records), suite)
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
