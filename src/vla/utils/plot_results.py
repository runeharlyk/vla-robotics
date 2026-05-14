"""Generate comparison plots and tables from eval result JSON files.

Reads all ``*.json`` files produced by ``scripts/evaluate.py`` from a
results directory and exposes four figure / table commands:

* ``comparison`` -- single-suite grouped bar chart (the original
  per-suite SFT vs RL vs SRPO view).
* ``headline`` -- four-panel grouped bars over the SFT, RL, WiSE-FT
  alpha\\*, and distilled anchor checkpoints, one panel per LIBERO suite.
  This is the thesis headline figure.
* ``alpha-sweep`` -- line plot of suite-avg success rate vs WiSE-FT
  alpha for the Phase-3 alpha sweep evals.
* ``robustness-table`` -- markdown / CSV table from LIBERO-Plus
  per-perturbation evals.

Examples::

    # Old-style per-suite comparison (no change in behaviour):
    uv run python -m vla.utils.plot_results comparison \\
        --results-dir results/evals --suite spatial

    # New: four-panel thesis figure across SFT/RL/WiSE-FT/distill anchors.
    uv run python -m vla.utils.plot_results headline \\
        --results-dir results/evals \\
        --sft-source eval_sft_spatial_l40s_28242119.json \\
        --rl-source eval_p1a_v8a_*.json \\
        --wiseft-source eval_wiseft_alpha050_*.json \\
        --distill-source eval_distill_*.json

    # Alpha sweep figure (one line per suite, x = alpha).
    uv run python -m vla.utils.plot_results alpha-sweep \\
        --results-dir results/evals --pattern 'eval_p3b_spatial_alpha_*.json'

    # Robustness table (one row per perturbation, one column per anchor).
    uv run python -m vla.utils.plot_results robustness-table \\
        --results-dir results/evals --pattern 'eval_p5b_libero_plus_*.json'
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


app = typer.Typer(add_completion=False, help=__doc__)


@app.command(name="comparison")
def comparison_cmd(
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
    """Generate the per-suite SFT vs RL grouped-bar comparison."""
    results_path = Path(results_dir)
    if not results_path.is_absolute():
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
        logger.info("  %s -- %s (%.1f%%)", r["_source"], r.get("training_method", "unknown"), r["success_rate"] * 100)

    records = select_best_per_method(records)
    logger.info("Selected %d best records (one per method)", len(records))

    df = build_comparison_dataframe(records)

    if output is None:
        output = ASSETS_DIR / f"libero_{suite}_comparison.png"

    plot_comparison(df, suite, output)


SUITE_ORDER = ["spatial", "object", "goal", "long"]
ANCHOR_ORDER = ["SFT", "RL", "WiSE-FT", "Distill"]
ANCHOR_COLORS = {
    "SFT": "#7F7F7F",
    "RL": "#4878A8",
    "WiSE-FT": "#6DA86D",
    "Distill": "#C0504D",
}


def _load_single(results_dir: Path, pattern: str) -> dict[str, Any] | None:
    """Load the single eval JSON best matching ``pattern`` (most recent wins on ties)."""
    matches = sorted(results_dir.glob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _records_by_suite(results_dir: Path, pattern: str) -> dict[str, dict[str, Any]]:
    """Return ``{suite: record}`` for the eval JSONs matching ``pattern``.

    When several JSONs share a suite (e.g. multiple cross-suite jobs over
    the same anchor), the most recent ``last-modified`` wins.
    """
    out: dict[str, tuple[float, dict[str, Any]]] = {}
    for path in results_dir.glob(pattern):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        suite = str(record.get("suite", ""))
        if suite not in SUITE_ORDER:
            continue
        mtime = path.stat().st_mtime
        if suite not in out or mtime > out[suite][0]:
            record["_source"] = path.name
            out[suite] = (mtime, record)
    return {suite: pair[1] for suite, pair in out.items()}


def build_headline_dataframe(records_by_anchor: dict[str, dict[str, dict[str, Any]]]) -> pd.DataFrame:
    """Build a long-form DataFrame for the four-panel headline figure.

    ``records_by_anchor[anchor][suite] = eval_record``.
    """
    rows: list[dict[str, Any]] = []
    for anchor, by_suite in records_by_anchor.items():
        for suite, record in by_suite.items():
            for task in record.get("task_metrics", []):
                rows.append(
                    {
                        "Suite": suite,
                        "Task": f"Task {task['task_id']}",
                        "Anchor": anchor,
                        "Success Rate": float(task["success_rate"]) * 100.0,
                    }
                )
            rows.append(
                {
                    "Suite": suite,
                    "Task": "Overall",
                    "Anchor": anchor,
                    "Success Rate": float(record.get("success_rate", 0.0)) * 100.0,
                }
            )
    return pd.DataFrame(rows)


def plot_headline_four_panel(df: pd.DataFrame, output: Path) -> Path:
    """Render the four-panel SFT / RL / WiSE-FT / Distill grouped-bar figure."""
    sns.set_theme(style="whitegrid", font_scale=0.95)
    suites_present = [s for s in SUITE_ORDER if s in set(df["Suite"])]
    if not suites_present:
        raise ValueError("No suite data to plot; df is empty after filtering")

    fig, axes = plt.subplots(
        nrows=len(suites_present),
        ncols=1,
        figsize=(15, 4.0 * len(suites_present)),
        sharey=True,
    )
    if len(suites_present) == 1:
        axes = [axes]

    anchors_present = [a for a in ANCHOR_ORDER if a in set(df["Anchor"])]
    palette = [ANCHOR_COLORS[a] for a in anchors_present]

    for ax, suite in zip(axes, suites_present, strict=True):
        sub = df[df["Suite"] == suite].copy()
        task_order = [t for t in (f"Task {i}" for i in range(10)) if t in set(sub["Task"])]
        task_order.append("Overall")
        sns.barplot(
            data=sub,
            x="Task",
            y="Success Rate",
            hue="Anchor",
            order=task_order,
            hue_order=anchors_present,
            palette=palette,
            ax=ax,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_title(f"LIBERO {suite.capitalize()}", fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Success Rate (%)" if suite == suites_present[0] else "")
        ax.set_ylim(0, 105)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(
            title="",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=len(anchors_present),
            frameon=False,
            fontsize=9,
        )
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", fontsize=6, padding=1)

    fig.suptitle("LIBERO-4 success rate across thesis anchor checkpoints", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved headline figure to %s", output)
    return output


@app.command(name="headline")
def headline_cmd(
    results_dir: Path = typer.Option("results/evals", "--results-dir", "-r"),  # noqa: B008
    sft_pattern: str = typer.Option("eval_sft_spatial_*.json", "--sft-pattern"),
    rl_pattern: str = typer.Option("eval_p5a_cross_suite_*.json", "--rl-pattern"),
    wiseft_pattern: str | None = typer.Option(
        None,
        "--wiseft-pattern",
        help="Glob over eval JSONs for the WiSE-FT alpha* anchor (one per suite).",
    ),
    distill_pattern: str | None = typer.Option(
        None,
        "--distill-pattern",
        help="Glob over eval JSONs for the distilled student anchor (one per suite).",
    ),
    output: Path = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Output PNG path. Default: assets/thesis_headline_four_panel.png"
    ),
) -> None:
    """Render the thesis four-panel headline figure (SFT vs RL vs WiSE-FT vs distill)."""
    results_path = Path(results_dir)
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path

    records_by_anchor: dict[str, dict[str, dict[str, Any]]] = {}
    for anchor, pattern in [
        ("SFT", sft_pattern),
        ("RL", rl_pattern),
        ("WiSE-FT", wiseft_pattern),
        ("Distill", distill_pattern),
    ]:
        if pattern is None:
            logger.info("Skipping anchor %s (no pattern provided)", anchor)
            continue
        records_by_anchor[anchor] = _records_by_suite(results_path, pattern)
        if not records_by_anchor[anchor]:
            logger.warning("No eval JSONs matched %s for anchor %s", pattern, anchor)

    if not records_by_anchor:
        logger.error("No anchor records loaded; nothing to plot.")
        raise typer.Exit(1)

    df = build_headline_dataframe(records_by_anchor)
    if df.empty:
        logger.error("All matching eval records had empty task_metrics; nothing to plot.")
        raise typer.Exit(1)

    if output is None:
        output = ASSETS_DIR / "thesis_headline_four_panel.png"
    plot_headline_four_panel(df, output)


def build_alpha_sweep_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Extract (alpha, suite, success_rate) rows from alpha-sweep eval JSONs."""
    rows: list[dict[str, Any]] = []
    for record in records:
        alpha = record.get("wise_ft_alpha")
        if alpha is None or alpha == "":
            continue
        try:
            alpha_f = float(alpha)
        except (TypeError, ValueError):
            continue
        suite = str(record.get("suite", "unknown"))
        rows.append(
            {
                "alpha": alpha_f,
                "Suite": suite,
                "Success Rate": float(record.get("success_rate", 0.0)) * 100.0,
                "Source": record.get("_source", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["Suite", "alpha"])


def plot_alpha_sweep_line(df: pd.DataFrame, output: Path) -> Path:
    """Line plot of suite-avg success rate vs alpha, one line per suite."""
    if df.empty:
        raise ValueError("Alpha-sweep DataFrame is empty; check --pattern matches anything")

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(
        data=df,
        x="alpha",
        y="Success Rate",
        hue="Suite",
        marker="o",
        ax=ax,
    )
    ax.set_title("WiSE-FT alpha sweep -- suite-avg success rate", fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$\alpha$ (0 = SFT, 1 = RL)")
    ax.set_ylabel("Suite-avg Success Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(sorted(df["alpha"].unique()))
    ax.legend(title="Suite", loc="lower right", frameon=False)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved alpha-sweep figure to %s", output)
    return output


@app.command(name="alpha-sweep")
def alpha_sweep_cmd(
    results_dir: Path = typer.Option("results/evals", "--results-dir", "-r"),  # noqa: B008
    pattern: str = typer.Option("eval_p3b_*alpha_*.json", "--pattern", "-p"),
    output: Path = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Default: assets/thesis_alpha_sweep.png"
    ),
) -> None:
    """Render the WiSE-FT alpha-sweep line plot from Phase-3 evals."""
    results_path = Path(results_dir)
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path

    records: list[dict[str, Any]] = []
    for path in sorted(results_path.glob(pattern)):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Skipping invalid JSON: %s", path)
            continue
        record["_source"] = path.name
        records.append(record)

    if not records:
        logger.error("No alpha-sweep records matched %s in %s", pattern, results_path)
        raise typer.Exit(1)

    df = build_alpha_sweep_dataframe(records)
    if output is None:
        output = ASSETS_DIR / "thesis_alpha_sweep.png"
    plot_alpha_sweep_line(df, output)


def build_robustness_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Pivot LIBERO-Plus per-suite evals into an anchor x perturbation table.

    The W&B name convention is ``eval_p5b_libero_plus_<perturbation>_<jobid>_<suite>``,
    so we parse the perturbation label from the run name. The anchor name
    must be set via ``--label <anchor>`` when submitting the job (or via
    the env var ``ANCHOR_LABEL`` in the LSF script).
    """
    rows: list[dict[str, Any]] = []
    for record in records:
        name = str(record.get("wandb_run_name") or record.get("_source", ""))
        name = Path(name).stem
        anchor = str(record.get("anchor_label", "unknown"))
        perturbation = "unknown"
        if "libero_plus_" in name:
            tail = name.split("libero_plus_", 1)[1]
            perturbation = tail.split("_", 1)[0]
        suite = str(record.get("suite", "unknown"))
        rows.append(
            {
                "Anchor": anchor,
                "Perturbation": perturbation,
                "Suite": suite,
                "Success Rate": float(record.get("success_rate", 0.0)) * 100.0,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["Anchor", "Perturbation"], as_index=False)["Success Rate"].mean().pivot(
        index="Perturbation", columns="Anchor", values="Success Rate"
    )


@app.command(name="robustness-table")
def robustness_table_cmd(
    results_dir: Path = typer.Option("results/evals", "--results-dir", "-r"),  # noqa: B008
    pattern: str = typer.Option("eval_p5b_libero_plus_*.json", "--pattern", "-p"),
    output_csv: Path = typer.Option(  # noqa: B008
        None, "--output-csv", help="Default: assets/thesis_libero_plus_table.csv"
    ),
    output_md: Path = typer.Option(  # noqa: B008
        None, "--output-md", help="Default: assets/thesis_libero_plus_table.md"
    ),
) -> None:
    """Print and write a LIBERO-Plus robustness table (perturbation x anchor)."""
    results_path = Path(results_dir)
    if not results_path.is_absolute():
        results_path = PROJECT_ROOT / results_path

    records: list[dict[str, Any]] = []
    for path in sorted(results_path.glob(pattern)):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Skipping invalid JSON: %s", path)
            continue
        record["_source"] = path.name
        records.append(record)

    if not records:
        logger.error("No LIBERO-Plus records matched %s in %s", pattern, results_path)
        raise typer.Exit(1)

    table = build_robustness_table(records)
    if table.empty:
        logger.error("Pivot produced an empty table -- check that records carry 'anchor_label' or are named correctly.")
        raise typer.Exit(1)

    if output_csv is None:
        output_csv = ASSETS_DIR / "thesis_libero_plus_table.csv"
    if output_md is None:
        output_md = ASSETS_DIR / "thesis_libero_plus_table.md"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, float_format="%.1f")
    output_md.write_text(table.to_markdown(floatfmt=".1f"), encoding="utf-8")
    logger.info("Wrote robustness table to %s and %s", output_csv, output_md)
    print(table.to_string(float_format=lambda v: f"{v:6.2f}"))


if __name__ == "__main__":
    app()
