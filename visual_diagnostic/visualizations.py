#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
import seaborn as sns

"""
run example for HPC servers (adjust paths as needed):
uv run visual_diagnostic/visualizations.py   
--csv /work3/s234863/visual_cor_suc_s1/visual_noise_rollouts_raw.csv   
--csv /work3/s234863/visual_cor_suc_s3/visual_noise_rollouts_raw.csv   
--csv /work3/s234863/visual_cor_suc_s5/visual_noise_rollouts_raw.csv   
--outdir /zhome/27/3/205343/vla-robotics/visual_diagnostic/images   
--focus-severity 3
"""

# ---------------------------------------------------------------------------
# Global theme
# ---------------------------------------------------------------------------

_PALETTE = "colorblind"


def _apply_theme() -> None:
    """Set a clean, publication-ready theme for all plots."""
    sns.set_theme(
        style="whitegrid",
        font_scale=1.2,
        rc={
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "figure.dpi": 200,
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot visual sensitivity results from one or more CSVs."
    )
    parser.add_argument(
        "--csv",
        action="append",
        required=True,
        help="Path to visual_noise_rollouts_raw.csv (repeatable).",
    )
    parser.add_argument("--outdir", default="visual_sensitivity_plots")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    parser.add_argument(
        "--focus-severity",
        type=int,
        default=None,
        help="Severity to emphasize in task-level plots.",
    )
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_csvs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["source"] = Path(p).parent.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["success"] = df["success"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["severity"] = df["severity"].astype(int)
    df["episode_length"] = df["episode_length"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Baseline helpers
# ---------------------------------------------------------------------------


def clean_baseline(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    clean = df[df["noise_type"] == "clean"]
    if clean.empty:
        return pd.DataFrame(columns=group_cols + ["clean_success"])
    if not group_cols:
        per_source = clean.groupby("source")["success"].mean().reset_index()
        return pd.DataFrame({"clean_success": [per_source["success"].mean()]})

    per_source = (
        clean.groupby(["source"] + group_cols)["success"]
        .mean()
        .reset_index()
    )
    baseline = (
        per_source.groupby(group_cols)["success"]
        .mean()
        .reset_index()
        .rename(columns={"success": "clean_success"})
    )
    return baseline


def choose_focus_severity(df: pd.DataFrame, focus: int | None) -> int | None:
    severities = sorted(df.loc[df["noise_type"] != "clean", "severity"].unique())
    if not severities:
        return None
    if focus in severities:
        return focus
    return severities[len(severities) // 2]


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------


def print_coverage(df: pd.DataFrame) -> None:
    noised = df[df["noise_type"] != "clean"]
    if noised.empty:
        print("No noised rows found.")
        return
    total_tasks = (
        noised[["suite", "task_id"]]
        .drop_duplicates()
        .shape[0]
    )
    counts = (
        noised[["severity", "noise_type", "suite", "task_id"]]
        .drop_duplicates()
        .groupby(["severity", "noise_type"])
        .size()
        .reset_index(name="task_count")
    )
    missing = counts[counts["task_count"] < total_tasks]
    if not missing.empty:
        print("WARNING: Missing task coverage for some (severity, noise_type) pairs:")
        for _, row in missing.iterrows():
            sev = row["severity"]
            nt = row["noise_type"]
            tc = row["task_count"]
            print(f"  severity={sev} noise_type={nt}: {tc}/{total_tasks} tasks")


# ===================================================================
# EXISTING PLOTS (fixed)
# ===================================================================


def plot_success_vs_severity(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    """Line plot: success rate vs severity, one line per noise type."""
    noised = df[df["noise_type"] != "clean"]
    if noised.empty:
        return

    grouped = (
        noised.groupby(["noise_type", "severity"])["success"]
        .agg(["mean", "count"])
        .reset_index()
    )
    grouped["se"] = np.sqrt(
        grouped["mean"] * (1 - grouped["mean"]) / grouped["count"].clip(lower=1)
    )
    grouped["ci_low"] = (grouped["mean"] - 1.96 * grouped["se"]).clip(0, 1)
    grouped["ci_high"] = (grouped["mean"] + 1.96 * grouped["se"]).clip(0, 1)

    clean_mean = clean_baseline(df, []).get("clean_success")
    clean_rate = float(clean_mean.iloc[0]) if clean_mean is not None and not clean_mean.empty else None

    palette = sns.color_palette(_PALETTE, n_colors=grouped["noise_type"].nunique())

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for i, nt in enumerate(sorted(grouped["noise_type"].unique())):
        sub = grouped[grouped["noise_type"] == nt].sort_values("severity")
        ax.plot(sub["severity"], sub["mean"], marker="o", label=nt, color=palette[i])
        ax.fill_between(sub["severity"], sub["ci_low"], sub["ci_high"],
                        alpha=0.15, color=palette[i])

    if clean_rate is not None:
        ax.axhline(clean_rate, color="black", linestyle="--", linewidth=1,
                   label="clean baseline")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Severity")
    ax.set_ylabel("Success rate")
    ax.set_title("Success vs Severity by Noise Type")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(ncol=2, frameon=True, fancybox=True, framealpha=0.8,
              fontsize=10, loc="lower left")
    out_path = outdir / f"success_vs_severity_by_noise.{fmt}"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_suite_drop_heatmaps(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    """Heatmap: success-rate drop from clean baseline, per suite × noise type."""
    noised = df[df["noise_type"] != "clean"]
    if noised.empty:
        return

    clean_suite = clean_baseline(df, ["suite"])
    grouped = (
        noised.groupby(["suite", "noise_type", "severity"])["success"]
        .mean()
        .reset_index()
    )
    merged = grouped.merge(clean_suite, on="suite", how="left")
    merged["drop"] = merged["clean_success"] - merged["success"]

    for sev in sorted(merged["severity"].unique()):
        sub = merged[merged["severity"] == sev]
        pivot = sub.pivot(index="suite", columns="noise_type", values="drop")

        n_rows, n_cols = pivot.shape
        fig_h = max(2.5, n_rows * 0.6 + 1.5)
        fig_w = max(5, n_cols * 1.0 + 2.5)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

        # Diverging colormap so negative drops (improvements) are visible
        abs_max = max(abs(pivot.values[np.isfinite(pivot.values)]).max(), 0.01)
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            vmin=-abs_max,
            vmax=abs_max,
            center=0,
            cmap="RdBu_r",
            cbar_kws={"label": "Drop from clean", "shrink": 0.8},
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )
        ax.set_title(f"Suite Sensitivity (Severity {sev})")
        ax.set_ylabel("Suite")
        ax.set_xlabel("Noise type")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

        out_path = outdir / f"suite_noise_drop_heatmap_s{sev}.{fmt}"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def plot_task_drop_heatmap(df: pd.DataFrame, sev: int, outdir: Path, fmt: str) -> None:
    """Heatmap: per-task success-rate drop at a given severity."""
    noised = df[(df["noise_type"] != "clean") & (df["severity"] == sev)]
    if noised.empty:
        return

    task_cols = ["suite", "task_id", "task_description"]
    clean_task = clean_baseline(df, task_cols)
    grouped = (
        noised.groupby(task_cols + ["noise_type"])["success"]
        .mean()
        .reset_index()
    )
    merged = grouped.merge(clean_task, on=task_cols, how="left")
    merged["drop"] = merged["clean_success"] - merged["success"]

    def _label(row: pd.Series) -> str:
        desc = row["task_description"]
        if len(desc) > 45:
            desc = desc[:42] + "…"
        return f"{row['suite'][0].upper()}{row['task_id']}: {desc}"

    merged["task_label"] = merged.apply(_label, axis=1)

    order = (
        merged.groupby("task_label")["drop"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    pivot = merged.pivot(index="task_label", columns="noise_type", values="drop").loc[order]

    n_tasks = len(pivot)
    n_noise = len(pivot.columns)

    # Dynamic sizing: scale height with number of tasks
    fig_h = max(4, n_tasks * 0.35 + 1.5)
    fig_w = max(7, n_noise * 1.2 + 3.5)
    annot_fontsize = 10 if n_tasks <= 15 else 8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    abs_max = max(abs(pivot.values[np.isfinite(pivot.values)]).max(), 0.01)
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        vmin=-abs_max,
        vmax=abs_max,
        center=0,
        cmap="RdBu_r",
        cbar_kws={"label": "Drop from clean", "shrink": 0.7},
        linewidths=0.5,
        linecolor="white",
        annot_kws={"fontsize": annot_fontsize},
        ax=ax,
    )
    ax.set_title(f"Task Sensitivity by Noise Type (Severity {sev})")
    ax.set_ylabel("Task")
    ax.set_xlabel("Noise type")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    out_path = outdir / f"task_noise_drop_heatmap_s{sev}.{fmt}"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_noise_ranking(df: pd.DataFrame, sev: int, outdir: Path, fmt: str) -> None:
    """Horizontal bar chart ranking noise types by success rate."""
    sub = df[(df["noise_type"] != "clean") & (df["severity"] == sev)]
    if sub.empty:
        return

    rates = (
        sub.groupby("noise_type")["success"]
        .mean()
        .sort_values()
    )
    clean_mean = clean_baseline(df, []).get("clean_success")
    clean_rate = float(clean_mean.iloc[0]) if clean_mean is not None and not clean_mean.empty else None

    palette = sns.color_palette(_PALETTE, n_colors=len(rates))

    fig, ax = plt.subplots(figsize=(7, max(3, len(rates) * 0.5 + 1)),
                           constrained_layout=True)
    rates.plot(kind="barh", color=palette, ax=ax, edgecolor="white", linewidth=0.5)

    if clean_rate is not None:
        ax.axvline(clean_rate, color="black", linestyle="--", linewidth=1,
                   label="clean baseline")
        ax.legend(frameon=True, fancybox=True, framealpha=0.8, fontsize=10)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Success rate")
    ax.set_title(f"Noise Ranking (Severity {sev})")

    # Add value annotations on bars
    for i, (idx, val) in enumerate(rates.items()):
        ax.text(val + 0.02, i, f"{val:.1%}", va="center", fontsize=10)

    out_path = outdir / f"noise_ranking_s{sev}.{fmt}"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ===================================================================
# NEW PLOTS
# ===================================================================


def plot_episode_length_distribution(
    df: pd.DataFrame, outdir: Path, fmt: str
) -> None:
    """Box plot of episode lengths by noise type, coloured by success/failure."""
    noised = df[df["noise_type"] != "clean"].copy()
    if noised.empty:
        return

    noised["outcome"] = noised["success"].map({True: "Success", False: "Failure"})
    max_steps = noised["episode_length"].max()

    noise_types = sorted(noised["noise_type"].unique())
    fig_w = max(8, len(noise_types) * 1.3 + 2)

    fig, ax = plt.subplots(figsize=(fig_w, 5), constrained_layout=True)
    sns.boxplot(
        data=noised,
        x="noise_type",
        y="episode_length",
        hue="outcome",
        palette={"Success": "#4CAF50", "Failure": "#E53935"},
        fliersize=2,
        linewidth=0.7,
        ax=ax,
        order=noise_types,
    )

    # Show the max-steps timeout line
    ax.axhline(max_steps, color="grey", linestyle=":", linewidth=1,
               label=f"max steps ({max_steps})")

    ax.set_xlabel("Noise type")
    ax.set_ylabel("Episode length")
    ax.set_title("Episode Length Distribution by Noise Type")
    ax.tick_params(axis="x", rotation=25)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=True, fancybox=True,
              framealpha=0.8, fontsize=10, loc="upper right")

    out_path = outdir / f"episode_length_distribution.{fmt}"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_severity_by_suite(df: pd.DataFrame, outdir: Path, fmt: str) -> None:
    """Faceted line plot: success rate vs severity, one panel per suite."""
    noised = df[df["noise_type"] != "clean"]
    if noised.empty:
        return

    suites = sorted(noised["suite"].unique())
    if len(suites) < 1:
        return

    noise_types = sorted(noised["noise_type"].unique())
    palette = dict(zip(noise_types, sns.color_palette(_PALETTE, len(noise_types))))

    n_cols = min(len(suites), 3)
    n_rows = (len(suites) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             constrained_layout=True, squeeze=False)

    clean_suite = clean_baseline(df, ["suite"])

    for idx, suite in enumerate(suites):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]

        suite_data = noised[noised["suite"] == suite]
        grouped = (
            suite_data.groupby(["noise_type", "severity"])["success"]
            .agg(["mean", "count"])
            .reset_index()
        )
        grouped["se"] = np.sqrt(
            grouped["mean"] * (1 - grouped["mean"]) / grouped["count"].clip(lower=1)
        )

        for nt in noise_types:
            sub = grouped[grouped["noise_type"] == nt].sort_values("severity")
            if sub.empty:
                continue
            ax.plot(sub["severity"], sub["mean"], marker="o", label=nt,
                    color=palette[nt], linewidth=1.5, markersize=5)
            ax.fill_between(
                sub["severity"],
                (sub["mean"] - 1.96 * sub["se"]).clip(0, 1),
                (sub["mean"] + 1.96 * sub["se"]).clip(0, 1),
                alpha=0.12, color=palette[nt],
            )

        # Clean baseline for this suite
        suite_clean = clean_suite[clean_suite["suite"] == suite]
        if not suite_clean.empty:
            ax.axhline(float(suite_clean["clean_success"].iloc[0]),
                       color="black", linestyle="--", linewidth=1, label="clean")

        ax.set_ylim(0, 1.05)
        ax.set_title(f"Suite: {suite}", fontweight="bold")
        ax.set_xlabel("Severity")
        ax.set_ylabel("Success rate")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Single shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(noise_types) + 1, 6),
               frameon=True, fancybox=True, framealpha=0.8, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    # Hide unused axes
    for idx in range(len(suites), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].set_visible(False)

    out_path = outdir / f"severity_by_suite.{fmt}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_noise_robustness_radar(
    df: pd.DataFrame, sev: int, outdir: Path, fmt: str
) -> None:
    """Radar chart: success rate per noise type, one polygon per suite."""
    noised = df[(df["noise_type"] != "clean") & (df["severity"] == sev)]
    if noised.empty:
        return

    suites = sorted(noised["suite"].unique())
    noise_types = sorted(noised["noise_type"].unique())
    n_vars = len(noise_types)
    if n_vars < 3:
        return  # Radar charts need ≥ 3 axes

    # Compute success rates
    rates = (
        noised.groupby(["suite", "noise_type"])["success"]
        .mean()
        .reset_index()
    )

    angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    palette = sns.color_palette(_PALETTE, len(suites))

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"},
                           constrained_layout=True)

    for i, suite in enumerate(suites):
        values = []
        for nt in noise_types:
            row = rates[(rates["suite"] == suite) & (rates["noise_type"] == nt)]
            values.append(float(row["success"].iloc[0]) if not row.empty else 0.0)
        values += values[:1]  # close polygon

        ax.plot(angles, values, "o-", linewidth=1.5, markersize=4,
                label=suite, color=palette[i])
        ax.fill(angles, values, alpha=0.1, color=palette[i])

    ax.set_thetagrids(np.degrees(angles[:-1]), noise_types, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9, color="grey")
    ax.set_title(f"Noise Robustness Profile (Severity {sev})", y=1.08,
                 fontweight="bold")
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0.0),
              frameon=True, fancybox=True, framealpha=0.8, fontsize=10)

    out_path = outdir / f"noise_robustness_radar_s{sev}.{fmt}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_success_delta_bars(
    df: pd.DataFrame, sev: int, outdir: Path, fmt: str
) -> None:
    """Grouped bar chart: success-rate Δ (clean − noised) per noise type, by suite."""
    noised = df[(df["noise_type"] != "clean") & (df["severity"] == sev)]
    if noised.empty:
        return

    clean_suite = clean_baseline(df, ["suite"])
    grouped = (
        noised.groupby(["suite", "noise_type"])["success"]
        .agg(["mean", "count"])
        .reset_index()
    )
    merged = grouped.merge(clean_suite, on="suite", how="left")
    merged["delta"] = merged["clean_success"] - merged["mean"]

    # Binomial SE on the drop (approximate)
    merged["se"] = np.sqrt(
        merged["mean"] * (1 - merged["mean"]) / merged["count"].clip(lower=1)
    )

    noise_types = sorted(merged["noise_type"].unique())
    suites = sorted(merged["suite"].unique())
    n_noise = len(noise_types)
    n_suites = len(suites)

    x = np.arange(n_noise)
    width = 0.8 / max(n_suites, 1)
    palette = sns.color_palette(_PALETTE, n_suites)

    fig, ax = plt.subplots(
        figsize=(max(7, n_noise * 1.5 + 2), 5), constrained_layout=True
    )

    for i, suite in enumerate(suites):
        suite_data = merged[merged["suite"] == suite].set_index("noise_type")
        vals = [suite_data.loc[nt, "delta"] if nt in suite_data.index else 0
                for nt in noise_types]
        errs = [suite_data.loc[nt, "se"] if nt in suite_data.index else 0
                for nt in noise_types]
        offset = (i - n_suites / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, label=suite,
               color=palette[i], edgecolor="white", linewidth=0.5,
               capsize=2, error_kw={"linewidth": 0.8})

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(noise_types, rotation=25, ha="right")
    ax.set_ylabel("Δ Success rate (clean − noised)")
    ax.set_title(f"Success Rate Drop by Noise Type (Severity {sev})")
    ax.legend(frameon=True, fancybox=True, framealpha=0.8, fontsize=10,
              title="Suite", title_fontsize=10)

    out_path = outdir / f"success_delta_bars_s{sev}.{fmt}"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_success_delta_bars_combined(
    df: pd.DataFrame, outdir: Path, fmt: str
) -> None:
    """Grouped bar chart: success-rate Δ (clean − noised) per noise type, by suite, averaged across all severities."""
    noised = df[df["noise_type"] != "clean"]
    if noised.empty:
        return

    clean_suite = clean_baseline(df, ["suite"])
    grouped = (
        noised.groupby(["suite", "noise_type"])["success"]
        .agg(["mean", "count"])
        .reset_index()
    )
    merged = grouped.merge(clean_suite, on="suite", how="left")
    merged["delta"] = merged["clean_success"] - merged["mean"]

    # Binomial SE on the drop (approximate)
    merged["se"] = np.sqrt(
        merged["mean"] * (1 - merged["mean"]) / merged["count"].clip(lower=1)
    )

    noise_types = sorted(merged["noise_type"].unique())
    suites = sorted(merged["suite"].unique())
    n_noise = len(noise_types)
    n_suites = len(suites)

    x = np.arange(n_noise)
    width = 0.8 / max(n_suites, 1)
    palette = sns.color_palette(_PALETTE, n_suites)

    fig, ax = plt.subplots(
        figsize=(max(7, n_noise * 1.5 + 2), 5), constrained_layout=True
    )

    for i, suite in enumerate(suites):
        suite_data = merged[merged["suite"] == suite].set_index("noise_type")
        vals = [suite_data.loc[nt, "delta"] if nt in suite_data.index else 0
                for nt in noise_types]
        errs = [suite_data.loc[nt, "se"] if nt in suite_data.index else 0
                for nt in noise_types]
        offset = (i - n_suites / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, yerr=errs, label=suite,
               color=palette[i], edgecolor="white", linewidth=0.5,
               capsize=2, error_kw={"linewidth": 0.8})

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(noise_types, rotation=25, ha="right")
    ax.set_ylabel("Δ Success rate (clean − noised)")
    ax.set_title("Success Rate Drop by Noise Type (Averaged across Severities)")
    ax.legend(frameon=True, fancybox=True, framealpha=0.8, fontsize=10,
              title="Suite", title_fontsize=10)
    
    out_path = outdir / f"success_delta_bars_combined.{fmt}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    args = parse_args()
    _apply_theme()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_csvs(args.csv)
    print_coverage(df)

    plot_success_vs_severity(df, outdir, args.format)

    focus = choose_focus_severity(df, args.focus_severity)
    if focus is not None:
        plot_suite_drop_heatmaps(df, outdir, args.format)
        plot_task_drop_heatmap(df, focus, outdir, args.format)
        plot_noise_ranking(df, focus, outdir, args.format)

    plot_episode_length_distribution(df, outdir, args.format)
    plot_severity_by_suite(df, outdir, args.format)

    if focus is not None:
        plot_noise_robustness_radar(df, focus, outdir, args.format)
        plot_success_delta_bars(df, focus, outdir, args.format)
    
    plot_success_delta_bars_combined(df, outdir, args.format)

    print(f"Saved plots to: {outdir}")


if __name__ == "__main__":
    main()