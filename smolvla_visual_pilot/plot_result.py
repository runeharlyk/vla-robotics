"""Generate analysis plots for the SmolVLA visual-pilot experiment.

Reads the per-run ``results.csv`` files produced by ``run_evaluation.py``.
Two layouts are auto-detected under ``--results-dir``:

  1. **Multi-suite (preferred):** ``<results_dir>/<suite>/<sev>/results.csv``
     e.g. ``goal/s1/results.csv``, ``object/s5/results.csv``.  Suite identity
     is preserved so most plots are faceted by suite.

  2. **Flat (legacy):** ``<results_dir>/visual_pilot_eval_s{N}/results.csv``.
     A single ``suite="all"`` column is synthesised so every plot still works.

Example::

    python smolvla_visual_pilot/plot_result.py \\
        --results-dir smolvla_visual_pilot/l2_goal_object_spatial_result \\
        --out-dir smolvla_visual_pilot/plots
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless / cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Style constants — kept in sync with visual_diagnostic/visualizations.py so
# the same noise type is the same colour everywhere visual-noise results are
# plotted.  The whole visual family stays clearly distinct from the blue/orange
# Set2 / tab10 palettes used by the language-perturbation experiments.
# ---------------------------------------------------------------------------

NOISE_PALETTE = {
    "motion_blur":   "#FFD60A",   # bright yellow
    "gaussian_blur": "#FB8500",   # vivid orange
    "zoom_blur":     "#E63946",   # red
    "fog":           "#228B22",   # forest green
    "glass_blur":    "#6A0DAD",   # deep purple
}

NOISE_ORDER = ["motion_blur", "gaussian_blur", "zoom_blur", "fog", "glass_blur"]

NOISE_NICE = {
    "motion_blur":   "Motion Blur",
    "gaussian_blur": "Gaussian Blur",
    "zoom_blur":     "Zoom Blur",
    "fog":           "Fog",
    "glass_blur":    "Glass Blur",
}

# Warm intensity progression; used wherever severity is the colour axis
# (headline summary, quality bars, violins).  Muted/matte palette so the bars
# read calmly rather than screaming saturated.
SEVERITY_LABELS = {1: "Severity 1", 3: "Severity 3", 5: "Severity 5"}
SEVERITY_COLORS = {1: "#E5C07B", 3: "#D08A56", 5: "#B04A48"}

# Suite colours stay warm — tan / brick / brown — so they don't clash with the
# pink/orange/red noise palette and stay clearly distinct from the language
# experiments (Set2 / tab10 blues and greens).
SUITE_PALETTE = {
    "goal":    "#D4A373",   # tan
    "object":  "#BC4749",   # brick red
    "spatial": "#774936",   # warm brown
}
SUITE_ORDER = ["goal", "object", "spatial"]

DIM_NAMES = ["X", "Y", "Z", "Roll", "Pitch", "Yaw", "Gripper"]
ACTION_DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ABS_ERR_COLS = [f"abs_err_{d}" for d in ACTION_DIMS]

# Libero dataset ground-truth action standard deviations (for normalising).
ACTION_STDS = np.array(
    [0.33552372, 0.378447, 0.4447286, 0.03924354,
     0.06339297, 0.07797027, 0.99876714]
)

USECOLS = [
    "task_index", "task_name", "noise_type", "noise_severity",
    "l2_distance", "rel_l2_distance", "quality_delta_l2",
    *ABS_ERR_COLS,
]


def _setup_style() -> None:
    """Apply publication-quality matplotlib defaults."""
    plt.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size":          11,
        "axes.titlesize":     13,
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    10,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.alpha":         0.3,
        "grid.linewidth":     0.6,
    })


def _truncate(name: str, max_len: int = 55) -> str:
    return name if len(name) <= max_len else name[: max_len - 1].rstrip() + "…"


def _save(fig: plt.Figure, path: Path) -> None:
    # Don't call tight_layout when the figure already uses constrained_layout
    # (matplotlib raises if a colorbar was added in the other engine).
    engine = fig.get_layout_engine()
    if engine is None or engine.__class__.__name__.lower().startswith("tight"):
        try:
            fig.tight_layout()
        except Exception:
            pass
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def _suites_in(df: pd.DataFrame) -> list[str]:
    cats = df["suite"].cat.categories if hasattr(df["suite"], "cat") else df["suite"].unique()
    return [s for s in cats if s in set(df["suite"])]


# ---------------------------------------------------------------------------
# Data loading — auto-detects multi-suite vs flat layout.
# ---------------------------------------------------------------------------

def _discover_csvs(results_dir: Path) -> list[tuple[str, Path]]:
    """Return ``[(suite, csv_path), ...]`` covering whichever layout is present.

    Multi-suite layout takes priority; falls back to the flat legacy layout.
    """
    out: list[tuple[str, Path]] = []

    # Multi-suite: <root>/<suite>/<sev>/results.csv
    for csv_path in sorted(results_dir.glob("*/*/results.csv")):
        suite = csv_path.parent.parent.name
        # Skip the legacy flat sentinel directories if they happen to nest.
        if suite.startswith("visual_pilot_eval_s"):
            continue
        out.append((suite, csv_path))

    if out:
        return out

    # Flat legacy: <root>/visual_pilot_eval_s{N}/results.csv
    for csv_path in sorted(results_dir.glob("visual_pilot_eval_s*/results.csv")):
        out.append(("all", csv_path))
    return out


def load_long_df(results_dir: Path) -> pd.DataFrame:
    """Concatenate every results.csv into one long DataFrame with a ``suite`` col."""
    discovered = _discover_csvs(results_dir)
    if not discovered:
        raise SystemExit(
            f"No results.csv found under {results_dir} "
            f"(looked for */*/results.csv and visual_pilot_eval_s*/results.csv)."
        )

    frames: list[pd.DataFrame] = []
    for suite, csv_path in discovered:
        try:
            df = pd.read_csv(csv_path, usecols=lambda c: c in USECOLS)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  ! skipping {csv_path}: {exc}")
            continue
        df["suite"] = suite
        frames.append(df)
        sev_dir = csv_path.parent.name
        print(
            f"  + {suite}/{sev_dir}: {len(df):>9,} rows  "
            f"(severities={sorted(df['noise_severity'].unique())})"
        )

    df = pd.concat(frames, ignore_index=True)
    df["noise_type"] = pd.Categorical(
        df["noise_type"],
        categories=[n for n in NOISE_ORDER if n in set(df["noise_type"])],
        ordered=True,
    )
    df["suite"] = pd.Categorical(
        df["suite"],
        categories=[s for s in SUITE_ORDER if s in set(df["suite"])] +
                   [s for s in df["suite"].unique() if s not in SUITE_ORDER],
        ordered=True,
    )
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_headline_summary(df: pd.DataFrame, out_dir: Path) -> None:
    """Relative deviation (%) per noise × severity, grouped bars with annotation."""
    _setup_style()
    g = (
        df.groupby(["noise_type", "noise_severity"], observed=True)["rel_l2_distance"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    g["pct"] = g["mean"] * 100
    g["sem_pct"] = g["sem"] * 100

    sevs = sorted(df["noise_severity"].unique())
    noise_types = [n for n in NOISE_ORDER if n in set(df["noise_type"])]

    fig, ax = plt.subplots(figsize=(11, 6))
    n_noise = len(noise_types)
    n_sev = len(sevs)
    bar_width = 0.24
    x_base = np.arange(n_noise)

    worst_val, worst_label = -np.inf, ""
    for i, s in enumerate(sevs):
        offset = (i - (n_sev - 1) / 2) * (bar_width + 0.03)
        subset = g[g["noise_severity"] == s].set_index("noise_type").reindex(noise_types)
        vals = subset["pct"].values
        errs = subset["sem_pct"].values
        ax.bar(x_base + offset, vals, width=bar_width,
               color=SEVERITY_COLORS.get(s, "#888"), edgecolor="white",
               linewidth=0.5, label=SEVERITY_LABELS.get(s, f"Severity {s}"),
               zorder=3)
        ax.errorbar(x_base + offset, vals, yerr=errs, fmt="none",
                    ecolor="#333", capsize=3, linewidth=1.0, zorder=4)
        for xi, val, nt in zip(x_base + offset, vals, noise_types):
            ax.text(xi, val + 1.0, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=13, color="#333")
            if val > worst_val:
                worst_val = val
                worst_label = f"{NOISE_NICE.get(nt, nt)} s{s}: {val:.0f}%"

    ax.annotate(
        f"Worst: {worst_label}",
        xy=(0.99, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=13, fontweight="bold", color="#E63946",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff0f0",
                  edgecolor="#E63946", linewidth=1.2),
    )

    ax.tick_params(axis="y", labelsize=12)

    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=2,
               label="Clean reference (0%)")
    ax.set_xticks(x_base)
    ax.set_xticklabels([NOISE_NICE.get(nt, nt) for nt in noise_types], fontsize=13)
    ax.set_ylabel("Relative Deviation  ‖noisy − clean‖ / ‖clean‖  (%)")
    ax.set_title("SmolVLA Action Deviation from Clean Rollout Under Visual Corruption",
                 fontweight="bold", fontsize=13.5, pad=14)
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, edgecolor="#ccc",
              loc="upper left")
    ax.set_ylim(0, max(g["pct"].max() * 1.3, 5))

    _save(fig, out_dir / "headline_summary.png")


def _plot_severity_curves_faceted(
    df: pd.DataFrame, out_dir: Path, value_col: str, fname: str,
    ylabel: str, title: str, scale_to_pct: bool = False,
) -> None:
    """Per-suite faceted line chart of ``value_col`` vs severity, noise hue."""
    _setup_style()
    suites = _suites_in(df)
    sevs = sorted(df["noise_severity"].unique())
    noise_types = [n for n in NOISE_ORDER if n in set(df["noise_type"])]

    fig, axes = plt.subplots(1, len(suites), figsize=(5.5 * len(suites), 5.2),
                             sharey=True, squeeze=False)
    y_max = 0.0
    for ax, suite in zip(axes[0], suites):
        sub = df[df["suite"] == suite]
        for nt in noise_types:
            grp = sub[sub["noise_type"] == nt]
            if grp.empty:
                continue
            agg = (grp.groupby("noise_severity", observed=True)[value_col]
                      .agg(mean="mean", std="std", n="count").reset_index())
            agg["sem"] = agg["std"] / np.sqrt(agg["n"].clip(lower=1))
            mult = 100 if scale_to_pct else 1
            means = agg["mean"].values * mult
            sems = agg["sem"].values * mult
            y_max = max(y_max, (means + sems).max())
            ax.plot(agg["noise_severity"], means, marker="o", markersize=6,
                    linewidth=2.0, color=NOISE_PALETTE[nt],
                    label=NOISE_NICE[nt])
            ax.fill_between(agg["noise_severity"],
                            np.maximum(means - sems, 0), means + sems,
                            alpha=0.20, color=NOISE_PALETTE[nt])
        ax.set_title(f"Suite: {suite}", fontweight="bold")
        ax.set_xlabel("Noise Severity Level")
        ax.set_xticks(sevs)
        ax.set_xticklabels([f"s{s}" for s in sevs])
        ax.set_ylim(bottom=0)

    axes[0][0].set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(noise_types), frameon=True, fancybox=True,
               framealpha=0.9, edgecolor="#ccc", bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(title, fontweight="bold", fontsize=14, y=1.02)
    _save(fig, out_dir / fname)


def plot_severity_degradation(df: pd.DataFrame, out_dir: Path) -> None:
    _plot_severity_curves_faceted(
        df, out_dir, "l2_distance", "severity_degradation_curves.png",
        ylabel="Mean L2 Distance (action space)",
        title="Action Deviation vs. Noise Severity (per Suite)",
    )


def plot_relative_degradation_curves(df: pd.DataFrame, out_dir: Path) -> None:
    _plot_severity_curves_faceted(
        df, out_dir, "rel_l2_distance", "relative_degradation_curves.png",
        ylabel="Relative Deviation  ‖noisy − clean‖ / ‖clean‖  (%)",
        title="Relative Deviation from Clean-Model Behaviour (per Suite)",
        scale_to_pct=True,
    )


def plot_quality_degradation_bars(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-suite faceted grouped bars of mean L2, noise × severity."""
    _setup_style()
    suites = _suites_in(df)
    sevs = sorted(df["noise_severity"].unique())
    noise_types = [n for n in NOISE_ORDER if n in set(df["noise_type"])]

    fig, axes = plt.subplots(1, len(suites), figsize=(5.5 * len(suites), 5.2),
                             sharey=True, squeeze=False)
    n_noise = len(noise_types)
    bar_width = 0.22
    x_base = np.arange(n_noise)

    y_max = 0.0
    for ax, suite in zip(axes[0], suites):
        sub = df[df["suite"] == suite]
        for i, s in enumerate(sevs):
            agg = (sub[sub["noise_severity"] == s]
                   .groupby("noise_type", observed=True)["l2_distance"]
                   .agg(mean="mean", std="std", n="count").reset_index())
            agg["sem"] = agg["std"] / np.sqrt(agg["n"].clip(lower=1))
            agg = agg.set_index("noise_type").reindex(noise_types)
            vals = agg["mean"].fillna(0).values
            errs = agg["sem"].fillna(0).values
            y_max = max(y_max, (vals + errs).max())
            offset = (i - (len(sevs) - 1) / 2) * (bar_width + 0.04)
            ax.bar(x_base + offset, vals, width=bar_width,
                   color=SEVERITY_COLORS.get(s, "#888"), edgecolor="white",
                   linewidth=0.5,
                   label=SEVERITY_LABELS.get(s, f"Severity {s}"), zorder=3)
            ax.errorbar(x_base + offset, vals, yerr=errs, fmt="none",
                        ecolor="#333", capsize=3, linewidth=1.0, zorder=4)
        ax.axhline(0, color="#888", linewidth=0.7, linestyle="--", zorder=2)
        ax.set_xticks(x_base)
        ax.set_xticklabels([NOISE_NICE[nt] for nt in noise_types],
                           rotation=20, ha="right", fontsize=9)
        ax.set_title(f"Suite: {suite}", fontweight="bold")

    axes[0][0].set_ylabel("Mean L2 Deviation from Clean Rollout")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(sevs), frameon=True, fancybox=True,
               framealpha=0.9, edgecolor="#ccc", bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Action Deviation Caused by Visual Noise (per Suite)",
                 fontweight="bold", fontsize=14, y=1.02)
    _save(fig, out_dir / "quality_degradation_bars.png")


def plot_dimension_errors_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-severity panels: noise × action dim of normalised mean abs error."""
    _setup_style()
    sevs = sorted(df["noise_severity"].unique())
    noise_types = [n for n in NOISE_ORDER if n in set(df["noise_type"])]
    n_sev = len(sevs)

    matrices = {}
    all_vals: list[float] = []
    for s in sevs:
        sub = df[df["noise_severity"] == s]
        agg = sub.groupby("noise_type", observed=True)[ABS_ERR_COLS].mean()
        agg = agg.div(ACTION_STDS, axis=1)
        mat = np.zeros((len(noise_types), len(DIM_NAMES)))
        for i, nt in enumerate(noise_types):
            if nt in agg.index:
                mat[i, :] = agg.loc[nt, ABS_ERR_COLS].values
        matrices[s] = mat
        all_vals.extend(mat.flatten())

    vmin = 0.0
    vmax = float(np.percentile(all_vals, 98)) * 1.05 if all_vals else 1.0

    fig, axes = plt.subplots(n_sev, 1, figsize=(9, 4.2 * n_sev + 0.8),
                             constrained_layout=True)
    if n_sev == 1:
        axes = [axes]

    for ax, s in zip(axes, sevs):
        mat = matrices[s]
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(DIM_NAMES)))
        ax.set_xticklabels(DIM_NAMES, fontsize=11, fontweight="medium")
        ax.set_yticks(range(len(noise_types)))
        ax.set_yticklabels([NOISE_NICE[nt] for nt in noise_types], fontsize=10)
        ax.set_title(f"Severity {s}", fontweight="bold", fontsize=12)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                tc = "white" if val > vmax * 0.65 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=10, color=tc, fontweight="medium")

    fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02,
                 label="Normalised MAE (per action std)")
    fig.suptitle("Per-Dimension Error by Noise Type & Severity\n"
                 "(normalised by action std)",
                 fontweight="bold", fontsize=14)
    _save(fig, out_dir / "dimension_errors_heatmap.png")


def plot_dimension_radar(df: pd.DataFrame, out_dir: Path) -> None:
    """Radar of normalised per-dim errors at the focus severity (s3 if present)."""
    _setup_style()
    sevs = sorted(df["noise_severity"].unique())
    target_sev = 3 if 3 in sevs else sevs[len(sevs) // 2]

    sub = df[df["noise_severity"] == target_sev]
    agg = sub.groupby("noise_type", observed=True)[ABS_ERR_COLS].mean()
    agg = agg.div(ACTION_STDS, axis=1)

    noise_types = [n for n in NOISE_ORDER if n in agg.index]
    n_dims = len(DIM_NAMES)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for nt in noise_types:
        vals = agg.loc[nt, ABS_ERR_COLS].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, marker="o", markersize=5,
                color=NOISE_PALETTE[nt], label=NOISE_NICE[nt])
        ax.fill(angles, vals, alpha=0.08, color=NOISE_PALETTE[nt])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(DIM_NAMES, fontsize=12, fontweight="medium")
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title(f"Per-Dimension Error Profile",
                 fontweight="bold", fontsize=17, pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05),
              frameon=True, fancybox=True, framealpha=0.9, edgecolor="#ccc",fontsize=12)
    _save(fig, out_dir / "dimension_radar_chart.png")


def plot_severity_radar_multiples(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-severity radar small multiples of normalised per-dim errors."""
    _setup_style()
    sevs = sorted(df["noise_severity"].unique())
    noise_types = [n for n in NOISE_ORDER if n in set(df["noise_type"])]
    n_dims = len(DIM_NAMES)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]

    # Pre-compute normalised means and a global max for shared radial scale.
    normalised = {}
    global_max = 0.0
    for s in sevs:
        sub = df[df["noise_severity"] == s]
        agg = sub.groupby("noise_type", observed=True)[ABS_ERR_COLS].mean()
        agg = agg.div(ACTION_STDS, axis=1)
        normalised[s] = agg
        if not agg.empty:
            global_max = max(global_max, float(agg.values.max()))
    global_max *= 1.1

    fig, axes = plt.subplots(1, len(sevs), figsize=(5.5 * len(sevs), 6.5),
                             subplot_kw={"polar": True})
    if len(sevs) == 1:
        axes = [axes]

    for ax, s in zip(axes, sevs):
        agg = normalised[s]
        for nt in noise_types:
            if nt not in agg.index:
                continue
            vals = agg.loc[nt, ABS_ERR_COLS].tolist()
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2, marker="o", markersize=4,
                    color=NOISE_PALETTE[nt], label=NOISE_NICE[nt])
            ax.fill(angles, vals, alpha=0.06, color=NOISE_PALETTE[nt])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(DIM_NAMES, fontsize=11, fontweight="medium")
        ax.set_ylim(0, global_max)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(f"Severity {s}", fontweight="bold", fontsize=13, pad=20)

    fig.subplots_adjust(wspace=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(noise_types),
               frameon=True, fancybox=True, framealpha=0.9, edgecolor="#ccc",
               fontsize=10, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Per-Dimension Error Profile Across Severities\n"
                 "(normalised by action std)",
                 fontweight="bold", fontsize=14, y=1.04)
    _save(fig, out_dir / "severity_radar_multiples.png")


def plot_noise_severity_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Single heatmap: noise type × (suite, severity) of mean L2."""
    _setup_style()
    suites = _suites_in(df)
    sevs = sorted(df["noise_severity"].unique())
    noise_types = [n for n in NOISE_ORDER if n in set(df["noise_type"])]

    g = (df.groupby(["noise_type", "suite", "noise_severity"], observed=True)
           ["l2_distance"].mean().reset_index())
    g["col"] = g["suite"].astype(str) + " · s" + g["noise_severity"].astype(str)
    pivot = g.pivot(index="noise_type", columns="col", values="l2_distance")
    pivot = pivot.reindex(noise_types)
    ordered_cols = [f"{s} · s{sev}" for s in suites for sev in sevs
                    if f"{s} · s{sev}" in pivot.columns]
    pivot = pivot[ordered_cols]

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(ordered_cols)), 5.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                cbar_kws={"label": "Mean L2"}, ax=ax,
                linewidths=0.4, linecolor="white")
    ax.set_yticklabels([NOISE_NICE[nt] for nt in pivot.index],
                       rotation=0, fontsize=10)
    ax.set_title("Mean L2 by Noise Type × Suite/Severity", fontweight="bold")
    ax.set_xlabel("Suite · Severity")
    ax.set_ylabel("Noise Type")
    _save(fig, out_dir / "noise_severity_heatmap.png")


def plot_task_noise_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    """Per-suite heatmaps: task (rows) × noise type (cols) of mean L2."""
    _setup_style()
    suites = _suites_in(df)
    noise_types = [n for n in NOISE_ORDER if n in set(df["noise_type"])]

    # Compute global range so colours are comparable across suites.
    g_all = (df.groupby(["suite", "task_name", "noise_type"], observed=True)
               ["l2_distance"].mean().reset_index())
    vmin = float(g_all["l2_distance"].min()) if not g_all.empty else 0.0
    vmax = float(g_all["l2_distance"].max()) * 1.05 if not g_all.empty else 1.0

    fig, axes = plt.subplots(1, len(suites), figsize=(6.5 * len(suites), 8),
                             squeeze=False)
    for idx, (ax, suite) in enumerate(zip(axes[0], suites)):
        sub = df[df["suite"] == suite]
        g = (sub.groupby(["task_name", "noise_type"], observed=True)
                ["l2_distance"].mean().reset_index())
        pivot = g.pivot(index="task_name", columns="noise_type",
                        values="l2_distance")
        pivot = pivot.reindex(columns=[n for n in noise_types if n in pivot.columns])
        pivot.index = [_truncate(t, 45) for t in pivot.index]

        show_cbar = (idx == len(suites) - 1)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                    vmin=vmin, vmax=vmax,
                    cbar=show_cbar,
                    cbar_kws={"label": "Mean L2"} if show_cbar else None,
                    linewidths=0.4, linecolor="white",
                    ax=ax)
        ax.set_xticklabels([NOISE_NICE[c] for c in pivot.columns],
                           rotation=35, ha="right", fontsize=10)
        ax.set_title(f"Suite: {suite}", fontweight="bold", fontsize=12)
        ax.set_xlabel("Noise Type")
        ax.set_ylabel("Task" if idx == 0 else "")
    fig.suptitle("Mean L2 Deviation per Task × Noise Type",
                 fontweight="bold", fontsize=14, y=1.02)
    _save(fig, out_dir / "task_noise_heatmap.png")


def plot_task_vulnerability(df: pd.DataFrame, out_dir: Path) -> None:
    """Horizontal bar ranking of tasks by mean L2, coloured by suite."""
    _setup_style()
    g = (df.groupby(["suite", "task_index", "task_name"], observed=True)
           ["l2_distance"].mean().reset_index()
           .sort_values("l2_distance", ascending=True))
    g["label"] = [f"T{ti}: {_truncate(name, 50)}  [{s}]"
                  for s, ti, name in zip(g["suite"], g["task_index"], g["task_name"])]

    colors = [SUITE_PALETTE.get(str(s), "#888") for s in g["suite"]]
    fig, ax = plt.subplots(figsize=(12, max(6, 0.34 * len(g))))
    ax.barh(g["label"], g["l2_distance"], color=colors,
            edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#555", linewidth=1.2)
    ax.set_xlabel("Mean L2 Deviation (averaged over noise × severity)")
    ax.set_title("Task Vulnerability Ranking",
                 fontweight="bold", pad=12)
    ax.set_xlim(left=0)
    ax.invert_yaxis()

    handles = [plt.Rectangle((0, 0), 1, 1,
                              color=SUITE_PALETTE.get(s, "#888"))
               for s in _suites_in(df)]
    ax.legend(handles, _suites_in(df), title="Suite", loc="lower right",
              frameon=True, fancybox=True, framealpha=0.9, edgecolor="#ccc")
    _save(fig, out_dir / "task_vulnerability_ranking.png")


def plot_l2_violins(df: pd.DataFrame, out_dir: Path,
                    sample_per_group: int = 4000) -> None:
    """Per-suite violins of per-timestep L2 distribution, severity hue."""
    _setup_style()
    rng = np.random.default_rng(0)
    parts = [
        sub.sample(min(len(sub), sample_per_group),
                   random_state=int(rng.integers(1 << 31)))
        for _, sub in df.groupby(["suite", "noise_severity", "noise_type"],
                                  observed=True)
    ]
    if not parts:
        print("  ! no data for violins, skipping")
        return
    sampled = pd.concat(parts, ignore_index=True)
    sevs = sorted(df["noise_severity"].unique())
    sev_palette = {f"s{s}": SEVERITY_COLORS.get(s, "#888") for s in sevs}

    suites = _suites_in(df)
    noise_types_nice = [NOISE_NICE[n] for n in NOISE_ORDER if n in set(df["noise_type"])]
    sampled["Noise Type"] = sampled["noise_type"].map(NOISE_NICE)
    sampled["Severity"] = sampled["noise_severity"].apply(lambda s: f"s{s}")

    p99 = float(sampled["l2_distance"].quantile(0.99))
    sampled_clip = sampled.copy()
    sampled_clip["l2_distance"] = sampled_clip["l2_distance"].clip(upper=p99)

    fig, axes = plt.subplots(1, len(suites), figsize=(6 * len(suites), 5.5),
                             sharey=True, squeeze=False)
    for ax, suite in zip(axes[0], suites):
        sub = sampled_clip[sampled_clip["suite"] == suite]
        sns.violinplot(
            data=sub, x="Noise Type", y="l2_distance", hue="Severity",
            order=noise_types_nice,
            hue_order=[f"s{s}" for s in sevs],
            palette=sev_palette, inner="quartile", cut=0,
            linewidth=0.7, ax=ax, density_norm="width",
        )
        ax.set_title(f"Suite: {suite}", fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)
        if ax is axes[0][0]:
            ax.set_ylabel(f"L2 Distance (clipped at p99 = {p99:.2f})")
        else:
            ax.set_ylabel("")
        if ax is axes[0][-1]:
            ax.legend(title="Severity", frameon=True, fancybox=True,
                      framealpha=0.9, edgecolor="#ccc")
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

    fig.suptitle("Distribution of Per-Timestep L2 Distances (per Suite)",
                 fontweight="bold", fontsize=14, y=1.02)
    _save(fig, out_dir / "l2_distribution_violins.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def plot_all(df: pd.DataFrame, out_dir: Path, do_violins: bool = True,
             violin_sample: int = 4000) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting figures to {out_dir} ...")
    plot_headline_summary(df, out_dir)
    plot_severity_degradation(df, out_dir)
    plot_relative_degradation_curves(df, out_dir)
    plot_quality_degradation_bars(df, out_dir)
    plot_dimension_errors_heatmap(df, out_dir)
    plot_dimension_radar(df, out_dir)
    plot_severity_radar_multiples(df, out_dir)
    plot_noise_severity_heatmap(df, out_dir)
    plot_task_noise_heatmap(df, out_dir)
    plot_task_vulnerability(df, out_dir)
    if do_violins:
        plot_l2_violins(df, out_dir, violin_sample)
    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).resolve().parent / "l2_goal_object_spatial_result",
        help="Root holding <suite>/<sev>/results.csv "
             "(or legacy visual_pilot_eval_s*/results.csv). "
             "Default: %(default)s",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parent / "plots",
        help="Directory for the generated PNGs (default: %(default)s)",
    )
    parser.add_argument(
        "--no-violins", action="store_true",
        help="Skip the (heavier) per-timestep distribution violins.",
    )
    parser.add_argument(
        "--violin-sample", type=int, default=4000,
        help="Max rows sampled per (suite, severity, noise) group for violins "
             "(default: %(default)s).",
    )
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir} ...")
    df = load_long_df(args.results_dir)
    print(
        f"Loaded {len(df):,} rows | suites={_suites_in(df)} | "
        f"severities={sorted(df['noise_severity'].unique())} | "
        f"noise types={[str(n) for n in df['noise_type'].cat.categories]}"
    )
    plot_all(df, args.out_dir, do_violins=not args.no_violins,
             violin_sample=args.violin_sample)


if __name__ == "__main__":
    main()
