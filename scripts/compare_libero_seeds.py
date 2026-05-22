from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import typer

from vla.constants import OUTPUTS_DIR
from vla.envs.libero import LiberoEnvFactory

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _capture_first_frame(suite: str, task_id: int, seed: int) -> tuple[np.ndarray, str, dict]:
    env = LiberoEnvFactory(suite=suite, task_id=task_id)(0)
    try:
        raw_obs, info = env.reset(seed=seed)
        return env.get_frame(raw_obs), env.task_description, dict(info)
    finally:
        env.close()


def _compute_bbox(mask: np.ndarray) -> dict[str, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max())
    y0 = int(ys.min())
    y1 = int(ys.max())
    return {
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "width": x1 - x0 + 1,
        "height": y1 - y0 + 1,
    }


def _summarize_difference(frame_a: np.ndarray, frame_b: np.ndarray, threshold: int) -> tuple[dict, np.ndarray, np.ndarray]:
    diff = np.abs(frame_a.astype(np.int16) - frame_b.astype(np.int16)).astype(np.uint8)
    diff_gray = diff.mean(axis=2).astype(np.uint8)
    changed_mask = diff_gray >= threshold
    bbox = _compute_bbox(changed_mask)
    bbox_area_fraction = 0.0
    if bbox is not None:
        bbox_area_fraction = (bbox["width"] * bbox["height"]) / float(changed_mask.shape[0] * changed_mask.shape[1])

    metrics = {
        "height": int(frame_a.shape[0]),
        "width": int(frame_a.shape[1]),
        "channels": int(frame_a.shape[2]),
        "threshold": int(threshold),
        "mean_abs_diff": float(diff.mean()),
        "mean_abs_diff_normalized": float(diff.mean() / 255.0),
        "rmse": float(np.sqrt(np.mean((frame_a.astype(np.float32) - frame_b.astype(np.float32)) ** 2))),
        "rmse_normalized": float(
            np.sqrt(np.mean((frame_a.astype(np.float32) - frame_b.astype(np.float32)) ** 2)) / 255.0
        ),
        "max_abs_diff": int(diff.max()),
        "changed_pixel_fraction": float(changed_mask.mean()),
        "changed_pixel_count": int(changed_mask.sum()),
        "bbox_area_fraction": float(bbox_area_fraction),
        "bbox": bbox,
    }
    return metrics, diff_gray, changed_mask


def _save_rgb(path: Path, frame: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def _save_figure(
    out_path: Path,
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    diff_gray: np.ndarray,
    changed_mask: np.ndarray,
    *,
    suite: str,
    task_id: int,
    task_description: str,
    seed_a: int,
    seed_b: int,
    metrics: dict,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(f"LIBERO {suite} task {task_id}: seed {seed_a} vs seed {seed_b}", fontsize=16)

    axes[0, 0].imshow(frame_a)
    axes[0, 0].set_title(f"Seed {seed_a}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(frame_b)
    axes[0, 1].set_title(f"Seed {seed_b}")
    axes[0, 1].axis("off")

    im = axes[1, 0].imshow(diff_gray, cmap="inferno", vmin=0, vmax=255)
    axes[1, 0].set_title("Absolute Difference Heatmap")
    axes[1, 0].axis("off")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(frame_b)
    axes[1, 1].imshow(changed_mask, cmap="cool", alpha=0.45)
    axes[1, 1].set_title(f"Changed Pixels (threshold >= {metrics['threshold']})")
    axes[1, 1].axis("off")

    bbox = metrics.get("bbox")
    if bbox:
        rect = plt.Rectangle(
            (bbox["x0"], bbox["y0"]),
            bbox["width"],
            bbox["height"],
            fill=False,
            edgecolor="white",
            linewidth=2,
        )
        axes[1, 1].add_patch(rect)

    summary = (
        f"{task_description}\n"
        f"mean_abs_diff={metrics['mean_abs_diff']:.2f} ({metrics['mean_abs_diff_normalized']:.3%}) | "
        f"rmse={metrics['rmse']:.2f} ({metrics['rmse_normalized']:.3%}) | "
        f"changed_pixel_fraction={metrics['changed_pixel_fraction']:.3%} | "
        f"bbox_area_fraction={metrics['bbox_area_fraction']:.3%}"
    )
    fig.text(0.5, 0.01, summary, ha="center", va="bottom", fontsize=10)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(
    suite: str = typer.Option("spatial", "--suite"),
    task_id: int = typer.Option(0, "--task-id"),
    seed_a: int = typer.Option(0, "--seed-a"),
    seed_b: int = typer.Option(1, "--seed-b"),
    threshold: int = typer.Option(25, "--threshold", min=0, max=255),
    output_dir: Path | None = typer.Option(None, "--output-dir"),
) -> None:
    out_dir = output_dir or (OUTPUTS_DIR / "libero_seed_diff" / suite / f"task{task_id:02d}_seed{seed_a}_vs_seed{seed_b}")
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_a, task_description, info_a = _capture_first_frame(suite, task_id, seed_a)
    frame_b, _, info_b = _capture_first_frame(suite, task_id, seed_b)
    metrics, diff_gray, changed_mask = _summarize_difference(frame_a, frame_b, threshold)

    payload = {
        "suite": suite,
        "task_id": task_id,
        "task_description": task_description,
        "seed_a": seed_a,
        "seed_b": seed_b,
        "init_state_id_a": info_a.get("libero_init_state_id"),
        "init_state_id_b": info_b.get("libero_init_state_id"),
        "num_init_states": info_a.get("libero_num_init_states") or info_b.get("libero_num_init_states"),
        "metrics": metrics,
    }

    metrics_path = out_dir / "metrics.json"
    figure_path = out_dir / "comparison.png"
    frame_a_path = out_dir / f"seed_{seed_a}_frame0.png"
    frame_b_path = out_dir / f"seed_{seed_b}_frame0.png"

    _save_rgb(frame_a_path, frame_a)
    _save_rgb(frame_b_path, frame_b)
    _save_figure(
        figure_path,
        frame_a,
        frame_b,
        diff_gray,
        changed_mask,
        suite=suite,
        task_id=task_id,
        task_description=task_description,
        seed_a=seed_a,
        seed_b=seed_b,
        metrics=metrics,
    )
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"Saved figure: {figure_path}")


if __name__ == "__main__":
    typer.run(main)
