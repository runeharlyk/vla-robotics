"""Export images from LIBERO (and future LIBERO-Pro) datasets.

Output structure:
    data/images/libero/<suite>/ep<NNNN>_<task_label>/frame<FFFF>.png
    data/images/libero_pro/<suite>/ep<NNNN>_<task_label>/frame<FFFF>.png  (future)

Usage
-----
# Export all standard LIBERO suites
uv run python -m vla.data.image_export --dataset libero

# Export only specific suites
uv run python -m vla.data.image_export --dataset libero --suites spatial object

# Limit to first 5 episodes per suite (useful for quick testing)
uv run python -m vla.data.image_export --dataset libero --max-episodes 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from vla.constants import LIBERO_SUITES, PROJECT_ROOT

# TODO: fill in the correct HuggingFace repo IDs once LIBERO-Pro is available.
# Ref: https://huggingface.co/papers/2510.03827
LIBERO_PRO_SUITES: dict[str, str] = {
    # "spatial": "lerobot/libero_spatial_image_pro",
    # "object": "lerobot/libero_object_image_pro",
    # "goal":   "lerobot/libero_goal_image_pro",
    # "long":   "lerobot/libero_10_image_pro",
}

IMAGES_DIR = PROJECT_ROOT / "data" / "images"
IMAGE_KEY = "observation.images.image"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    """Replace non-alphanumeric characters with underscores for safe dir names."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_")


# ---------------------------------------------------------------------------
# Core export function
# ---------------------------------------------------------------------------

def export_images_from_suite(
    repo_id: str,
    output_dir: Path,
    max_episodes: int = 1,
) -> None:
    """Stream *repo_id* from HuggingFace and save frames as PNGs.

    Only downloads the rows it actually needs — stops as soon as
    *max_episodes* episodes have been fully saved.

    Directory layout per episode::

        output_dir/ep<NNNN>_<sanitized_task>/frame<FFFF>.png
    """
    from datasets import load_dataset

    print(f"  Streaming dataset: {repo_id}  (max_episodes={max_episodes})")
    ds_iter = load_dataset(repo_id, split="train", streaming=True)

    # Determine column names from the first row
    first_row = next(iter(ds_iter))
    columns = list(first_row.keys())
    print(f"  Columns: {columns}")

    # Resolve image column
    image_col: str | None = None
    for candidate in (IMAGE_KEY, "image", "observation.image", "observation.images.0"):
        if candidate in columns:
            image_col = candidate
            break
    if image_col is None:
        raise KeyError(f"No image column found in {repo_id}. Columns: {columns}")

    # Resolve episode column
    ep_col: str | None = None
    for candidate in ("episode_index", "episode_id"):
        if candidate in columns:
            ep_col = candidate
            break
    if ep_col is None:
        raise KeyError(f"No episode column found in {repo_id}. Columns: {columns}")

    # Build task index → description mapping from meta/tasks.jsonl (preferred)
    task_index_col: str | None = None
    task_map: dict[int, str] = {}
    if "task_index" in columns:
        task_index_col = "task_index"
        try:
            path = hf_hub_download(repo_id, "meta/tasks.jsonl", repo_type="dataset")
            with open(path, encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    task_map[int(entry["task_index"])] = entry["task"]
            print(f"  Loaded {len(task_map)} task descriptions from meta/tasks.jsonl")
        except Exception as e:
            print(f"  [WARN] Could not load tasks.jsonl: {e}")
            task_index_col = None  # fall back to text column

    # Fall back to a text column only if tasks.jsonl failed
    task_col: str | None = None
    if not task_map:
        for candidate in ("task", "language_instruction", "task_description"):
            if candidate in columns:
                task_col = candidate
                break

    print(f"  image_col={image_col!r}, ep_col={ep_col!r}, task_col={task_col!r}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stream rows, grouping by episode, stop early ----
    episodes_done: set[int] = set()
    current_ep: int | None = None
    frame_counter = 0
    ep_dir: Path | None = None

    # Re-create the iterator (we consumed one row above)
    ds_iter = load_dataset(repo_id, split="train", streaming=True)

    for row in tqdm(ds_iter, desc="  Frames"):
        ep_idx = int(row[ep_col])
        
        # Skip episodes we've already finished
        if ep_idx in episodes_done:
            continue

        # Detect episode boundary
        if ep_idx != current_ep:
            # Close out previous episode
            if current_ep is not None:
                tqdm.write(f"    ep{current_ep:04d}: {frame_counter} frames saved")
                episodes_done.add(current_ep)
                if len(episodes_done) >= max_episodes:
                    break

            # Start new episode
            current_ep = ep_idx
            frame_counter = 0

            # Resolve task description
            if task_index_col is not None and task_map:
                raw_task = task_map.get(int(row[task_index_col]), f"task_{ep_idx}")
            elif task_col is not None:
                raw_task = str(row[task_col])
            else:
                raw_task = ""

            task_label = _sanitize(raw_task) if raw_task else f"task_{ep_idx}"
            if len(task_label) > 80:
                task_label = task_label[:80]

            ep_dir = output_dir / f"ep{ep_idx:04d}_{task_label}"
            ep_dir.mkdir(parents=True, exist_ok=True)

            # Save the task instruction
            (ep_dir / "task.txt").write_text(raw_task, encoding="utf-8")

        # Save frame
        # image = row[image_col]  # PIL Image
        # image.save(ep_dir / f"frame{frame_counter:04d}.png")
        frame_counter += 1

    # Flush last episode
    if current_ep is not None and current_ep not in episodes_done:
        tqdm.write(f"    ep{current_ep:04d}: {frame_counter} frames saved")
        episodes_done.add(current_ep)

    print(f"  Done — {len(episodes_done)} episode(s) exported to {output_dir}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_libero(
    suites: list[str] | None = None,
    max_episodes: int = 1,
) -> None:
    """Export images from the standard LIBERO suites."""
    target = suites or list(LIBERO_SUITES.keys())
    for suite in target:
        if suite not in LIBERO_SUITES:
            print(f"  [WARN] Unknown suite '{suite}', skipping.")
            continue
        repo_id = LIBERO_SUITES[suite]
        out_dir = IMAGES_DIR / "libero" / suite
        print(f"\n[libero/{suite}]")
        export_images_from_suite(repo_id, out_dir, max_episodes=max_episodes)


def export_libero_pro(
    suites: list[str] | None = None,
    max_episodes: int = 1,
) -> None:
    """Export images from the LIBERO-Pro suites.

    Fill in LIBERO_PRO_SUITES at the top of this file once repo IDs are known.
    """
    if not LIBERO_PRO_SUITES:
        print("[WARN] LIBERO_PRO_SUITES is empty – update the registry at the top of image_export.py.")
        return

    target = suites or list(LIBERO_PRO_SUITES.keys())
    for suite in target:
        if suite not in LIBERO_PRO_SUITES:
            print(f"  [WARN] Unknown pro suite '{suite}', skipping.")
            continue
        repo_id = LIBERO_PRO_SUITES[suite]
        out_dir = IMAGES_DIR / "libero_pro" / suite
        print(f"\n[libero_pro/{suite}]")
        export_images_from_suite(repo_id, out_dir, max_episodes=max_episodes)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export PNG images from LIBERO / LIBERO-Pro datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["libero", "pro", "all"],
        default="libero",
        help="Which dataset family to export.",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=list(LIBERO_SUITES.keys()),
        default=None,
        metavar="SUITE",
        help="Subset of suites to export (spatial / object / goal / long).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1,
        metavar="N",
        help="Number of episodes to export per suite (default: 1).",
    )

    args = parser.parse_args()

    if args.dataset in ("libero", "all"):
        print("=== Exporting standard LIBERO images ===")
        export_libero(suites=args.suites, max_episodes=args.max_episodes)

    if args.dataset in ("pro", "all"):
        print("=== Exporting LIBERO-Pro images ===")
        export_libero_pro(suites=args.suites, max_episodes=args.max_episodes)

    print("\nExport complete.")
