"""Download clean LIBERO datasets from HuggingFace and convert to h5.

Downloads lerobot/libero_object, libero_spatial, libero_goal, libero_10.
Converts each episode to Layout-B h5 (parquet/ + videos/) with task
metadata embedded — directly readable by data_loader.py.

Usage::

    python -m smolvla_visual_pilot.download_libero_clean \
        --datasets libero_object libero_spatial \
        --output-root libero_clean_data \
        --max-episodes 5

Requires: huggingface_hub, pyarrow, imageio[pyav], h5py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import h5py
import imageio.v3 as iio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from huggingface_hub import snapshot_download
except ImportError:
    raise SystemExit("huggingface_hub required: pip install huggingface_hub")

LIBERO_REPOS: dict[str, str] = {
    "libero_object":  "lerobot/libero_object",
    "libero_spatial": "lerobot/libero_spatial",
    "libero_goal":    "lerobot/libero_goal",
    "libero_10":      "lerobot/libero_10",
}

DEFAULT_CAMERAS = ["observation.images.image", "observation.images.wrist_image"]


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def load_tasks(meta_dir: Path) -> dict[int, str]:
    path = meta_dir / "tasks.parquet"
    if not path.exists():
        return {}
    result: dict[int, str] = {}
    pf = pq.ParquetFile(path)
    for rb in pf.iter_batches():
        d = rb.to_pydict()
        task_col = "task" if "task" in d else "__index_level_0__"
        for idx, task in zip(d["task_index"], d[task_col]):
            result[int(idx)] = str(task)
    return result

def load_episodes(meta_dir: Path) -> dict[int, dict]:
    # In v3.0, episodes metadata might be scattered across chunks.
    # Usually meta/episodes/chunk-000/file-000.parquet
    result: dict[int, dict] = {}
    ep_dir = meta_dir / "episodes"
    if not ep_dir.exists():
        return result
    for path in ep_dir.rglob("*.parquet"):
        pf = pq.ParquetFile(path)
        for rb in pf.iter_batches():
            d = rb.to_pydict()
            for i in range(len(d["episode_index"])):
                ep_idx = int(d["episode_index"][i])
                result[ep_idx] = {k: d[k][i] for k in d.keys()}
    return result

def resolve_task(episode_meta: dict, task_map: dict[int, str]) -> tuple[int, str]:
    """Return (task_index, instruction) for one episode."""
    if "task_index" in episode_meta:
        ti = int(episode_meta["task_index"])
        return ti, task_map.get(ti, "")
    tasks_list: list[str] = episode_meta.get("tasks", [])
    if tasks_list:
        instr = tasks_list[0]
        for ti, t in task_map.items():
            if t == instr:
                return ti, instr
        return -1, instr
    return -1, ""


# ---------------------------------------------------------------------------
# HDF5 writers
# ---------------------------------------------------------------------------

def _write_frames(video_path: Path, frame_indices: np.ndarray, cam_grp: h5py.Group,
                  batch: int = 64, compression: str | None = "lzf") -> int:
    ds: h5py.Dataset | None = None
    buf: list[np.ndarray] = []
    total = 0
    target_count = len(frame_indices)
    frame_set = set(frame_indices.tolist())
    
    def flush() -> None:
        nonlocal ds, total
        if not buf:
            return
        arr = np.stack(buf, axis=0)
        if ds is None:
            shape = arr.shape[1:]
            ds = cam_grp.create_dataset(
                "frames", shape=(0,) + shape, maxshape=(None,) + shape,
                dtype=arr.dtype, chunks=(min(batch, arr.shape[0]),) + shape,
                compression=compression,
            )
        old = ds.shape[0]
        new = old + arr.shape[0]
        ds.resize(new, axis=0)
        ds[old:new] = arr
        total = new
        buf.clear()

    # Fast forward through video and grab required frames
    for i, frame in enumerate(iio.imiter(video_path, plugin="pyav")):
        if i in frame_set:
            buf.append(frame)
            if len(buf) >= batch:
                flush()
        if i >= np.max(frame_indices):
            break
            
    flush()
    if ds is None:
        cam_grp.create_dataset("frames", shape=(0,), maxshape=(None,), dtype=np.uint8)
    return total

# ---------------------------------------------------------------------------
# Episode converter
# ---------------------------------------------------------------------------

def convert_episode(
    *,
    episode_data: dict[str, np.ndarray],
    video_paths: dict[str, Path],
    output_h5: Path,
    task_index: int,
    task_instruction: str,
    episode_index: int,
    video_frame_offset: int,
    compression: str | None = "lzf",
    frame_batch: int = 64,
) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    num_rows = len(episode_data["index"])

    with h5py.File(output_h5, "w") as h5:
        # parquet columns (observation.state, action, ...)
        pq_grp = h5.require_group("parquet")
        for col_name, arr in episode_data.items():
            if col_name == "frame_index":
                continue # we'll use this for videos, but can also save it
            tail = arr.shape[1:]
            ds = pq_grp.create_dataset(
                col_name, shape=(num_rows,) + tail, dtype=arr.dtype,
                chunks=(min(1024, max(1, num_rows)),) + tail, compression=compression,
            )
            ds[:] = arr

        # Embed task_index so data_loader.py can find it
        if "task_index" not in pq_grp:
            pq_grp.create_dataset(
                "task_index",
                data=np.full(num_rows, task_index, dtype=np.int32),
            )

        # videos
        vid_grp = h5.require_group("videos")
        frame_counts: dict[str, int] = {}
        frame_indices = episode_data.get("frame_index", np.arange(num_rows))
        
        for camera, vpath in sorted(video_paths.items()):
            if not vpath.exists():
                print(f"    WARNING: video missing — skipping camera '{camera}'")
                continue
            cam_grp = vid_grp.require_group(camera)
            abs_frame_indices = frame_indices + video_frame_offset
            nf = _write_frames(vpath, abs_frame_indices, cam_grp, frame_batch, compression)
            if nf != num_rows:
                raise RuntimeError(
                    f"Frame/parquet mismatch for '{camera}': {nf} frames extracted vs {num_rows} expected"
                )
            frame_counts[camera] = nf

        # root metadata
        h5.attrs["episode_index"] = episode_index
        h5.attrs["task_index"] = task_index
        h5.attrs["task_instruction"] = task_instruction
        h5.attrs["num_samples"] = num_rows
        h5.attrs["source"] = "libero_clean"
        h5.attrs["cameras"] = np.asarray(sorted(frame_counts.keys()), dtype="S")

        # meta sub-group
        meta_grp = h5.require_group("meta")
        meta_grp.attrs["task_index"] = task_index
        meta_grp.attrs["task_instruction"] = task_instruction
        meta_grp.attrs["episode_index"] = episode_index
        for cam, cnt in frame_counts.items():
            meta_grp.attrs[f"num_frames::{cam}"] = cnt


# ---------------------------------------------------------------------------
# Download + convert loop
# ---------------------------------------------------------------------------

def run(
    dataset_name: str,
    output_root: Path,
    cameras: list[str],
    max_episodes: int | None,
    overwrite: bool,
    hf_cache: Path | None,
    compression: str | None,
    frame_batch: int,
    parquet_batch: int,
) -> None:
    repo_id = LIBERO_REPOS[dataset_name]
    out_dir = output_root / dataset_name

    print(f"\n{'='*60}")
    print(f"Dataset : {repo_id}  ->  {out_dir}")
    print(f"Cameras : {cameras}")

    print("Downloading from HuggingFace Hub (cached after first run) …")
    local_dir = Path(snapshot_download(
        repo_id=repo_id, repo_type="dataset",
        cache_dir=str(hf_cache) if hf_cache else None,
        ignore_patterns=["*.git*"],
    ))
    print(f"  Local cache: {local_dir}")

    meta_dir = local_dir / "meta"
    task_map = load_tasks(meta_dir)
    episode_map = load_episodes(meta_dir)
    print(f"  Tasks: {len(task_map)}  |  Episodes in meta: {len(episode_map)}")

    data_dir = local_dir / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        print("  WARNING: No parquet files found.")
        return

    print(f"  Scanning parquet files to discover episodes ...")
    episodes_data = defaultdict(lambda: defaultdict(list))
    
    # Read all parquet files and group by episode_index
    for pq_path in parquet_files:
        pf = pq.ParquetFile(pq_path)
        col_names = pf.schema_arrow.names
        for rb in pf.iter_batches(batch_size=parquet_batch):
            d = rb.to_pydict()
            ep_indices = d["episode_index"]
            for i, ep_idx in enumerate(ep_indices):
                ep_idx = int(ep_idx)
                for col in col_names:
                    val = rb.column(col)[i]
                    t = rb.column(col).type
                    if pa.types.is_fixed_size_list(t):
                        arr = val.values.to_numpy(zero_copy_only=False)
                    elif pa.types.is_list(t) or pa.types.is_large_list(t):
                        arr = np.asarray(val.as_py())
                    else:
                        arr = val.as_py()
                    episodes_data[ep_idx][col].append(arr)

    episode_indices = sorted(episodes_data.keys())
    if max_episodes is not None:
        episode_indices = episode_indices[:max_episodes]

    print(f"  Converting {len(episode_indices)} episodes …")
    video_root = local_dir / "videos"
    processed = skipped = errors = 0
    
    # Pre-calculate absolute frame offsets within the monolithic chunk video
    ep_frame_offsets = {}
    current_offset = 0
    for ep_idx in sorted(episodes_data.keys()):
        ep_frame_offsets[ep_idx] = current_offset
        # Number of frames is the length of any column's list
        col_name = next(iter(episodes_data[ep_idx]))
        current_offset += len(episodes_data[ep_idx][col_name])

    # Chunk 000 is usually the standard, but we'll use it
    chunk_name = "chunk-000"

    for i, ep_idx in enumerate(episode_indices, 1):
        episode_stem = f"episode_{ep_idx:06d}"
        output_h5 = out_dir / chunk_name / f"{episode_stem}.h5"
        
        if output_h5.exists() and not overwrite:
            skipped += 1
            continue

        # Compile episode data into numpy arrays
        compiled_data = {}
        for col, val_list in episodes_data[ep_idx].items():
            compiled_data[col] = np.stack(val_list, axis=0)

        ep_meta = episode_map.get(ep_idx, {})
        task_index, task_instruction = resolve_task(ep_meta, task_map)
        if task_index == -1 and "task_index" in compiled_data:
            task_index = int(compiled_data["task_index"][0])
            task_instruction = task_map.get(task_index, "")

        # In v3.0, the videos are typically in video_root / camera / chunk-000 / file-000.mp4
        # and contain all frames. We can use the frame_index to slice them.
        video_paths = {
            cam: video_root / cam / chunk_name / "file-000.mp4"
            for cam in cameras
        }

        print(
            f"  [{i}/{len(episode_indices)}] {chunk_name}/{episode_stem} "
            f"task={task_index} '{task_instruction[:55]}'"
        )
        try:
            convert_episode(
                episode_data=compiled_data,
                video_paths=video_paths,
                output_h5=output_h5,
                task_index=task_index,
                task_instruction=task_instruction,
                episode_index=ep_idx,
                video_frame_offset=ep_frame_offsets[ep_idx],
                compression=compression,
                frame_batch=frame_batch,
            )
            processed += 1
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"    ERROR: {type(exc).__name__}: {exc}")
            errors += 1
            if output_h5.exists():
                output_h5.unlink()

    print(f"Done '{dataset_name}': processed={processed} skipped={skipped} errors={errors}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download clean LIBERO datasets and convert to h5 for SmolVLA evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--datasets", nargs="+",
        choices=list(LIBERO_REPOS) + ["all"],
        default=["libero_object"],
        help="Which LIBERO splits to download.",
    )
    p.add_argument("--output-root", type=Path, default=Path("libero_clean_data"),
                   help="Root directory for output h5 files.")
    p.add_argument("--cameras", nargs="+", default=DEFAULT_CAMERAS,
                   help="Camera keys to extract.")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Cap per dataset (useful for smoke-testing).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing h5 files.")
    p.add_argument("--hf-cache", type=Path, default=None,
                   help="HuggingFace cache dir (default: ~/.cache/huggingface).")
    p.add_argument("--compression", default="lzf",
                   help="HDF5 compression: lzf, gzip, or none.")
    p.add_argument("--frame-batch", type=int, default=64)
    p.add_argument("--parquet-batch", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets = list(LIBERO_REPOS) if "all" in args.datasets else list(args.datasets)
    compr: str | None = (
        None if args.compression.lower() in {"none", "null", "off", ""} else args.compression
    )
    for ds in datasets:
        run(
            dataset_name=ds,
            output_root=args.output_root,
            cameras=args.cameras,
            max_episodes=args.max_episodes,
            overwrite=args.overwrite,
            hf_cache=args.hf_cache,
            compression=compr,
            frame_batch=args.frame_batch,
            parquet_batch=args.parquet_batch,
        )
    print("\nAll done.")


if __name__ == "__main__":
    main()
