import argparse
from dataclasses import dataclass
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEOS_ROOT = PROJECT_ROOT / "libero_demo_samples" / "videos"
DEFAULT_PARQUET_ROOT = PROJECT_ROOT / "libero_demo_samples" / "data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "libero_demo_samples" / "combined_data"
DEFAULT_CAMERAS = ["observation.images.front", "observation.images.wrist"]


@dataclass(frozen=True)
class EpisodePair:
    key: str
    chunk: str
    episode: str
    video_paths: dict[str, Path]
    parquet_path: Path


def arrow_column_to_numpy(array: pa.Array) -> np.ndarray:
    """Convert an Arrow array to a dense NumPy array suitable for HDF5."""
    arrow_type = array.type

    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        data = np.asarray(array.to_pylist())
        if data.dtype == object:
            raise TypeError(
                f"Ragged list column is not supported for HDF5 dense storage: {arrow_type}"
            )
        return data

    if pa.types.is_fixed_size_list(arrow_type):
        return array.values.to_numpy(zero_copy_only=False).reshape(
            len(array), arrow_type.list_size
        )

    return array.to_numpy(zero_copy_only=False)


def write_parquet_columns_to_h5(
    parquet_path: Path,
    target_group: h5py.Group,
    batch_size: int = 1024,
    compression: str | None = "lzf",
) -> int:
    """Stream parquet columns to HDF5 datasets using row batches."""
    parquet_file = pq.ParquetFile(parquet_path)
    num_rows = parquet_file.metadata.num_rows
    # Use schema_arrow names (e.g. observation.state, action) to avoid
    # duplicate leaf names such as "element" from list-typed parquet fields.
    column_names = parquet_file.schema_arrow.names

    datasets: dict[str, h5py.Dataset] = {}
    rows_written = 0

    for record_batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_rows = record_batch.num_rows

        for col_idx, col_name in enumerate(column_names):
            arr = arrow_column_to_numpy(record_batch.column(col_idx))

            if col_name not in datasets:
                tail_shape = arr.shape[1:]
                chunk_rows = min(batch_size, max(1, num_rows))
                chunks = (chunk_rows,) + tail_shape
                datasets[col_name] = target_group.create_dataset(
                    col_name,
                    shape=(num_rows,) + tail_shape,
                    dtype=arr.dtype,
                    chunks=chunks,
                    compression=compression,
                )

            datasets[col_name][rows_written : rows_written + batch_rows] = arr

        rows_written += batch_rows

    if rows_written != num_rows:
        raise RuntimeError(
            f"Parquet rows mismatch: expected {num_rows}, wrote {rows_written}"
        )

    return num_rows


def write_video_frames_to_h5(
    video_path: Path,
    target_group: h5py.Group,
    frame_batch_size: int = 64,
    compression: str | None = "lzf",
) -> int:
    """Stream video frames to an extendable HDF5 dataset in mini-batches."""
    frame_ds = None
    frame_buffer = []
    frame_count = 0

    def flush_buffer() -> None:
        nonlocal frame_ds, frame_buffer, frame_count
        if not frame_buffer:
            return

        batch = np.stack(frame_buffer, axis=0)
        if frame_ds is None:
            frame_shape = batch.shape[1:]
            chunks = (min(frame_batch_size, batch.shape[0]),) + frame_shape
            frame_ds = target_group.create_dataset(
                "frames",
                shape=(0,) + frame_shape,
                maxshape=(None,) + frame_shape,
                dtype=batch.dtype,
                chunks=chunks,
                compression=compression,
            )

        old_size = frame_ds.shape[0]
        new_size = old_size + batch.shape[0]
        frame_ds.resize(new_size, axis=0)
        frame_ds[old_size:new_size] = batch
        frame_count = new_size
        frame_buffer = []

    for frame in iio.imiter(video_path, plugin="pyav"):
        frame_buffer.append(frame)
        if len(frame_buffer) >= frame_batch_size:
            flush_buffer()

    flush_buffer()

    if frame_ds is None:
        target_group.create_dataset("frames", shape=(0,), maxshape=(None,), dtype=np.uint8)

    return frame_count


def combine_episode_to_h5(
    video_paths: dict[str, Path],
    parquet_path: Path,
    output_path: Path,
    parquet_batch_size: int = 1024,
    frame_batch_size: int = 64,
    compression: str | None = "lzf",
) -> None:
    """Write frame-aligned multi-camera video + parquet data into one HDF5 file."""
    for camera_name, camera_path in sorted(video_paths.items()):
        print(f"Video ({camera_name}): {camera_path}")
    print(f"Parquet: {parquet_path}")
    print(f"Output:  {output_path}")

    with h5py.File(output_path, "w") as h5_file:
        parquet_group = h5_file.require_group("parquet")
        num_rows = write_parquet_columns_to_h5(
            parquet_path=parquet_path,
            target_group=parquet_group,
            batch_size=parquet_batch_size,
            compression=compression,
        )

        videos_group = h5_file.require_group("videos")
        frame_counts: dict[str, int] = {}
        for camera_name, camera_path in sorted(video_paths.items()):
            camera_group = videos_group.require_group(camera_name)
            num_frames = write_video_frames_to_h5(
                video_path=camera_path,
                target_group=camera_group,
                frame_batch_size=frame_batch_size,
                compression=compression,
            )
            if num_rows != num_frames:
                raise RuntimeError(
                    f"Alignment mismatch for camera '{camera_name}': "
                    f"{num_frames} frames vs {num_rows} parquet rows"
                )
            frame_counts[camera_name] = num_frames

        h5_file.attrs["num_samples"] = num_rows
        h5_file.attrs["alignment"] = "videos/<camera>/frames[i] <-> parquet/*[i]"
        h5_file.attrs["cameras"] = np.asarray(sorted(video_paths.keys()), dtype="S")
        meta_group = h5_file.require_group("meta")
        for camera_name, count in frame_counts.items():
            meta_group.attrs[f"num_frames::{camera_name}"] = count

    print(f"Done. Wrote {num_rows} aligned samples.")


def parse_csv_tokens(raw_values: list[str]) -> list[str]:
    tokens = []
    for value in raw_values:
        for token in value.split(","):
            token = token.strip()
            if token:
                tokens.append(token)
    return tokens


def parse_episode_selectors(raw_values: list[str]) -> list[str]:
    selectors = parse_csv_tokens(raw_values)
    return selectors or ["all"]


def parse_camera_selectors(raw_values: list[str]) -> list[str]:
    cameras = parse_csv_tokens(raw_values)
    if not cameras:
        raise ValueError("At least one camera must be provided.")
    return cameras


def normalize_compression(value: str) -> str | None:
    if value.lower() in {"none", "null", "off", ""}:
        return None
    return value


def discover_episode_pairs(
    parquet_root: Path,
    videos_root: Path,
    cameras: list[str],
) -> tuple[dict[str, EpisodePair], dict[str, list[str]], dict[str, list[str]]]:
    parquet_map = {
        f"{path.parent.name}/{path.stem}": path
        for path in parquet_root.glob("chunk-*/episode_*.parquet")
    }
    parquet_keys = set(parquet_map.keys())

    video_maps: dict[str, dict[str, Path]] = {}
    missing_videos_by_camera: dict[str, list[str]] = {}
    extra_videos_by_camera: dict[str, list[str]] = {}
    common_key_set = set(parquet_keys)

    for camera in cameras:
        camera_video_map = {
            f"{path.parent.parent.name}/{path.stem}": path
            for path in videos_root.glob(f"chunk-*/{camera}/episode_*.mp4")
        }
        camera_keys = set(camera_video_map.keys())
        video_maps[camera] = camera_video_map
        missing_videos_by_camera[camera] = sorted(parquet_keys - camera_keys)
        extra_videos_by_camera[camera] = sorted(camera_keys - parquet_keys)
        common_key_set &= camera_keys

    common_keys = sorted(common_key_set)

    pairs = {
        key: EpisodePair(
            key=key,
            chunk=key.split("/")[0],
            episode=key.split("/")[1],
            video_paths={camera: video_maps[camera][key] for camera in cameras},
            parquet_path=parquet_map[key],
        )
        for key in common_keys
    }
    return pairs, missing_videos_by_camera, extra_videos_by_camera


def select_episode_keys(
    available_keys: list[str],
    selectors: list[str],
) -> tuple[list[str], list[str]]:
    normalized = [s.strip() for s in selectors if s.strip()]
    if not normalized or any(s.lower() == "all" for s in normalized):
        return available_keys, []

    full_key_filters = {s for s in normalized if "/" in s}
    episode_filters = {s for s in normalized if "/" not in s}

    selected = [
        key
        for key in available_keys
        if key in full_key_filters or key.split("/")[1] in episode_filters
    ]

    missing = [
        token
        for token in normalized
        if (
            token not in available_keys
            and token not in {key.split("/")[1] for key in selected}
        )
    ]
    return selected, missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine Libero+ video and parquet episodes into aligned HDF5 files."
    )
    parser.add_argument(
        "--videos-root",
        type=Path,
        default=DEFAULT_VIDEOS_ROOT,
        help="Root path containing chunk-*/<camera>/episode_*.mp4",
    )
    parser.add_argument(
        "--parquet-root",
        type=Path,
        default=DEFAULT_PARQUET_ROOT,
        help="Root path containing chunk-*/episode_*.parquet",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory where output files are written as chunk-*/episode_*.h5",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        type=str,
        default=DEFAULT_CAMERAS,
        help=(
            "Camera subdirectories under each chunk. Default uses both front and wrist. "
            "Supports space-separated and comma-separated values."
        ),
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        default=["all"],
        help=(
            "Which episodes to process. Use 'all' (default), a list of episode IDs "
            "like episode_000000, or full keys like chunk-000/episode_000000. "
            "Comma-separated values are also supported."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files. By default, existing files are skipped.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered/selected episodes without writing output files.",
    )
    parser.add_argument(
        "--parquet-batch-size",
        type=int,
        default=2048,
        help="Number of parquet rows to read per batch.",
    )
    parser.add_argument(
        "--frame-batch-size",
        type=int,
        default=128,
        help="Number of video frames to read per batch.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="lzf",
        help="HDF5 compression: lzf, gzip, or none.",
    )
    args = parser.parse_args()

    selectors = parse_episode_selectors(args.episodes)
    camera_raw_values = [args.camera] if args.camera else args.cameras
    cameras = parse_camera_selectors(camera_raw_values)
    compression = normalize_compression(args.compression)

    pairs, missing_videos_by_camera, extra_videos_by_camera = discover_episode_pairs(
        parquet_root=args.parquet_root,
        videos_root=args.videos_root,
        cameras=cameras,
    )
    available_keys = sorted(pairs.keys())
    selected_keys, missing = select_episode_keys(available_keys, selectors)

    print(
        f"Found {len(available_keys)} matching episode pairs for cameras: "
        f"{', '.join(cameras)}"
    )
    for camera in cameras:
        missing_for_camera = missing_videos_by_camera[camera]
        extra_for_camera = extra_videos_by_camera[camera]
        if missing_for_camera:
            print(
                f"Missing {camera} video for {len(missing_for_camera)} parquet episodes."
            )
        if extra_for_camera:
            print(
                f"Found {len(extra_for_camera)} {camera} videos without parquet rows."
            )
    if missing:
        print(f"Warning: requested selectors not found: {missing}")

    if not selected_keys:
        raise SystemExit("No matching episodes selected. Nothing to do.")

    print(f"Selected {len(selected_keys)} episodes to process.")
    if args.dry_run:
        for key in selected_keys:
            pair = pairs[key]
            videos_text = ", ".join(
                f"{camera}={pair.video_paths[camera].name}" for camera in cameras
            )
            print(
                f"DRY RUN -> {key}: {videos_text}, parquet={pair.parquet_path.name}"
            )
        raise SystemExit(0)

    args.output_root.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0

    for key in selected_keys:
        pair = pairs[key]
        output_path = args.output_root / pair.chunk / f"{pair.episode}.h5"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing file: {output_path}")
            skipped += 1
            continue

        combine_episode_to_h5(
            video_paths=pair.video_paths,
            parquet_path=pair.parquet_path,
            output_path=output_path,
            parquet_batch_size=args.parquet_batch_size,
            frame_batch_size=args.frame_batch_size,
            compression=compression,
        )
        processed += 1

    print(f"Summary: processed={processed}, skipped={skipped}, total_selected={len(selected_keys)}")