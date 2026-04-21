from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

DATA_FILE_RE = re.compile(r"^data/chunk-\d+/episode_(\d+)\.parquet$")
VIDEO_FILE_RE = re.compile(r"^videos/chunk-\d+/(observation\.images\.[^/]+)/episode_(\d+)\.mp4$")
DEFAULT_VIDEO_STREAMS = [
    "observation.images.front",
    "observation.images.wrist",
]


def _as_int(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return _as_int(value[0], default=default)
    try:
        return int(value)
    except Exception:
        return default


def _build_episode_file_index(repo: str) -> dict[int, dict[str, Any]]:
    print(f"Indexing remote files for {repo} ...")
    files = list_repo_files(repo_id=repo, repo_type="dataset")
    index: dict[int, dict[str, Any]] = {}

    for path in files:
        data_match = DATA_FILE_RE.match(path)
        if data_match is not None:
            ep = int(data_match.group(1))
            index.setdefault(ep, {"data": None, "videos": {}})
            index[ep]["data"] = path
            continue

        video_match = VIDEO_FILE_RE.match(path)
        if video_match is not None:
            stream = str(video_match.group(1))
            ep = int(video_match.group(2))
            index.setdefault(ep, {"data": None, "videos": {}})
            index[ep]["videos"][stream] = path

    print(f"  Indexed {len(index)} episodes with data and/or videos.")
    return index


def _collect_episode_ids_by_group(
    repo: str,
    split: str,
    group_col: str,
    samples_per_group: int,
    max_groups: int | None,
) -> dict[str, list[int]]:
    print(f"Connecting to {repo} (streaming mode)...")
    dataset = cast(Any, load_dataset(repo, split=split, streaming=True))

    grouped: dict[str, set[int]] = defaultdict(set)
    print(
        f"Scanning for distinct '{group_col}' values to collect "
        f"{samples_per_group} episode(s) per group ...",
    )

    for row_any in dataset:
        row = cast(dict[str, Any], row_any)
        group_value = str(row.get(group_col, "default"))
        episode_index = _as_int(row.get("episode_index"), default=-1)
        if episode_index < 0:
            continue

        if group_value not in grouped and max_groups is not None and len(grouped) >= max_groups:
            continue

        episode_set = grouped[group_value]
        if len(episode_set) < samples_per_group and episode_index not in episode_set:
            episode_set.add(episode_index)
            print(
                f"  -> Selected episode {episode_index} "
                f"for {group_col}={group_value} ({len(episode_set)}/{samples_per_group})",
            )

        if max_groups is not None and len(grouped) >= max_groups:
            if all(len(v) >= samples_per_group for v in grouped.values()):
                break

    completed = {
        group: sorted(episodes)
        for group, episodes in grouped.items()
        if len(episodes) >= samples_per_group
    }

    if max_groups is not None and len(completed) < max_groups:
        print(
            f"[warn] Requested {max_groups} groups, but only found "
            f"{len(completed)} groups with {samples_per_group} episode(s).",
        )
    return completed


def _download_dataset_file(repo: str, remote_path: str, save_dir: Path, dry_run: bool) -> str:
    if dry_run:
        return str(save_dir / remote_path)

    local_path = hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        filename=remote_path,
        local_dir=str(save_dir),
    )
    return str(local_path)


def _download_episode_bundle(
    repo: str,
    save_dir: Path,
    episode_index: int,
    entry: dict[str, Any],
    video_streams: list[str] | None,
    dry_run: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "episode_index": episode_index,
        "data": None,
        "videos": {},
    }

    data_path = entry.get("data")
    if isinstance(data_path, str):
        result["data"] = _download_dataset_file(repo, data_path, save_dir, dry_run)
        print(f"    data: {data_path}")
    else:
        print(f"    [warn] missing data parquet for episode {episode_index}")

    available_videos: dict[str, str] = entry.get("videos", {})
    streams = list(video_streams) if video_streams is not None else sorted(available_videos.keys())
    for stream in streams:
        video_path = available_videos.get(stream)
        if video_path is None:
            print(f"    [warn] missing stream '{stream}' for episode {episode_index}")
            continue
        result["videos"][stream] = _download_dataset_file(repo, video_path, save_dir, dry_run)
        print(f"    video ({stream}): {video_path}")

    return result


def download_by_column(
    repo: str,
    split: str,
    save_dir: str,
    group_col: str,
    num_samples: int,
    max_groups: int | None = 7,
    video_streams: list[str] | None = None,
    dry_run: bool = False,
) -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    grouped_episodes = _collect_episode_ids_by_group(
        repo=repo,
        split=split,
        group_col=group_col,
        samples_per_group=num_samples,
        max_groups=max_groups,
    )
    if not grouped_episodes:
        raise RuntimeError("No episode groups were collected from the dataset stream.")

    episode_index = _build_episode_file_index(repo)

    print("Downloading metadata files ...")
    metadata_files = ["meta/tasks.jsonl", "meta/episodes.jsonl"]
    for meta_path in metadata_files:
        try:
            _download_dataset_file(repo, meta_path, save_path, dry_run)
            print(f"  -> {meta_path}")
        except Exception as exc:
            print(f"  [warn] Could not download {meta_path}: {exc}")

    manifest: dict[str, Any] = {
        "repo": repo,
        "split": split,
        "group_col": group_col,
        "samples_per_group": num_samples,
        "max_groups": max_groups,
        "video_streams": video_streams,
        "dry_run": dry_run,
        "groups": {},
    }

    for group_value in sorted(grouped_episodes.keys()):
        episodes = grouped_episodes[group_value]
        print(f"\n[group {group_col}={group_value}] Downloading {len(episodes)} episode bundle(s) ...")
        manifest_rows: list[dict[str, Any]] = []
        for ep in episodes:
            entry = episode_index.get(ep)
            if entry is None:
                print(f"  [warn] episode {ep} not found in indexed dataset files")
                manifest_rows.append({
                    "episode_index": ep,
                    "data": None,
                    "videos": {},
                    "missing": True,
                })
                continue

            bundle = _download_episode_bundle(
                repo=repo,
                save_dir=save_path,
                episode_index=ep,
                entry=entry,
                video_streams=video_streams,
                dry_run=dry_run,
            )
            manifest_rows.append(bundle)

        manifest["groups"][group_value] = manifest_rows

    manifest_path = save_path / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved manifest to {manifest_path}")


def download_by_configs(repo, split, save_dir, num_samples):
    """
    So the creator of libero plus decided it would be smart to use unsupported file type of zip, z01, z02, etc. for the dataset
    Therefore this form of downloading is just not possible so this function is practically useless, but I will keep it here for posterity 
    and in case they change the dataset structure in the future which would be smart, even huggingface does not like it.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    configs = get_dataset_config_names(repo)
    print(f"Found {len(configs)} dataset configs (noise types): {configs}")

    for config in configs:
        print(f"\nStreaming config: {config}...")
        dataset = cast(Any, load_dataset(repo, name=config, split=split, streaming=True))

        for i, row_any in enumerate(dataset.take(num_samples)):
            row = cast(dict[str, Any], row_any)
            parquet_path = save_path / f"noise_{config}_sample_{i}.parquet"
            df = pd.DataFrame([row])
            df.to_parquet(parquet_path)
            print(f"  -> Saved sample to {parquet_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a subset of trajectories per LIBERO-plus noise type.")

    parser.add_argument("--repo", type=str, default="Sylvest/libero_plus_lerobot",
                        help="Hugging Face dataset repo (e.g., Sylvest/libero_plus_lerobot)")
    parser.add_argument("--samples", type=int, default=1,
                        help="Number of episode bundles to download per group")
    parser.add_argument("--save_dir", type=str, default="./libero_demo_samples",
                        help="Directory where downloaded data/videos and manifest are stored")
    parser.add_argument("--mode", type=str, choices=["column", "config"], default="column",
                        help="'column' downloads full episode bundles. 'config' keeps legacy subset download behavior.")
    parser.add_argument("--group_col", type=str, default="task_index",
                        help="Grouping column in streamed rows (e.g., 'task_index').")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to pull from (usually 'train')")
    parser.add_argument("--max_groups", type=int, default=7,
                        help="Maximum number of groups to collect in column mode. Use 0 for no cap.")
    parser.add_argument("--video_streams", nargs="*", default=DEFAULT_VIDEO_STREAMS,
                        help=(
                            "Video stream folder names to download in column mode "
                            "(default: observation.images.front observation.images.wrist)."
                        ))
    parser.add_argument("--dry_run", action="store_true",
                        help="Plan and print downloads without fetching files.")

    args = parser.parse_args()

    max_groups = None if args.max_groups == 0 else args.max_groups

    if args.mode == "column":
        download_by_column(
            repo=args.repo,
            split=args.split,
            save_dir=args.save_dir,
            group_col=args.group_col,
            num_samples=args.samples,
            max_groups=max_groups,
            video_streams=args.video_streams,
            dry_run=args.dry_run,
        )
    else:
        download_by_configs(args.repo, args.split, args.save_dir, args.samples)

    print("\n Download complete!")