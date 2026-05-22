from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files

from vla.constants import LIBERO_SUITES, PROJECT_ROOT


def _read_jsonl(repo_id: str, path: str) -> list[dict[str, Any]]:
    local_path = hf_hub_download(repo_id, path, repo_type="dataset")
    rows: list[dict[str, Any]] = []
    with open(local_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def _task_map(repo_id: str) -> dict[int, str]:
    files = set(list_repo_files(repo_id, repo_type="dataset"))
    if "meta/tasks.jsonl" in files:
        rows = _read_jsonl(repo_id, "meta/tasks.jsonl")
        return {int(r["task_index"]): str(r["task"]) for r in rows}
    if "meta/tasks.parquet" in files:
        ds = load_dataset(repo_id, data_files="meta/tasks.parquet", split="train")
        return {int(r["task_index"]): str(r["task"]) for r in ds}
    raise FileNotFoundError(f"No task metadata found for dataset {repo_id}")


def _as_int(v: Any, default: int = -1) -> int:
    if v is None:
        return default
    if isinstance(v, (list, tuple)):
        if not v:
            return default
        return _as_int(v[0], default=default)
    try:
        return int(v)
    except Exception:
        return default


def _episodes_rows(repo_id: str) -> list[dict[str, Any]]:
    files = set(list_repo_files(repo_id, repo_type="dataset"))
    if "meta/episodes.jsonl" in files:
        return _read_jsonl(repo_id, "meta/episodes.jsonl")
    if "meta/episodes/chunk-000/file-000.parquet" in files:
        ds = load_dataset(repo_id, data_files="meta/episodes/chunk-000/file-000.parquet", split="train")
        return [dict(r) for r in ds]
    raise FileNotFoundError(f"No episode metadata found for dataset {repo_id}")


def _episodes_by_task_and_file(repo_id: str) -> tuple[dict[int, list[int]], dict[int, int]]:
    rows = _episodes_rows(repo_id)
    by_task: dict[int, list[int]] = defaultdict(list)
    ep_to_file_idx: dict[int, int] = {}
    for r in rows:
        ep_idx = _as_int(r.get("episode_index"))
        if ep_idx < 0:
            continue
        task_idx = _as_int(r.get("task_index"))
        if task_idx < 0:
            task_idx = _as_int(r.get("stats/task_index/min"))
        if task_idx < 0:
            continue
        by_task[task_idx].append(ep_idx)
        file_idx = _as_int(r.get("data/file_index"))
        if file_idx >= 0:
            ep_to_file_idx[ep_idx] = file_idx
    return by_task, ep_to_file_idx


def _to_uint8_hwc(image_obj: Any) -> np.ndarray:
    if image_obj is None:
        raise ValueError("image is None")
    if hasattr(image_obj, "convert"):
        return np.asarray(image_obj.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image_obj)
    if arr.ndim != 3:
        raise ValueError(f"unexpected image shape: {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"unexpected image shape: {arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _select_task_rollouts(
    episodes_by_task: dict[int, list[int]],
    task_map: dict[int, str],
    num_tasks: int,
    num_rollouts: int,
    seed: int,
    task_selection: str = "first",
    task_indices: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    all_tasks = sorted(k for k, v in episodes_by_task.items() if v)
    rng = np.random.default_rng(seed)

    if task_indices is not None:
        chosen_tasks = []
        missing: list[int] = []
        for task_id in task_indices:
            if task_id in episodes_by_task and episodes_by_task[task_id]:
                chosen_tasks.append(task_id)
            else:
                missing.append(task_id)
        if missing:
            raise ValueError(f"requested task indices have no episodes: {missing}")
    else:
        if len(all_tasks) < num_tasks:
            raise ValueError(f"requested {num_tasks} tasks, but only found {len(all_tasks)} tasks with episodes")

        if task_selection == "first":
            chosen_tasks = all_tasks[:num_tasks]
        elif task_selection == "random":
            chosen_tasks = [int(x) for x in rng.choice(np.asarray(all_tasks), size=num_tasks, replace=False).tolist()]
        elif task_selection == "diverse":
            def _action_key(task_id: int) -> str:
                text = task_map.get(task_id, "").strip().lower()
                prefixes = [
                    "pick up", "open", "close", "turn on", "turn off",
                    "push", "pull", "stack", "move", "place", "put",
                ]
                for p in prefixes:
                    if text.startswith(p):
                        return p
                return text.split()[0] if text else "unknown"

            groups: dict[str, list[int]] = defaultdict(list)
            for task_id in all_tasks:
                groups[_action_key(task_id)].append(task_id)

            chosen_set: set[int] = set()
            action_keys = sorted(groups.keys())
            rng.shuffle(action_keys)

            for key in action_keys:
                if len(chosen_set) >= num_tasks:
                    break
                candidates = groups[key]
                pick = int(rng.choice(np.asarray(candidates), size=1, replace=False)[0])
                chosen_set.add(pick)

            if len(chosen_set) < num_tasks:
                remaining = [t for t in all_tasks if t not in chosen_set]
                need = num_tasks - len(chosen_set)
                extra = rng.choice(np.asarray(remaining), size=need, replace=False)
                chosen_set.update(int(x) for x in extra.tolist())

            chosen_tasks = sorted(chosen_set)
        else:
            raise ValueError(f"unknown task selection mode: {task_selection}")

    num_selected_tasks = len(chosen_tasks)
    if num_selected_tasks == 0:
        raise ValueError("no tasks selected")
    if num_rollouts < num_selected_tasks:
        raise ValueError(
            f"num_rollouts ({num_rollouts}) must be >= number of selected tasks ({num_selected_tasks})"
        )

    base = num_rollouts // num_selected_tasks
    rem = num_rollouts % num_selected_tasks

    chosen_episodes: list[int] = []
    for idx, task_id in enumerate(chosen_tasks):
        need = base + (1 if idx < rem else 0)
        candidates = episodes_by_task[task_id]
        if len(candidates) < need:
            raise ValueError(
                f"task {task_id} has only {len(candidates)} episodes, but {need} are required for current split"
            )
        picks = rng.choice(np.asarray(candidates), size=need, replace=False)
        chosen_episodes.extend(int(x) for x in picks.tolist())
    return chosen_tasks, sorted(chosen_episodes)


def export_suite(
    suite: str,
    num_tasks: int,
    num_rollouts: int,
    output_dir: Path,
    seed: int,
    task_selection: str,
    task_indices: list[int] | None,
) -> Path:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py is required for H5 export. Install it with: pip install h5py") from exc

    if suite not in LIBERO_SUITES:
        raise ValueError(f"unknown suite '{suite}', choose from: {list(LIBERO_SUITES.keys())}")

    repo_id = LIBERO_SUITES[suite]
    task_map = _task_map(repo_id)
    episodes_by_task, ep_to_file_idx = _episodes_by_task_and_file(repo_id)
    selected_tasks, selected_eps = _select_task_rollouts(
        episodes_by_task=episodes_by_task,
        task_map=task_map,
        num_tasks=num_tasks,
        num_rollouts=num_rollouts,
        seed=seed,
        task_selection=task_selection,
        task_indices=task_indices,
    )
    selected_set = set(selected_eps)

    print(f"Selected task indices ({suite}): {selected_tasks}")
    for task_id in selected_tasks:
        print(f"  - [{task_id}] {task_map.get(task_id, '')}")

    needed_file_indices = sorted({_as_int(ep_to_file_idx.get(ep)) for ep in selected_eps if ep in ep_to_file_idx})
    if not needed_file_indices:
        raise ValueError(f"could not resolve data file indices for selected episodes in {repo_id}")
    data_files = [f"data/chunk-000/file-{i:03d}.parquet" for i in needed_file_indices]
    ds = load_dataset(repo_id, data_files=data_files, split="train")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"libero_{suite}_tasks{num_tasks}_rollouts{num_rollouts}.h5"

    demos: dict[int, dict[str, Any]] = {}
    for row in ds:
        ep = int(row["episode_index"])
        if ep not in selected_set:
            continue
        task_idx = int(row.get("task_index", -1))
        action = np.asarray(row["action"], dtype=np.float32)
        state_raw = row.get("observation.state")
        state = np.asarray(state_raw, dtype=np.float32) if state_raw is not None else np.zeros((0,), dtype=np.float32)
        cam0 = _to_uint8_hwc(row["observation.images.image"])
        cam1_raw = row.get("observation.images.image2")
        if cam1_raw is None:
            cam1_raw = row.get("observation.images.wrist_image")
        cam1 = _to_uint8_hwc(cam1_raw) if cam1_raw is not None else cam0.copy()

        if ep not in demos:
            demos[ep] = {
                "episode_index": ep,
                "task_index": task_idx,
                "task": task_map.get(task_idx, ""),
                "actions": [],
                "states": [],
                "cam0": [],
                "cam1": [],
            }
        demos[ep]["actions"].append(action)
        demos[ep]["states"].append(state)
        demos[ep]["cam0"].append(cam0)
        demos[ep]["cam1"].append(cam1)

    with h5py.File(out_path, "w") as f:
        f.attrs["suite"] = suite
        f.attrs["repo_id"] = repo_id
        f.attrs["num_tasks"] = num_tasks
        f.attrs["num_rollouts"] = num_rollouts
        f.attrs["selected_task_indices"] = np.asarray(selected_tasks, dtype=np.int32)

        demos_group = f.create_group("demonstrations")
        for demo_idx, ep in enumerate(sorted(demos.keys())):
            data = demos[ep]
            grp = demos_group.create_group(f"demo_{demo_idx:05d}")
            grp.attrs["episode_index"] = int(data["episode_index"])
            grp.attrs["task_index"] = int(data["task_index"])
            grp.attrs["task"] = str(data["task"])

            actions = np.stack(data["actions"], axis=0).astype(np.float32)
            states = np.stack(data["states"], axis=0).astype(np.float32)
            cam0 = np.stack(data["cam0"], axis=0).astype(np.uint8)
            cam1 = np.stack(data["cam1"], axis=0).astype(np.uint8)

            grp.create_dataset("actions", data=actions, compression="gzip")
            grp.create_dataset("states", data=states, compression="gzip")
            obs = grp.create_group("observations")
            obs.create_dataset("cam0", data=cam0, compression="gzip")
            obs.create_dataset("cam1", data=cam1, compression="gzip")

    return out_path


def _split_total_across_suites(total: int, n_suites: int) -> list[int]:
    if total < 0:
        raise ValueError("total must be >= 0")
    if n_suites <= 0:
        return []
    base = total // n_suites
    rem = total % n_suites
    return [base + (1 if i < rem else 0) for i in range(n_suites)]


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["spatial", "object"],
        choices=list(LIBERO_SUITES.keys()),
        help="LIBERO suites to export",
    )
    parser.add_argument("--num-tasks", type=int, default=5, help="number of tasks per suite")
    parser.add_argument("--num-rollouts", type=int, default=50, help="number of episodes per suite")
    parser.add_argument(
        "--num-tasks-total",
        type=int,
        default=None,
        help="total number of tasks across all selected suites (overrides --num-tasks)",
    )
    parser.add_argument(
        "--num-rollouts-total",
        type=int,
        default=None,
        help="total number of episodes across all selected suites (overrides --num-rollouts)",
    )
    parser.add_argument(
        "--rollouts-per-task",
        type=int,
        default=None,
        help="fixed number of rollouts per selected task (overrides --num-rollouts and --num-rollouts-total)",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for episode sampling")
    parser.add_argument(
        "--task-selection",
        type=str,
        default="first",
        choices=["first", "random", "diverse"],
        help="How to choose tasks when --task-indices is not provided.",
    )
    parser.add_argument(
        "--task-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit task indices to export (overrides --num-tasks and --task-selection).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "h5" / "libero",
        help="output directory for H5 files",
    )
    args = parser.parse_args()

    suites = list(args.suites)
    n_suites = len(suites)

    if args.task_indices is not None and args.num_tasks_total is not None:
        raise ValueError("--task-indices cannot be combined with --num-tasks-total")

    if args.num_tasks_total is not None:
        if args.num_tasks_total < n_suites:
            raise ValueError(
                f"--num-tasks-total ({args.num_tasks_total}) must be >= number of suites ({n_suites})"
            )
        tasks_per_suite = _split_total_across_suites(args.num_tasks_total, n_suites)
    else:
        tasks_per_suite = [args.num_tasks] * n_suites

    if args.num_rollouts_total is not None:
        if args.num_rollouts_total < n_suites:
            raise ValueError(
                f"--num-rollouts-total ({args.num_rollouts_total}) must be >= number of suites ({n_suites})"
            )
        rollouts_per_suite = _split_total_across_suites(args.num_rollouts_total, n_suites)
    else:
        rollouts_per_suite = [args.num_rollouts] * n_suites

    if args.rollouts_per_task is not None:
        if args.rollouts_per_task <= 0:
            raise ValueError("--rollouts-per-task must be > 0")
        rollouts_per_suite = [task_count * args.rollouts_per_task for task_count in tasks_per_suite]

    for suite, suite_num_tasks, suite_num_rollouts in zip(suites, tasks_per_suite, rollouts_per_suite):
        if suite_num_tasks <= 0:
            print(f"skipping {suite}: assigned 0 tasks")
            continue
        if suite_num_rollouts <= 0:
            print(f"skipping {suite}: assigned 0 rollouts")
            continue
        print(
            f"\nPreparing suite '{suite}' with {suite_num_tasks} task(s) and {suite_num_rollouts} rollout(s)"
        )
        path = export_suite(
            suite=suite,
            num_tasks=suite_num_tasks,
            num_rollouts=suite_num_rollouts,
            output_dir=args.output_dir,
            seed=args.seed,
            task_selection=args.task_selection,
            task_indices=args.task_indices,
        )
        print(f"saved: {path}")


if __name__ == "__main__":
    main()