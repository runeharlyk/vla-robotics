"""Load Libero+ demonstrations from a combined h5 file.

Mirrors the ``_iter_demos`` helper in
``smolvla_language_pilot/multitask_diagnostic.py`` but additionally
yields ground-truth actions when the h5 file contains them.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np
import torch
from typing import cast


@dataclass
class Demo:
    """A single demonstration loaded from the h5 file."""

    task_index: int
    task_instruction: str
    images: torch.Tensor       # (T, C, H, W) or (T, V, C, H, W) float32 in [0, 1]
    states: torch.Tensor       # (T, state_dim)
    gt_actions: torch.Tensor | None  # (T, action_dim) or None if missing


def iter_demos(
    h5_path: str,
    max_tasks: int | None = None,
    max_demos: int | None = None,
    cameras: list[str] | None = None,
) -> list[Demo]:
    """Yield :class:`Demo` objects from a combined Libero(+) h5 file.

    Supported h5 layout A (legacy, matches
    ``smolvla_language_pilot/multitask_diagnostic.py``)::

        demonstrations/
            demo_0000/
                observations/cam0   (T, H, W, C) uint8
                states              (T, state_dim) float
                actions             (T, action_dim) float   [optional]
                attrs: task_index, task

    Supported h5 layout B (new episode format)::

        parquet/
            observation.state      (T, state_dim)
            action                 (T, action_dim) [optional]
            task_index             (T,)            [optional]
        videos/
            <camera>/
                frames             (T, H, W, C) uint8

    Parameters
    ----------
    h5_path : str
        Path to the h5 file.
    max_tasks : int, optional
        Stop after this many *distinct* tasks.
    max_demos : int, optional
        Stop after this many demos *per task*.
    cameras : list[str], optional
        Camera keys under ``videos/<camera>/frames`` for new-format files.
        If omitted, all available cameras in the file are used.
    """
    demos: list[Demo] = []
    seen_tasks: dict[int, int] = {}  # task_index -> demo count

    with h5py.File(h5_path, "r") as f:
        # ------------------------------------------------------------------
        # Layout A (legacy): demonstrations/demo_xxx/...
        # ------------------------------------------------------------------
        if "demonstrations" in f and isinstance(f["demonstrations"], h5py.Group):
            demo_group = cast(h5py.Group, f["demonstrations"])

            for demo_key in sorted(demo_group.keys()):
                demo_obj = demo_group[demo_key]
                if not isinstance(demo_obj, h5py.Group):
                    continue
                demo = demo_obj
                task_index = int(demo.attrs.get("task_index", -1))
                task_instruction = str(demo.attrs.get("task", "")).strip()

                # ----- task cap -----
                if task_index not in seen_tasks:
                    if max_tasks is not None and len(seen_tasks) >= max_tasks:
                        continue
                    seen_tasks[task_index] = 0

                # ----- per-task demo cap -----
                if max_demos is not None and seen_tasks[task_index] >= max_demos:
                    continue
                seen_tasks[task_index] += 1

                # ----- images -----
                raw = np.array(demo["observations/cam0"])  # (T, H, W, C)
                images = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255.0

                # ----- states -----
                states = torch.tensor(np.array(demo["states"]), dtype=torch.float32)

                # ----- ground-truth actions (optional) -----
                gt_actions: torch.Tensor | None = None
                if "actions" in demo:
                    gt_actions = torch.tensor(
                        np.array(demo["actions"]), dtype=torch.float32
                    )

                demos.append(
                    Demo(
                        task_index=task_index,
                        task_instruction=task_instruction,
                        images=images,
                        states=states,
                        gt_actions=gt_actions,
                    )
                )

            return demos

        # ------------------------------------------------------------------
        # Layout B (new): parquet/* and videos/<camera>/frames
        # ------------------------------------------------------------------
        if (
            "parquet" in f
            and isinstance(f["parquet"], h5py.Group)
            and "videos" in f
            and isinstance(f["videos"], h5py.Group)
        ):
            parquet = cast(h5py.Group, f["parquet"])
            videos = cast(h5py.Group, f["videos"])

            selected_cameras = cameras or sorted(videos.keys())
            selected_cameras = [c.strip() for c in selected_cameras if c.strip()]
            if not selected_cameras:
                raise ValueError("At least one camera must be selected for loading.")

            camera_groups: dict[str, h5py.Group] = {}
            for camera in selected_cameras:
                if camera not in videos or not isinstance(videos[camera], h5py.Group):
                    available = sorted(videos.keys())
                    raise KeyError(
                        f"Camera '{camera}' not found in h5. Available cameras: {available}"
                    )
                camera_group = cast(h5py.Group, videos[camera])
                if "frames" not in camera_group:
                    raise KeyError(
                        f"Dataset 'videos/{camera}/frames' missing in h5 file: {h5_path}"
                    )
                camera_groups[camera] = camera_group

            if "observation.state" not in parquet:
                raise KeyError(
                    f"Dataset 'parquet/observation.state' missing in h5 file: {h5_path}"
                )

            raw_views: list[np.ndarray] = []
            lengths: list[int] = []
            for camera in selected_cameras:
                raw = np.array(camera_groups[camera]["frames"])  # (T, H, W, C)
                if raw.ndim != 4:
                    raise RuntimeError(
                        f"Expected 4D frames for camera '{camera}', got shape {raw.shape}"
                    )
                raw_views.append(raw)
                lengths.append(raw.shape[0])

            if len(set(lengths)) != 1:
                raise RuntimeError(
                    f"Camera frame count mismatch in {h5_path}: {dict(zip(selected_cameras, lengths))}"
                )

            # (T, V, H, W, C) -> (T, V, C, H, W)
            stacked = np.stack(raw_views, axis=1)
            images = torch.from_numpy(stacked).permute(0, 1, 4, 2, 3).float() / 255.0

            states = torch.tensor(
                np.array(parquet["observation.state"]), dtype=torch.float32
            )

            gt_actions: torch.Tensor | None = None
            if "action" in parquet:
                gt_actions = torch.tensor(np.array(parquet["action"]), dtype=torch.float32)

            if len(images) != len(states):
                raise RuntimeError(
                    f"Length mismatch in {h5_path}: images={len(images)} vs states={len(states)}"
                )
            if gt_actions is not None and len(gt_actions) != len(images):
                raise RuntimeError(
                    f"Length mismatch in {h5_path}: actions={len(gt_actions)} vs images={len(images)}"
                )

            task_index = -1
            if "task_index" in parquet:
                task_index_arr = np.array(parquet["task_index"])
                if len(task_index_arr) > 0:
                    task_index = int(task_index_arr[0])

            demos.append(
                Demo(
                    task_index=task_index,
                    task_instruction=f"{' + '.join(selected_cameras)} rollout",
                    images=images,
                    states=states,
                    gt_actions=gt_actions,
                )
            )
            return demos

        raise KeyError(
            "Unsupported h5 layout. Expected either root 'demonstrations' or both 'parquet' and 'videos'."
        )
