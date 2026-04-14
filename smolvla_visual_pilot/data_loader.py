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


@dataclass
class Demo:
    """A single demonstration loaded from the h5 file."""

    task_index: int
    task_instruction: str
    images: torch.Tensor       # (T, C, H, W) float32 in [0, 1]
    states: torch.Tensor       # (T, state_dim)
    gt_actions: torch.Tensor | None  # (T, action_dim) or None if missing


def iter_demos(
    h5_path: str,
    max_tasks: int | None = None,
    max_demos: int | None = None,
) -> list[Demo]:
    """Yield :class:`Demo` objects from a combined Libero(+) h5 file.

    Expected h5 layout (matches the format used by
    ``smolvla_language_pilot/multitask_diagnostic.py``)::

        demonstrations/
            demo_0000/
                observations/cam0   (T, H, W, C) uint8
                states              (T, state_dim) float
                actions             (T, action_dim) float   [optional]
                attrs: task_index, task

    Parameters
    ----------
    h5_path : str
        Path to the h5 file.
    max_tasks : int, optional
        Stop after this many *distinct* tasks.
    max_demos : int, optional
        Stop after this many demos *per task*.
    """
    demos: list[Demo] = []
    seen_tasks: dict[int, int] = {}  # task_index -> demo count

    with h5py.File(h5_path, "r") as f:
        print(f.keys())
        demo_group = f["demonstrations"]

        for demo_key in sorted(demo_group.keys()):
            demo = demo_group[demo_key]
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
            raw = np.array(demo["observations/cam0"])            # (T, H, W, C)
            images = (
                torch.from_numpy(raw)
                .permute(0, 3, 1, 2)
                .float()
                / 255.0
            )  # (T, C, H, W)

            # ----- states -----
            states = torch.tensor(
                np.array(demo["states"]), dtype=torch.float32,
            )

            # ----- ground-truth actions (optional) -----
            gt_actions: torch.Tensor | None = None
            if "actions" in demo:
                gt_actions = torch.tensor(
                    np.array(demo["actions"]), dtype=torch.float32,
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
