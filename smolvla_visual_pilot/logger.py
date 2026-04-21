"""Structured result logging to JSON and CSV.

Records per-timestep L2 distances for every (task, rollout, noise_variant)
combination, then serialises to disk for downstream analysis.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Single-record dataclass
# ---------------------------------------------------------------------------

@dataclass
class TimestepRecord:
    """One row in the flat results table."""

    task_index: int
    task_name: str
    rollout_index: int
    source_h5: str
    source_chunk: str
    source_episode: str
    noise_type: str
    noise_severity: int
    timestep: int
    l2_distance: float
    rel_l2_distance: float


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class ResultLogger:
    """Accumulates :class:`TimestepRecord` objects and writes them out."""

    def __init__(self) -> None:
        self._records: list[TimestepRecord] = []

    # -- recording --

    def log_timestep(
        self,
        task_index: int,
        task_name: str,
        rollout_index: int,
        source_h5: str,
        source_chunk: str,
        source_episode: str,
        noise_type: str,
        noise_severity: int,
        timestep: int,
        l2_distance: float,
        rel_l2_distance: float,
    ) -> None:
        self._records.append(
            TimestepRecord(
                task_index=task_index,
                task_name=task_name,
                rollout_index=rollout_index,
                source_h5=source_h5,
                source_chunk=source_chunk,
                source_episode=source_episode,
                noise_type=noise_type,
                noise_severity=noise_severity,
                timestep=timestep,
                l2_distance=l2_distance,
                rel_l2_distance=rel_l2_distance,
            )
        )

    def log_trajectory(
        self,
        task_index: int,
        task_name: str,
        rollout_index: int,
        source_h5: str,
        source_chunk: str,
        source_episode: str,
        noise_type: str,
        noise_severity: int,
        l2_distances: list[float] | np.ndarray,
        rel_l2_distances: list[float] | np.ndarray,
    ) -> None:
        """Convenience: log all timesteps for one (rollout, noise_variant) pair."""
        for t, (l2, rel_l2) in enumerate(zip(l2_distances, rel_l2_distances, strict=True)):
            self.log_timestep(
                task_index=task_index,
                task_name=task_name,
                rollout_index=rollout_index,
                source_h5=source_h5,
                source_chunk=source_chunk,
                source_episode=source_episode,
                noise_type=noise_type,
                noise_severity=noise_severity,
                timestep=t,
                l2_distance=float(l2),
                rel_l2_distance=float(rel_l2),
            )

    @property
    def n_records(self) -> int:
        return len(self._records)

    # -- serialisation: CSV --

    _CSV_COLUMNS = [
        "task_index",
        "task_name",
        "rollout_index",
        "source_h5",
        "source_chunk",
        "source_episode",
        "noise_type",
        "noise_severity",
        "timestep",
        "l2_distance",
        "rel_l2_distance",
    ]

    def save_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._CSV_COLUMNS)
            writer.writeheader()
            for r in self._records:
                writer.writerow(
                    {
                        "task_index": r.task_index,
                        "task_name": r.task_name,
                        "rollout_index": r.rollout_index,
                        "source_h5": r.source_h5,
                        "source_chunk": r.source_chunk,
                        "source_episode": r.source_episode,
                        "noise_type": r.noise_type,
                        "noise_severity": r.noise_severity,
                        "timestep": r.timestep,
                        "l2_distance": f"{r.l2_distance:.6f}",
                        "rel_l2_distance": f"{r.rel_l2_distance:.6f}",
                    }
                )
        print(f"Saved CSV → {path}  ({len(self._records)} rows)")

    # -- serialisation: JSON (hierarchical) --

    def save_json(self, path: str | Path) -> None:
        """Write a hierarchical JSON: tasks → rollouts → noise_variants → timestep L2s."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # group records
        tasks: dict[int, dict] = {}
        for r in self._records:
            task = tasks.setdefault(
                r.task_index,
                {"task_index": r.task_index, "task_name": r.task_name, "rollouts": {}},
            )
            rollout = task["rollouts"].setdefault(
                r.rollout_index,
                {
                    "rollout_index": r.rollout_index,
                    "source_h5": r.source_h5,
                    "source_chunk": r.source_chunk,
                    "source_episode": r.source_episode,
                    "noise_variants": {},
                },
            )
            nv_key = f"{r.noise_type}_s{r.noise_severity}"
            variant = rollout["noise_variants"].setdefault(
                nv_key,
                {
                    "noise_type": r.noise_type,
                    "severity": r.noise_severity,
                    "timestep_l2": [],
                    "timestep_rel_l2": [],
                },
            )
            variant["timestep_l2"].append(r.l2_distance)
            variant["timestep_rel_l2"].append(r.rel_l2_distance)

        # convert dict-of-dicts to lists
        payload = {
            "tasks": [
                {
                    **{k: v for k, v in task_data.items() if k != "rollouts"},
                    "rollouts": [
                        {
                            **{k: v for k, v in ro.items() if k != "noise_variants"},
                            "noise_variants": list(ro["noise_variants"].values()),
                        }
                        for ro in task_data["rollouts"].values()
                    ],
                }
                for task_data in tasks.values()
            ]
        }

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON → {path}")

    # -- summary statistics --

    def get_summary(self) -> dict:
        """Return aggregate statistics per noise type and per task."""
        from collections import defaultdict

        by_noise: dict[str, list[float]] = defaultdict(list)
        by_noise_rel: dict[str, list[float]] = defaultdict(list)
        by_task: dict[int, list[float]] = defaultdict(list)
        by_task_noise: dict[tuple[int, str], list[float]] = defaultdict(list)
        by_source_noise: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for r in self._records:
            by_noise[r.noise_type].append(r.l2_distance)
            by_noise_rel[r.noise_type].append(r.rel_l2_distance)
            by_task[r.task_index].append(r.l2_distance)
            by_task_noise[(r.task_index, r.noise_type)].append(r.l2_distance)
            by_source_noise[r.source_h5][r.noise_type].append(r.l2_distance)

        def _stats(vals: list[float]) -> dict:
            a = np.array(vals)
            return {
                "mean": float(a.mean()),
                "std": float(a.std()),
                "max": float(a.max()),
                "n_timesteps": len(vals),
            }

        return {
            "by_noise_type": {k: _stats(v) for k, v in sorted(by_noise.items())},
            "by_noise_type_relative": {k: _stats(v) for k, v in sorted(by_noise_rel.items())},
            "by_task": {str(k): _stats(v) for k, v in sorted(by_task.items())},
            "by_task_noise": {f"task{k[0]}_{k[1]}": _stats(v) for k, v in sorted(by_task_noise.items())},
            "by_source_h5_noise_type": {
                source_h5: {noise_type: _stats(values) for noise_type, values in sorted(by_noise_map.items())}
                for source_h5, by_noise_map in sorted(by_source_noise.items())
            },
        }
