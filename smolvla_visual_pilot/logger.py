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
    quality_delta_l2: float | None = None
    quality_ratio_l2: float | None = None


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class ResultLogger:
    """Accumulates :class:`TimestepRecord` objects and writes them out."""

    def __init__(self) -> None:
        self._records: list[TimestepRecord] = []
        self._per_dim_abs_by_noise: dict[str, list[np.ndarray]] = {}
        self._per_dim_rel_by_noise: dict[str, list[np.ndarray]] = {}

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
        quality_delta_l2: float | None = None,
        quality_ratio_l2: float | None = None,
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
                quality_delta_l2=quality_delta_l2,
                quality_ratio_l2=quality_ratio_l2,
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
        per_dim_abs_errors: np.ndarray | None = None,
        per_dim_rel_errors: np.ndarray | None = None,
        quality_delta_l2: list[float] | np.ndarray | None = None,
        quality_ratio_l2: list[float] | np.ndarray | None = None,
    ) -> None:
        """Convenience: log all timesteps for one (rollout, noise_variant) pair."""
        l2_arr = np.asarray(l2_distances, dtype=np.float64)
        rel_l2_arr = np.asarray(rel_l2_distances, dtype=np.float64)
        T = min(len(l2_arr), len(rel_l2_arr))

        quality_delta_arr: np.ndarray | None = None
        quality_ratio_arr: np.ndarray | None = None
        if quality_delta_l2 is not None:
            quality_delta_arr = np.asarray(quality_delta_l2, dtype=np.float64)
            T = min(T, len(quality_delta_arr))
        if quality_ratio_l2 is not None:
            quality_ratio_arr = np.asarray(quality_ratio_l2, dtype=np.float64)
            T = min(T, len(quality_ratio_arr))

        if (per_dim_abs_errors is None) != (per_dim_rel_errors is None):
            raise ValueError("per_dim_abs_errors and per_dim_rel_errors must be provided together")

        if per_dim_abs_errors is not None and per_dim_rel_errors is not None:
            per_dim_abs_arr = np.asarray(per_dim_abs_errors, dtype=np.float64)
            per_dim_rel_arr = np.asarray(per_dim_rel_errors, dtype=np.float64)
            if per_dim_abs_arr.shape != per_dim_rel_arr.shape:
                raise ValueError("per-dimension absolute and relative error arrays must match in shape")
            if per_dim_abs_arr.ndim != 2:
                raise ValueError("per-dimension arrays must have shape (T, action_dim)")

            T = min(T, per_dim_abs_arr.shape[0])
            self._per_dim_abs_by_noise.setdefault(noise_type, []).append(per_dim_abs_arr[:T])
            self._per_dim_rel_by_noise.setdefault(noise_type, []).append(per_dim_rel_arr[:T])

        for t in range(T):
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
                l2_distance=float(l2_arr[t]),
                rel_l2_distance=float(rel_l2_arr[t]),
                quality_delta_l2=(float(quality_delta_arr[t]) if quality_delta_arr is not None else None),
                quality_ratio_l2=(float(quality_ratio_arr[t]) if quality_ratio_arr is not None else None),
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
        "quality_delta_l2",
        "quality_ratio_l2",
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
                        "quality_delta_l2": ("" if r.quality_delta_l2 is None else f"{r.quality_delta_l2:.6f}"),
                        "quality_ratio_l2": ("" if r.quality_ratio_l2 is None else f"{r.quality_ratio_l2:.6f}"),
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
                    "timestep_quality_delta_l2": [],
                    "timestep_quality_ratio_l2": [],
                },
            )
            variant["timestep_l2"].append(r.l2_distance)
            variant["timestep_rel_l2"].append(r.rel_l2_distance)
            variant["timestep_quality_delta_l2"].append(r.quality_delta_l2)
            variant["timestep_quality_ratio_l2"].append(r.quality_ratio_l2)

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
        by_noise_quality_delta: dict[str, list[float]] = defaultdict(list)
        by_noise_quality_ratio: dict[str, list[float]] = defaultdict(list)
        by_task: dict[int, list[float]] = defaultdict(list)
        by_task_noise: dict[tuple[int, str], list[float]] = defaultdict(list)
        by_source_noise: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for r in self._records:
            by_noise[r.noise_type].append(r.l2_distance)
            by_noise_rel[r.noise_type].append(r.rel_l2_distance)
            if r.quality_delta_l2 is not None:
                by_noise_quality_delta[r.noise_type].append(r.quality_delta_l2)
            if r.quality_ratio_l2 is not None:
                by_noise_quality_ratio[r.noise_type].append(r.quality_ratio_l2)
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

        def _dim_stats(mats: list[np.ndarray]) -> dict:
            stacked = np.concatenate(mats, axis=0)
            return {
                "mean": [float(v) for v in stacked.mean(axis=0)],
                "std": [float(v) for v in stacked.std(axis=0)],
                "max": [float(v) for v in stacked.max(axis=0)],
                "n_timesteps": int(stacked.shape[0]),
                "action_dim": int(stacked.shape[1]),
            }

        return {
            "by_noise_type": {k: _stats(v) for k, v in sorted(by_noise.items())},
            "by_noise_type_relative": {k: _stats(v) for k, v in sorted(by_noise_rel.items())},
            "by_noise_type_quality_delta_l2": {k: _stats(v) for k, v in sorted(by_noise_quality_delta.items())},
            "by_noise_type_quality_ratio_l2": {k: _stats(v) for k, v in sorted(by_noise_quality_ratio.items())},
            "by_noise_type_per_action_dim_abs": {
                k: _dim_stats(v) for k, v in sorted(self._per_dim_abs_by_noise.items())
            },
            "by_noise_type_per_action_dim_relative": {
                k: _dim_stats(v) for k, v in sorted(self._per_dim_rel_by_noise.items())
            },
            "by_task": {str(k): _stats(v) for k, v in sorted(by_task.items())},
            "by_task_noise": {f"task{k[0]}_{k[1]}": _stats(v) for k, v in sorted(by_task_noise.items())},
            "by_source_h5_noise_type": {
                source_h5: {noise_type: _stats(values) for noise_type, values in sorted(by_noise_map.items())}
                for source_h5, by_noise_map in sorted(by_source_noise.items())
            },
        }
