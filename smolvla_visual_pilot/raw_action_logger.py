"""Raw per-dimension action recorder for the SmolVLA evaluation pipeline.

Records the model's raw predicted action vector at every timestep — one value
per robot degree of freedom — so that per-joint sensitivity to noise can be
analysed offline without having to re-run inference.

Output files
------------
One HDF5 file per evaluation run:

    <output_dir>/raw_actions_<run_tag>.h5

Layout::

    /episodes/
        <episode_key>/              e.g. "task0_ep0_none_s0"
            attrs:
                task_index          int
                task_name           str
                rollout_index       int
                source_h5           str
                noise_type          str
                noise_severity      int
                action_dim_names    str (JSON list of names)
                action_dim          int
            predicted_actions   (T, action_dim)   float32  — raw model output
            gt_actions          (T, action_dim)   float32  — ground truth (if available)
            abs_errors          (T, action_dim)   float32  — |pred - gt| per dim
            timesteps           (T,)              int32    — 0..T-1

Usage::

    from smolvla_visual_pilot.raw_action_logger import RawActionLogger, ACTION_DIM_NAMES

    logger = RawActionLogger("outputs/raw_actions_run1.h5")
    logger.record(
        episode_key="task0_ep0_gaussian_s2",
        task_index=0,
        task_name="pick up the ketchup and place it in the basket",
        rollout_index=0,
        source_h5="libero_clean_data/libero_object/chunk-000/episode_000000.h5",
        noise_type="gaussian",
        noise_severity=2,
        predicted_actions=pred_tensor,   # (T, 7) torch.Tensor or np.ndarray
        gt_actions=gt_tensor,            # (T, 7) or None
    )
    logger.close()
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

# ---------------------------------------------------------------------------
# SmolVLA / Franka-style action dimension names
# 7-DOF: 6 arm joints + gripper (index 6 = gripper open/close)
# ---------------------------------------------------------------------------

ACTION_DIM_NAMES: list[str] = [
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper",
]


def _to_numpy(tensor_or_array: torch.Tensor | np.ndarray | None) -> np.ndarray | None:
    if tensor_or_array is None:
        return None
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().float().numpy()
    return np.asarray(tensor_or_array, dtype=np.float32)


class RawActionLogger:
    """Records raw per-dimension action arrays to HDF5 for offline analysis.

    The file is opened in append mode so multiple calls to :meth:`record`
    accumulate episodes in the same file.  Call :meth:`close` when done
    (or use as a context manager).

    Parameters
    ----------
    output_path : str | Path
        Path to the output HDF5 file.
    dim_names : list[str], optional
        Human-readable names for each action dimension.
        Defaults to :data:`ACTION_DIM_NAMES` (7-DOF Franka style).
    compression : str | None
        HDF5 compression filter.  ``"lzf"`` is fast and lossless.
    """

    def __init__(
        self,
        output_path: str | Path,
        dim_names: list[str] | None = None,
        compression: str | None = "lzf",
    ) -> None:
        self._path = Path(output_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._dim_names = dim_names or ACTION_DIM_NAMES
        self._compression = compression
        self._h5: h5py.File = h5py.File(self._path, "a")
        if "episodes" not in self._h5:
            self._h5.require_group("episodes")
        self._h5.attrs.setdefault("action_dim_names", json.dumps(self._dim_names))

    # -- context manager --

    def __enter__(self) -> "RawActionLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Flush and close the HDF5 file."""
        if self._h5.id.valid:
            self._h5.flush()
            self._h5.close()
            print(f"RawActionLogger closed: {self._path}")

    # -- recording --

    def record(
        self,
        *,
        episode_key: str,
        task_index: int,
        task_name: str,
        rollout_index: int,
        source_h5: str,
        noise_type: str,
        noise_severity: int,
        predicted_actions: torch.Tensor | np.ndarray,
        gt_actions: torch.Tensor | np.ndarray | None = None,
        overwrite: bool = False,
    ) -> None:
        """Write one episode's raw actions to the HDF5 file.

        Parameters
        ----------
        episode_key : str
            Unique key for this episode/variant combination, e.g.
            ``"task0_ep0_gaussian_s2"``.  Must be unique per file unless
            *overwrite* is True.
        predicted_actions : (T, action_dim) array
            Raw model-predicted actions.
        gt_actions : (T, action_dim) array or None
            Ground-truth actions from the dataset (if available).
        overwrite : bool
            If True, delete an existing group with the same *episode_key*
            before writing.
        """
        eps_grp = self._h5["episodes"]

        if episode_key in eps_grp:
            if not overwrite:
                return   # silently skip already-written episodes
            del eps_grp[episode_key]

        ep_grp = eps_grp.require_group(episode_key)

        pred_np = _to_numpy(predicted_actions)
        gt_np   = _to_numpy(gt_actions)

        if pred_np is None:
            raise ValueError("predicted_actions must not be None")
        if pred_np.ndim != 2:
            raise ValueError(f"predicted_actions must be (T, action_dim), got {pred_np.shape}")

        T, action_dim = pred_np.shape

        # Align GT length
        if gt_np is not None:
            T = min(T, gt_np.shape[0])
            pred_np = pred_np[:T]
            gt_np   = gt_np[:T]
        else:
            pred_np = pred_np[:T]

        # Timestep indices
        timesteps = np.arange(T, dtype=np.int32)

        # Per-dimension absolute errors (raw difference, not just abs)
        abs_err: np.ndarray | None = None
        raw_err: np.ndarray | None = None
        if gt_np is not None:
            raw_err = pred_np - gt_np          # (T, action_dim)  signed
            abs_err = np.abs(raw_err)          # (T, action_dim)  unsigned

        # Write datasets
        cmp = self._compression
        ep_grp.create_dataset("predicted_actions", data=pred_np, dtype=np.float32, compression=cmp)
        ep_grp.create_dataset("timesteps", data=timesteps, dtype=np.int32)
        if gt_np is not None:
            ep_grp.create_dataset("gt_actions",     data=gt_np,   dtype=np.float32, compression=cmp)
            ep_grp.create_dataset("abs_errors",     data=abs_err, dtype=np.float32, compression=cmp)
            ep_grp.create_dataset("signed_errors",  data=raw_err, dtype=np.float32, compression=cmp)

        # Attributes
        ep_grp.attrs["task_index"]       = task_index
        ep_grp.attrs["task_name"]        = task_name
        ep_grp.attrs["rollout_index"]    = rollout_index
        ep_grp.attrs["source_h5"]        = source_h5
        ep_grp.attrs["noise_type"]       = noise_type
        ep_grp.attrs["noise_severity"]   = noise_severity
        ep_grp.attrs["action_dim"]       = action_dim
        ep_grp.attrs["action_dim_names"] = json.dumps(
            self._dim_names[:action_dim] if action_dim <= len(self._dim_names)
            else self._dim_names + [f"dim_{i}" for i in range(len(self._dim_names), action_dim)]
        )
        ep_grp.attrs["num_timesteps"]    = T
        ep_grp.attrs["has_gt"]           = (gt_np is not None)

        self._h5.flush()

    # -- summary helpers --

    def list_episodes(self) -> list[str]:
        """Return all recorded episode keys."""
        return sorted(self._h5["episodes"].keys())

    def load_episode(self, episode_key: str) -> dict:
        """Load a recorded episode as numpy arrays and metadata.

        Returns a dict with keys:
            - ``predicted_actions`` : (T, D) float32
            - ``gt_actions``        : (T, D) float32 or None
            - ``abs_errors``        : (T, D) float32 or None
            - ``signed_errors``     : (T, D) float32 or None
            - ``timesteps``         : (T,)   int32
            - ``attrs``             : dict of HDF5 attributes
        """
        ep = self._h5["episodes"][episode_key]
        return {
            "predicted_actions": np.array(ep["predicted_actions"]),
            "gt_actions":        np.array(ep["gt_actions"])     if "gt_actions"    in ep else None,
            "abs_errors":        np.array(ep["abs_errors"])     if "abs_errors"    in ep else None,
            "signed_errors":     np.array(ep["signed_errors"])  if "signed_errors" in ep else None,
            "timesteps":         np.array(ep["timesteps"]),
            "attrs":             dict(ep.attrs),
        }

    def per_dim_stats(
        self,
        noise_type: str | None = None,
    ) -> dict[str, dict]:
        """Aggregate per-dimension statistics across all matching episodes.

        Parameters
        ----------
        noise_type : str | None
            Filter to episodes with this noise_type.  ``None`` includes all.

        Returns
        -------
        dict
            Keys are dimension names.  Each value has ``mean``, ``std``,
            ``max``, ``n_timesteps``.
        """
        eps_grp = self._h5["episodes"]
        dim_buckets: dict[int, list[np.ndarray]] = {}

        for key in eps_grp:
            ep = eps_grp[key]
            if noise_type is not None and ep.attrs.get("noise_type") != noise_type:
                continue
            if "abs_errors" not in ep:
                continue
            arr = np.array(ep["abs_errors"])  # (T, D)
            for d in range(arr.shape[1]):
                dim_buckets.setdefault(d, []).append(arr[:, d])

        result: dict[str, dict] = {}
        for d, values in sorted(dim_buckets.items()):
            cat = np.concatenate(values)
            name = (
                self._dim_names[d]
                if d < len(self._dim_names)
                else f"dim_{d}"
            )
            result[name] = {
                "mean": float(cat.mean()),
                "std":  float(cat.std()),
                "max":  float(cat.max()),
                "n_timesteps": int(len(cat)),
            }
        return result
