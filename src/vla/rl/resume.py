"""Full-state save/load for resumable SRPO training.

Writes every piece of state required for a bit-for-bit resume to
``<ckpt_dir>/latest/state.pt`` using an atomic tmp + rename.
Pairs with :func:`vla.rl.trainer.train_srpo` and :mod:`scripts.train_srpo`.

Design spec: ``src/vla/docs/training_resume.md``.
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla.rl.config import SRPOConfig
from vla.rl.rollout import Trajectory
from vla.rl.srpo_reward import (
    ClusterDiagnostics,
    MultiTaskWorldProgressReward,
)

logger = logging.getLogger(__name__)


LATEST_DIR_NAME = "latest"
LATEST_TMP_NAME = "latest.tmp"
STATE_FILENAME = "state.pt"
CONFIG_FILENAME = "config.json"
WANDB_ID_FILENAME = "wandb_run_id.txt"
SNAPSHOT_DIR_NAME = "snapshots"


DETERMINISM_FIELDS: tuple[str, ...] = (
    "seed",
    "lr",
    "max_grad_norm",
    "betas",
    "weight_decay",
    "update_method",
    "advantage_mode",
    "adv_eps",
    "adv_skip_threshold",
    "clip_epsilon",
    "clip_epsilon_high",
    "num_fm_noise_samples",
    "awr_epochs",
    "awr_temperature",
    "awr_weight_clip",
    "ppo_epochs",
    "ppo_minibatch_trajs",
    "fpo_full_chunk_target",
    "fpo_loss_reduction",
    "fpo_positive_adv_only",
    "fpo_negative_adv_scale",
    "fpo_log_ratio_clip",
    "fpo_use_ref_policy_kl",
    "sft_kl_coeff",
    "adaptive_kl",
    "kl_target",
    "kl_adapt_factor",
    "mode",
    "simulator",
    "suite",
    "task_id",
    "state_dim",
    "num_rollout_envs",
    "num_envs",
    "n_action_steps",
    "fm_batch_size",
    "max_steps",
    "world_model_type",
    "distance_metric",
    "subsample_every",
    "dbscan_eps",
    "dbscan_min_samples",
    "dbscan_auto_eps",
    "use_failure_rewards",
    "use_standard_scaler",
    "include_demos_in_update",
    "success_replay_buffer_size",
    "success_replay_total_size",
    "success_replay_alpha",
    "success_replay_ema_decay",
    "success_replay_max_ratio",
    "dynamic_sampling",
    "dynamic_sampling_max_retries",
)

EXTRA_DETERMINISM_KEYS: tuple[str, ...] = ("trajs_per_task_per_iter",)


def _capture_rng() -> dict[str, Any]:
    rng: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    return rng


def _restore_rng(rng: dict[str, Any]) -> None:
    torch.set_rng_state(rng["torch"])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    if torch.cuda.is_available() and "cuda" in rng:
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except (RuntimeError, AssertionError) as exc:
            logger.warning("Could not restore CUDA RNG (device mismatch?): %s", exc)


def _reward_model_state(reward_model: MultiTaskWorldProgressReward | None) -> dict[str, Any] | None:
    if reward_model is None:
        return None
    per_task: dict[str, dict[str, Any]] = {}
    for tid, rm in reward_model._per_task.items():
        per_task[tid] = {
            "demo_embeddings": [e.detach().cpu() for e in rm._demo_embeddings],
            "online_embeddings": [e.detach().cpu() for e in rm._online_embeddings],
            "cluster_centers": rm.cluster_centers.detach().cpu() if rm.cluster_centers is not None else None,
            "last_labels": list(rm._last_labels) if rm._last_labels is not None else None,
            "last_diagnostics": asdict(rm._last_diagnostics) if rm._last_diagnostics is not None else None,
        }
    return {"per_task": per_task, "config": asdict(reward_model.cfg)}


def _restore_reward_model(
    reward_model: MultiTaskWorldProgressReward | None,
    saved: dict[str, Any] | None,
) -> None:
    if reward_model is None or saved is None:
        return
    per_task = saved.get("per_task", {})
    for tid, snap in per_task.items():
        rm = reward_model._get_or_create(tid)
        rm._demo_embeddings = list(snap.get("demo_embeddings", []))
        rm._online_embeddings = list(snap.get("online_embeddings", []))
        rm.cluster_centers = snap.get("cluster_centers")
        rm._last_labels = list(snap.get("last_labels")) if snap.get("last_labels") is not None else None
        diag = snap.get("last_diagnostics")
        rm._last_diagnostics = ClusterDiagnostics(**diag) if diag is not None else None


def _config_to_dict(config: SRPOConfig) -> dict[str, Any]:
    raw = {f.name: getattr(config, f.name) for f in fields(config)}
    return {k: _serialise(v) for k, v in raw.items()}


def _serialise(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_serialise(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialise(v) for k, v in value.items()}
    if hasattr(value, "value") and hasattr(value, "name"):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _diff_config(saved: dict[str, Any], current: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    diffs: list[str] = []
    for key in keys:
        saved_v = saved.get(key, "<missing>")
        current_v = current.get(key, "<missing>")
        if saved_v != current_v:
            diffs.append(f"  {key}: saved={saved_v!r}  current={current_v!r}")
    return diffs


def build_config_snapshot(
    config: SRPOConfig,
    extras: dict[str, Any],
) -> dict[str, Any]:
    """Return the dict that is written to ``config.json`` on every save."""
    return {
        "config": _config_to_dict(config),
        "extras": {k: _serialise(v) for k, v in extras.items()},
    }


def assert_resume_compatible(
    saved: dict[str, Any],
    config: SRPOConfig,
    extras: dict[str, Any],
) -> None:
    """Compare saved config/extras against current run and raise on mismatch.

    Only fields in :data:`DETERMINISM_FIELDS` and
    :data:`EXTRA_DETERMINISM_KEYS` are compared.
    """
    current_cfg = _config_to_dict(config)
    saved_cfg = saved.get("config", {})
    diffs = _diff_config(saved_cfg, current_cfg, DETERMINISM_FIELDS)

    current_extras = {k: _serialise(v) for k, v in extras.items()}
    saved_extras = saved.get("extras", {})
    diffs.extend(_diff_config(saved_extras, current_extras, EXTRA_DETERMINISM_KEYS))

    if diffs:
        lines = "\n".join(diffs)
        raise RuntimeError(
            "Cannot resume: determinism-affecting config changed between runs.\n"
            f"Mismatched fields:\n{lines}\n"
            "Re-submit with the original flags or point --resume-from at a different directory."
        )


def latest_dir(ckpt_dir: Path | str) -> Path:
    return Path(ckpt_dir) / LATEST_DIR_NAME


def snapshot_dir(ckpt_dir: Path | str, iteration: int) -> Path:
    return Path(ckpt_dir) / SNAPSHOT_DIR_NAME / f"iter_{iteration:05d}"


def save_full_state(
    *,
    ckpt_dir: Path | str,
    policy: Any,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    best_success: float,
    best_rollout_successes: int,
    success_buffer: dict[str, list[Trajectory]],
    success_rate_ema: dict[str, float],
    reward_model: MultiTaskWorldProgressReward | None,
    config: SRPOConfig,
    extras: dict[str, Any],
    wandb_run_id: str | None,
    keep_every: int = 0,
) -> None:
    """Atomically write full trainer state to ``<ckpt_dir>/latest/``.

    When ``keep_every > 0`` and ``iteration`` is a multiple of it, also
    mirror the ``latest/`` tree to ``<ckpt_dir>/snapshots/iter_<N>/``.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tmp = ckpt_dir / LATEST_TMP_NAME
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    policy.save_checkpoint(tmp)

    state = {
        "schema_version": 1,
        "iteration": iteration,
        "best_success": best_success,
        "best_rollout_successes": best_rollout_successes,
        "optimizer": optimizer.state_dict(),
        "rng": _capture_rng(),
        "success_buffer": {tid: list(trajs) for tid, trajs in success_buffer.items()},
        "success_rate_ema": dict(success_rate_ema),
        "reward_model": _reward_model_state(reward_model),
        "kl_coeff": float(config.kl_coeff),
    }
    torch.save(state, tmp / STATE_FILENAME)

    snapshot = build_config_snapshot(config, extras)
    with open(tmp / CONFIG_FILENAME, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

    if wandb_run_id:
        (tmp / WANDB_ID_FILENAME).write_text(wandb_run_id + "\n")

    target = ckpt_dir / LATEST_DIR_NAME
    backup = ckpt_dir / f"{LATEST_DIR_NAME}.old"
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.rename(target, backup)
    os.rename(tmp, target)
    if backup.exists():
        shutil.rmtree(backup)

    if keep_every > 0 and iteration > 0 and iteration % keep_every == 0:
        snap = snapshot_dir(ckpt_dir, iteration)
        if snap.exists():
            shutil.rmtree(snap)
        shutil.copytree(target, snap)
        logger.info("Wrote iteration snapshot to %s", snap)


def _load_config_snapshot(ckpt_dir: Path) -> dict[str, Any] | None:
    path = ckpt_dir / LATEST_DIR_NAME / CONFIG_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _read_wandb_id(ckpt_dir: Path) -> str | None:
    path = ckpt_dir / LATEST_DIR_NAME / WANDB_ID_FILENAME
    if not path.exists():
        return None
    value = path.read_text().strip()
    return value or None


def has_resume_state(ckpt_dir: Path | str) -> bool:
    ckpt_dir = Path(ckpt_dir)
    latest = ckpt_dir / LATEST_DIR_NAME
    backup = ckpt_dir / f"{LATEST_DIR_NAME}.old"
    # Recover a mid-rename crash: the previous save renamed latest → latest.old
    # but was killed before the tmp → latest rename completed.
    if not latest.exists() and backup.exists():
        os.rename(backup, latest)
        logger.warning("Recovered %s from interrupted save (%s → %s).", latest, backup, latest)
    return (latest / STATE_FILENAME).exists()


def load_wandb_id(ckpt_dir: Path | str) -> str | None:
    """Return the stored W&B run id, if any.

    Used by the CLI before :func:`wandb.init` so the run resumes on the
    same id.
    """
    return _read_wandb_id(Path(ckpt_dir))


def load_config_snapshot(ckpt_dir: Path | str) -> dict[str, Any] | None:
    """Public entry point mirroring :func:`_load_config_snapshot`."""
    return _load_config_snapshot(Path(ckpt_dir))


class ResumeState:
    """In-memory view of the resume state loaded from ``latest/state.pt``.

    Kept lightweight so it can be passed around by value inside the
    trainer; attributes are applied one at a time where they are
    needed.
    """

    def __init__(self, raw: dict[str, Any]) -> None:
        self.iteration: int = int(raw.get("iteration", 0))
        self.best_success: float = float(raw.get("best_success", -1.0))
        self.best_rollout_successes: int = int(raw.get("best_rollout_successes", -1))
        self.optimizer: dict[str, Any] = raw.get("optimizer", {})
        self.rng: dict[str, Any] = raw.get("rng", {})
        self.success_buffer: dict[str, list[Trajectory]] = raw.get("success_buffer", {})
        self.success_rate_ema: dict[str, float] = raw.get("success_rate_ema", {})
        self.reward_model: dict[str, Any] | None = raw.get("reward_model")
        self.kl_coeff: float | None = raw.get("kl_coeff")


def load_full_state(ckpt_dir: Path | str) -> ResumeState:
    """Load ``<ckpt_dir>/latest/state.pt`` into a :class:`ResumeState`."""
    path = Path(ckpt_dir) / LATEST_DIR_NAME / STATE_FILENAME
    raw = torch.load(path, map_location="cpu", weights_only=False)
    return ResumeState(raw)


def apply_resume_state(
    state: ResumeState,
    *,
    policy: Any,
    optimizer: torch.optim.Optimizer,
    success_buffer: dict[str, list[Trajectory]],
    success_rate_ema: dict[str, float],
    reward_model: MultiTaskWorldProgressReward | None,
    config: SRPOConfig,
    ckpt_dir: Path | str,
) -> None:
    """Apply a loaded :class:`ResumeState` back into live trainer objects.

    Restores:
      - model weights (from ``latest/policy.pt``)
      - optimizer state
      - RNG
      - replay buffer and EMA (in place)
      - reward-model state
      - adaptive kl_coeff
    """
    latest = Path(ckpt_dir) / LATEST_DIR_NAME
    policy.load_checkpoint(latest)
    optimizer.load_state_dict(state.optimizer)
    _restore_rng(state.rng)

    success_buffer.clear()
    for tid, trajs in state.success_buffer.items():
        success_buffer[tid] = list(trajs)

    success_rate_ema.clear()
    success_rate_ema.update(state.success_rate_ema)

    _restore_reward_model(reward_model, state.reward_model)

    if state.kl_coeff is not None:
        config.kl_coeff = float(state.kl_coeff)


def verify_or_start_fresh(
    ckpt_dir: Path | str,
    config: SRPOConfig,
    extras: dict[str, Any],
) -> bool:
    """Check whether ``ckpt_dir`` has a resumable state.

    Returns ``True`` when a resume is possible *and* the determinism
    fields match.  Raises when state exists but fields mismatch.
    Returns ``False`` when no state exists (fresh run).
    """
    if not has_resume_state(ckpt_dir):
        logger.info("No resume state at %s/%s — starting fresh.", ckpt_dir, LATEST_DIR_NAME)
        return False

    snapshot = _load_config_snapshot(Path(ckpt_dir))
    if snapshot is None:
        logger.warning(
            "Found %s/%s/%s without accompanying %s — treating as fresh start.",
            ckpt_dir,
            LATEST_DIR_NAME,
            STATE_FILENAME,
            CONFIG_FILENAME,
        )
        return False

    assert_resume_compatible(snapshot, config, extras)
    return True


__all__ = [
    "LATEST_DIR_NAME",
    "SNAPSHOT_DIR_NAME",
    "STATE_FILENAME",
    "ResumeState",
    "apply_resume_state",
    "assert_resume_compatible",
    "build_config_snapshot",
    "has_resume_state",
    "latest_dir",
    "load_config_snapshot",
    "load_full_state",
    "load_wandb_id",
    "save_full_state",
    "snapshot_dir",
    "verify_or_start_fresh",
]
