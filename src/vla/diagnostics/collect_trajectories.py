"""Trajectory collection utilities for clustering analysis.

Provides functions to collect and cache trajectories from three sources:

1. **Demo trajectories** — loaded from the HuggingFace LeRobot dataset.
2. **SFT rollouts** — collected using the SmolVLA SFT checkpoint, split
   into success and failure buffers.
3. **Random-action rollouts** — collected with a random policy.

All buffers are cached to ``.pt`` files for fast re-loading.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch

from vla.data.libero import LiberoSFTDataset
from vla.rl.rollout import Trajectory

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────


@dataclass
class CollectionConfig:
    """Configuration for trajectory collection."""

    checkpoint: str = "HuggingFaceVLA/smolvla_libero"
    libero_suite: str = "spatial"
    task_id: int = 5
    action_dim: int = 7
    state_dim: int = 8
    num_demos: int = 100
    num_rollouts: int = 100
    num_envs: int = 4
    max_steps: int = 300
    seed: int = 42
    cache_dir: Path = field(default_factory=lambda: Path("notebooks/cache"))

    @property
    def task_key(self) -> str:
        return f"{self.libero_suite}_task_{self.task_id}"


# ──────────────────────────────────────────────────────────────────────
# Cache helpers
# ──────────────────────────────────────────────────────────────────────


def _cache_path(cfg: CollectionConfig, name: str) -> Path:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    return cfg.cache_dir / f"{cfg.libero_suite}_task{cfg.task_id}_{name}.pt"


def save_trajectories(trajs: list[Trajectory], cfg: CollectionConfig, name: str) -> None:
    """Save a list of trajectories to a cache file."""
    path = _cache_path(cfg, name)
    torch.save(trajs, path)
    logger.info("Saved %d trajectories → %s", len(trajs), path)


def load_trajectories(cfg: CollectionConfig, name: str) -> list[Trajectory] | None:
    """Load cached trajectories, or return ``None`` if the cache doesn't exist."""
    path = _cache_path(cfg, name)
    if path.exists():
        trajs = torch.load(path, weights_only=False)
        logger.info("Loaded %d trajectories from %s", len(trajs), path)
        return trajs
    return None


# ──────────────────────────────────────────────────────────────────────
# Collection functions
# ──────────────────────────────────────────────────────────────────────


def collect_demo_trajectories(cfg: CollectionConfig) -> list[Trajectory]:
    """Load demonstration trajectories from HuggingFace (cached to disk)."""
    cached = load_trajectories(cfg, "demos")
    if cached is not None:
        return cached

    logger.info("Loading demo trajectories from HuggingFace...")
    ds = LiberoSFTDataset(
        suite=cfg.libero_suite,
        num_demos=cfg.num_demos,
        seed=cfg.seed,
        task_id=cfg.task_id,
    )
    trajs = ds.episodes_as_trajectories(task_id=cfg.task_id)
    for t in trajs:
        t.task_id = cfg.task_key

    save_trajectories(trajs, cfg, "demos")
    return trajs


def _build_rollout_engine(cfg: CollectionConfig):
    """Create a vectorised LIBERO rollout engine."""
    from vla.rl.libero_rollout import LiberoRollout

    return LiberoRollout(
        suite_name=cfg.libero_suite,
        task_id=cfg.task_id,
        num_envs=cfg.num_envs,
        max_steps=cfg.max_steps,
        image_size=256,
        state_dim=cfg.state_dim,
    )


def _build_policy(cfg: CollectionConfig, device: torch.device):
    """Load the SFT checkpoint policy for rollout."""
    from vla.models.smolvla import SmolVLAPolicy

    logger.info("Loading SmolVLA policy from %s ...", cfg.checkpoint)
    policy = SmolVLAPolicy(
        cfg.checkpoint,
        action_dim=cfg.action_dim,
        state_dim=cfg.state_dim,
        device=str(device),
    )
    policy.eval()
    return policy


def collect_rollouts(
    cfg: CollectionConfig,
    device: torch.device,
) -> tuple[list[Trajectory], list[Trajectory], list[Trajectory]]:
    """Collect three trajectory buffers using vectorised LIBERO envs.

    Returns ``(sft_success, sft_failed, random_failed)``.

    Each buffer is populated until it has ``>= cfg.num_rollouts`` entries.
    All results are cached so subsequent runs skip collection.

    .. note::
        The SmolVLA policy is loaded **before** the LIBERO vec-env to
        avoid CUDA OOM — EGL-rendered envs claim GPU memory, so the
        policy must grab VRAM first.
    """
    sft_success = load_trajectories(cfg, "sft_success")
    sft_failed = load_trajectories(cfg, "sft_failed")
    random_failed = load_trajectories(cfg, "random_failed")

    if sft_success is not None and sft_failed is not None and random_failed is not None:
        return sft_success, sft_failed, random_failed

    # ── SFT rollouts ──
    # Load policy FIRST (needs GPU), then create envs (EGL also uses GPU).
    need_sft = sft_success is None or sft_failed is None
    policy = None
    if need_sft:
        policy = _build_policy(cfg, device)

    # Now create the rollout engine (spawns EGL-rendered subprocesses).
    rollout = _build_rollout_engine(cfg)
    instruction = rollout.task_description
    logger.info("Task instruction: %s", instruction)

    if need_sft and policy is not None:
        policy_fn = lambda img, instr, state: policy.predict_action(img, instr, state)
        policy_batch_fn = lambda imgs, instr, states: policy.predict_action_batch(imgs, instr, states)

        sft_success = sft_success or []
        sft_failed = sft_failed or []
        attempt = 0

        while len(sft_success) < cfg.num_rollouts or len(sft_failed) < cfg.num_rollouts:
            n_collect = min(
                cfg.num_envs,
                max(cfg.num_rollouts - len(sft_success), cfg.num_rollouts - len(sft_failed), cfg.num_envs),
            )
            logger.info(
                "SFT rollout wave %d — collecting %d (success=%d/%d, failed=%d/%d)",
                attempt, n_collect,
                len(sft_success), cfg.num_rollouts,
                len(sft_failed), cfg.num_rollouts,
            )
            trajs = rollout.collect_batch(
                policy_fn=policy_fn,
                instruction=instruction,
                num_trajectories=n_collect,
                seed=cfg.seed + attempt * cfg.num_envs,
                policy_batch_fn=policy_batch_fn,
            )
            for t in trajs:
                t.task_id = cfg.task_key
                if t.success and len(sft_success) < cfg.num_rollouts:
                    sft_success.append(t)
                elif not t.success and len(sft_failed) < cfg.num_rollouts:
                    sft_failed.append(t)
            attempt += 1
            if attempt > cfg.num_rollouts * 4:
                logger.warning("Stopping SFT collection after %d waves", attempt)
                break

        sft_success = sft_success[: cfg.num_rollouts]
        sft_failed = sft_failed[: cfg.num_rollouts]
        save_trajectories(sft_success, cfg, "sft_success")
        save_trajectories(sft_failed, cfg, "sft_failed")
        del policy
        gc.collect()
        torch.cuda.empty_cache()

    # ── Random-action rollouts ──
    if random_failed is None:
        random_failed = []

        def random_policy_fn(img, instr, state):
            return torch.randn(cfg.action_dim) * 0.5

        def random_policy_batch_fn(imgs, instr, states):
            return torch.randn(imgs.shape[0], cfg.action_dim) * 0.5

        attempt = 0
        while len(random_failed) < cfg.num_rollouts:
            n_collect = min(cfg.num_envs, cfg.num_rollouts - len(random_failed))
            logger.info(
                "Random rollout wave %d — collecting %d (have %d/%d)",
                attempt, n_collect, len(random_failed), cfg.num_rollouts,
            )
            trajs = rollout.collect_batch(
                policy_fn=random_policy_fn,
                instruction=instruction,
                num_trajectories=n_collect,
                seed=cfg.seed + 10000 + attempt * cfg.num_envs,
                policy_batch_fn=random_policy_batch_fn,
            )
            for t in trajs:
                t.task_id = cfg.task_key
                random_failed.append(t)
            attempt += 1
            if attempt > cfg.num_rollouts * 4:
                break

        random_failed = random_failed[: cfg.num_rollouts]
        save_trajectories(random_failed, cfg, "random_failed")

    rollout.close()
    return sft_success, sft_failed, random_failed
