"""Trajectory collection utilities for clustering analysis.

Provides functions to collect and cache trajectories from four sources:

1. **Demo trajectories** — loaded from the HuggingFace LeRobot dataset.
2. **SFT rollouts** — collected using the SmolVLA SFT checkpoint, split
   into success and failure buffers.
3. **Random-action rollouts** — collected with a random policy.
4. **Progress trajectories** — replay the first N% of a reference
   trajectory's actions, then random for the remainder.

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
from vla.utils.tensor import action_to_numpy

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
    demo_replay_seed_mode: str = "episode_index"
    demo_replay_fixed_seed: int = 0

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


def _replay_seed_for_demo(cfg: CollectionConfig, demo: Trajectory, demo_idx: int) -> int:
    """Resolve the env reset seed used for demo action replay."""
    mode = cfg.demo_replay_seed_mode.lower()
    if mode == "fixed":
        return cfg.demo_replay_fixed_seed
    if mode == "fixed_offset":
        return cfg.demo_replay_fixed_seed + demo_idx
    if mode == "episode_index":
        for entry in demo.privileged_states:
            if "source_episode_index" in entry:
                return int(entry["source_episode_index"])
        logger.warning(
            "Demo %d has no source_episode_index metadata; falling back to seed=%d.",
            demo_idx,
            cfg.seed + demo_idx,
        )
        return cfg.seed + demo_idx
    if mode == "collection_offset":
        return cfg.seed + 50_000 + demo_idx
    raise ValueError(
        f"Unknown demo_replay_seed_mode={cfg.demo_replay_seed_mode!r}. "
        "Choose from: episode_index, fixed, fixed_offset, collection_offset."
    )


def _replay_recorded_actions(adapter, demo: Trajectory, max_steps: int, seed: int) -> Trajectory:
    """Replay a recorded action sequence and collect live simulator observations."""
    raw_obs = adapter.reset(seed)
    images: list[torch.Tensor] = []
    states: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    dones: list[torch.Tensor] = []
    success = False

    recorded_len = int(demo.length or demo.actions.shape[0])
    horizon = min(recorded_len, max_steps)
    for t in range(horizon):
        img_t, state_t = adapter.obs_to_tensors(raw_obs)
        action_t = demo.actions[t].detach().cpu().float()

        images.append(img_t)
        states.append(state_t)
        actions.append(action_t)

        result = adapter.step(action_to_numpy(action_t))
        raw_obs = result.raw_obs
        rewards.append(torch.tensor(float(result.reward), dtype=torch.float32))
        done = bool(result.terminated or result.truncated)
        dones.append(torch.tensor(float(done), dtype=torch.float32))

        if result.success:
            success = True
        if done:
            break

    used = len(actions)
    return Trajectory(
        images=torch.stack(images) if images else torch.empty(0),
        states=torch.stack(states).float() if states else torch.empty(0),
        actions=torch.stack(actions).float() if actions else torch.empty(0),
        rewards=torch.stack(rewards) if rewards else torch.empty(0),
        dones=torch.stack(dones) if dones else torch.empty(0),
        success=success,
        length=used,
        task_id=demo.task_id,
        privileged_states=list(demo.privileged_states),
    )


def collect_replayed_demo_trajectories(
    cfg: CollectionConfig,
    demo_trajs: list[Trajectory] | None = None,
) -> list[Trajectory]:
    """Replay demo actions in LIBERO and cache simulator-view trajectories.

    Raw HuggingFace demos are useful for actions, but their stored camera stream
    can differ from the live rollout camera stream. Replaying the demo actions
    gives the same observation format as SFT/random rollouts, which makes the
    reward-cluster comparison much less confounded by view/source differences.
    """
    cached = load_trajectories(cfg, "replayed_demos")
    if cached is not None:
        return cached

    demos = demo_trajs if demo_trajs is not None else collect_demo_trajectories(cfg)
    if not demos:
        save_trajectories([], cfg, "replayed_demos")
        return []

    from vla.rl.libero_rollout import LiberoRollout

    rollout = LiberoRollout(
        suite_name=cfg.libero_suite,
        task_id=cfg.task_id,
        num_envs=1,
        max_steps=cfg.max_steps,
        image_size=256,
        state_dim=cfg.state_dim,
    )
    adapter = rollout._make_single_adapter()

    replayed: list[Trajectory] = []
    success_count = 0
    try:
        for demo_idx, demo in enumerate(demos[: cfg.num_demos]):
            replay_seed = _replay_seed_for_demo(cfg, demo, demo_idx)
            traj = _replay_recorded_actions(
                adapter=adapter,
                demo=demo,
                max_steps=cfg.max_steps,
                seed=replay_seed,
            )
            traj.task_id = cfg.task_key
            replayed.append(traj)
            success_count += int(traj.success)
            logger.info(
                "Replayed demo %d with seed=%d using seed_mode=%s (success=%s, length=%d).",
                demo_idx,
                replay_seed,
                cfg.demo_replay_seed_mode,
                traj.success,
                traj.length,
            )
    finally:
        rollout.close()

    logger.info(
        "Replayed %d demo trajectory/trajectories for %s (%d successes).",
        len(replayed),
        cfg.task_key,
        success_count,
    )
    save_trajectories(replayed, cfg, "replayed_demos")
    return replayed


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
        def policy_fn(img, instr, state):
            return policy.predict_action(img, instr, state)

        def policy_batch_fn(imgs, instr, states):
            return policy.predict_action_batch(imgs, instr, states)

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


# ──────────────────────────────────────────────────────────────────────
# Progress-level trajectories (action replay + random tail)
# ──────────────────────────────────────────────────────────────────────


def _make_replay_policy(
    recorded_actions: torch.Tensor,
    cutoff_step: int,
    action_dim: int,
):
    """Build a policy that replays recorded actions up to cutoff, then random."""
    step_counter = [0]

    def policy_fn(image, instruction, state=None):
        t = step_counter[0]
        step_counter[0] += 1
        if t < cutoff_step and t < len(recorded_actions):
            return recorded_actions[t]
        return torch.randn(action_dim) * 0.5

    return policy_fn, step_counter


def collect_progress_trajectories(
    cfg: CollectionConfig,
    reference_trajs: list[Trajectory],
    progress_levels: list[float] | None = None,
    source_name: str = "demos",
) -> dict[float, list[Trajectory]]:
    """Collect trajectories at varying progress levels via action replay.

    For each reference trajectory and each progress level, replays the
    first ``N%`` of the recorded actions in the environment, then switches
    to random actions for the remaining steps.

    Results are cached per level as
    ``{suite}_task{task_id}_progress_{pct}_from_{source}.pt``.

    Args:
        cfg: Collection configuration (suite, task, cache dir, etc.).
        reference_trajs: Successful trajectories whose actions we replay.
        progress_levels: Fractions of the trajectory to replay (default
            ``[1.0, 0.75, 0.5, 0.25, 0.0]``).
        source_name: Label for cache keys (``"demos"`` or ``"sft_success"``).

    Returns:
        ``{level: [Trajectory, ...]}`` dict, one list per progress level.
    """
    from vla.rl.libero_rollout import LiberoRollout
    from vla.rl.rollout import collect_single_episode

    if progress_levels is None:
        progress_levels = [1.0, 0.75, 0.5, 0.25, 0.0]

    result: dict[float, list[Trajectory]] = {}
    all_cached = True

    for level in progress_levels:
        pct = int(level * 100)
        cache_name = f"progress_{pct}_from_{source_name}"
        cached = load_trajectories(cfg, cache_name)
        if cached is not None:
            result[level] = cached
        else:
            all_cached = False
            result[level] = []

    if all_cached:
        return result

    rollout = LiberoRollout(
        suite_name=cfg.libero_suite,
        task_id=cfg.task_id,
        num_envs=1,
        max_steps=cfg.max_steps,
        image_size=256,
        state_dim=cfg.state_dim,
    )

    from vla.rl.libero_rollout import _LiberoSingleAdapter
    adapter = _LiberoSingleAdapter(rollout)
    instruction = rollout.task_description

    for level in progress_levels:
        pct = int(level * 100)
        cache_name = f"progress_{pct}_from_{source_name}"
        if result[level]:
            continue

        logger.info(
            "Collecting progress trajectories: level=%d%% from %s (%d refs)",
            pct, source_name, len(reference_trajs),
        )

        trajs: list[Trajectory] = []
        for ref_idx, ref_traj in enumerate(reference_trajs):
            cutoff = int(ref_traj.length * level)
            policy_fn, counter = _make_replay_policy(
                ref_traj.actions, cutoff, cfg.action_dim,
            )
            traj = collect_single_episode(
                adapter=adapter,
                policy_fn=policy_fn,
                instruction=instruction,
                max_steps=cfg.max_steps,
                seed=cfg.seed + ref_idx + pct * 1000,
            )
            traj.task_id = cfg.task_key
            trajs.append(traj)

        result[level] = trajs
        save_trajectories(trajs, cfg, cache_name)

    rollout.close()
    return result
