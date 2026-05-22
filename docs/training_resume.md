# SRPO training resume

Spec for resumable SRPO runs driven by `scripts/train_srpo.py`.
Designed so that a 24h LSF slot can be chained into the next one with zero loss of state, and so that an interrupted job (e.g. SIGTERM from the scheduler) can be resumed from the last completed iteration.

## Why

`src/vla/docs/fpo_hyperparameter_experiments.md` shows that Run 2 needed ~120 iterations (~26 wall-clock hours) to peak at 94 % on LIBERO spatial task 2.
`src/vla/docs/hpc.md` caps `gpul40s` at 24 h walltime per job.
A single slot cannot reach that regime without a clean resume path.

## State that must round-trip

Every piece of state below lives inside `train_srpo` in `src/vla/src/vla/rl/trainer.py` or in the outer CLI in `src/vla/scripts/train_srpo.py`.
Resuming must restore all of it bit-for-bit (modulo floating-point ordering tolerance).

### Model and optimizer
- Model weights — already covered by `policy.save_checkpoint(path)` (writes `policy.pt` + LeRobot sidecars).
- Optimizer state — `torch.optim.AdamW.state_dict()`.
  Contains the Adam moments.
  Missing today.
- LR scheduler — SRPO has none, so nothing to save.
  If one is ever added, its `state_dict()` goes alongside the optimizer.

### Loop counters and best-metric tracking
- `iteration` — index of the last completed training iteration (so resume starts at `iteration + 1`).
- `best_success` — highest eval success rate seen so far, drives the `best/` checkpoint write.
- `best_rollout_successes` — highest rollout-success count seen so far, drives the `best_rollout/` checkpoint write.

### RNG states
- `torch.get_rng_state()`
- `torch.cuda.get_rng_state_all()` when CUDA is available
- `numpy.random.get_state()`
- Python `random.getstate()`

### Success replay buffer
Used by v4 (`--success-replay-total-size 320`) and controlled by the `success_replay_*` fields in `SRPOConfig`.
Round-trip requires:
- `success_buffer: dict[str, list[Trajectory]]` — the trajectories themselves.
  `Trajectory` objects contain tensors, so this is serialised with `torch.save` rather than JSON.
- `success_rate_ema: dict[str, float]` — per-task EMA driving the inverse-success balancing.

### SRPO reward model (only when `mode == "srpo"`)
`MultiTaskWorldProgressReward` holds per-task state inside `_per_task[tid]`:
- `_demo_embeddings: list[torch.Tensor]` — encoded demo references (expensive to recompute, cheap to store).
- `_online_embeddings: list[torch.Tensor]` — rolling online successes added during training.
- `cluster_centers: torch.Tensor | None` — last DBSCAN fit.
- `_last_labels: list[int] | None` — last DBSCAN labels, needed for cluster diagnostics.
- `_last_diagnostics: ClusterDiagnostics | None` — kept so the first iteration after resume logs the correct cluster numbers.

The encoder itself is frozen and rebuilt from `config.world_model_type` on resume, so it does not need to be serialised.

### Adaptive KL state
Only active when `--adaptive-kl` is set, which v4 does not use.
The code mutates `config.kl_coeff` in-place each iteration, so the resumed config must carry the adapted value, not the CLI default.

### Weights & Biases
- Store the run id in `wandb_run_id.txt` on first save.
- On resume, read it and pass `id=<run_id>, resume="allow"` to `wandb.init` (or set `WANDB_RESUME=allow` + `WANDB_RUN_ID`).
  This keeps the iteration-axis continuous in the dashboard.

### Config snapshot
Serialise the resolved `SRPOConfig` (via `config.to_dict()`) plus every CLI flag that is not already in the dataclass (`trajs_per_task_per_iter`, `num_demos`, `checkpoint`, `sft_checkpoint`, `wandb_name`, `use_wandb`).
On resume, re-load and compare against the new invocation's resolved values.
Mismatches on any determinism-affecting field are a hard error — the script prints a diff and exits.

Determinism-affecting fields (checked strictly):

- `seed`, `lr`, `max_grad_norm`, `betas`, `weight_decay`
- `update_method`, `advantage_mode`, `adv_eps`, `adv_skip_threshold`
- `clip_epsilon`, `clip_epsilon_high`, `num_fm_noise_samples`
- `awr_epochs`, `awr_temperature`, `awr_weight_clip`
- `ppo_epochs`, `ppo_minibatch_trajs`
- `fpo_full_chunk_target`, `fpo_loss_reduction`, `fpo_positive_adv_only`
- `fpo_negative_adv_scale`, `fpo_log_ratio_clip`, `fpo_use_ref_policy_kl`
- `kl_coeff`, `sft_kl_coeff`, `adaptive_kl`, `kl_target`, `kl_adapt_factor`
- `mode`, `simulator`, `suite`, `task_id`, `state_dim`
- `num_rollout_envs`, `num_envs`, `fm_batch_size`, `max_steps`
- `world_model_type`, `distance_metric`, `subsample_every`
- `dbscan_eps`, `dbscan_min_samples`, `dbscan_auto_eps`
- `use_failure_rewards`, `use_standard_scaler`
- `include_demos_in_update`
- `success_replay_buffer_size`, `success_replay_total_size`, `success_replay_alpha`
- `success_replay_ema_decay`, `success_replay_max_ratio`
- `dynamic_sampling`, `dynamic_sampling_max_retries`
- `trajs_per_task_per_iter`

Non-determinism-affecting fields (allowed to change on resume):

- `num_iterations` — typically extended on resume.
- `eval_every`, `eval_episodes`, `eval_zero_sample` — may be tuned between slots.
- `gradient_checkpointing` — memory knob, not a training-signal knob.
- `use_wandb`, `wandb_name` — logging.

## Save layout under `<ckpt-dir>`

```
<ckpt-dir>/
  latest/                      # written every iteration, atomic via tmp + rename
    policy.pt                  # existing policy checkpoint (weights + normalisers)
    config.json                # resolved config + CLI extras
    model.safetensors          # LeRobot sidecar (from policy.save_checkpoint)
    policy_preprocessor.json   # LeRobot sidecar
    policy_postprocessor.json  # LeRobot sidecar
    state.pt                   # optimizer, RNG, iteration, replay buffer, reward-model, best-metric tracking, adaptive-KL
    wandb_run_id.txt           # plain-text W&B id for shell inspection
  best/                        # existing: highest eval success rate
  best_rollout/                # existing: highest rollout success count
  snapshots/                   # optional, keep every N iterations
    iter_00050/
      policy.pt, state.pt, ...
    iter_00100/
      ...
  metrics.jsonl                # unchanged (JSONL log sink appends on resume)
  training_run.json            # unchanged
```

The existing `save_dir/last/` directory is retained for backward compatibility (equivalent to `latest/`, but without the extra state).
The new `latest/` is the only directory that the resume path reads from.

### Atomic write

Every write to `latest/` happens via:

1. Write files under `<ckpt-dir>/latest.tmp/`.
2. `os.replace(latest.tmp, latest)` after all writes complete, after removing the old `latest/` first on systems where `replace` cannot overwrite directories.

This guarantees that a job killed mid-save leaves either the previous `latest/` intact or the new one fully written.

### Snapshot retention

Optional, off by default.
When `--checkpoint-keep-every N` is set, every `N`-th iteration additionally writes to `snapshots/iter_<iter>/` with the same contents as `latest/`.

## `--resume-from <ckpt-dir>` contract

- If `<ckpt-dir>/latest/state.pt` does not exist:
  - Log a clear notice: `no resume state at <path>, starting fresh run`.
  - Continue as a fresh run.
  - The run still writes to `<ckpt-dir>/latest/` from iteration 1.
- If `<ckpt-dir>/latest/state.pt` exists:
  - Load the saved config and compare against the current CLI's resolved config.
  - On any mismatch in a determinism-affecting field: print a diff and exit with a non-zero code.
    The operator must either re-submit with the same flags or explicitly start a fresh run into a new directory.
  - On match: restore model (from `policy.pt`), optimizer, RNG, iteration, replay buffer, reward-model, best-metric trackers, adaptive-KL, and W&B id.
  - Training loop runs `for iteration in range(saved_iter + 1, num_iterations + 1)`.
  - If `saved_iter >= num_iterations`: log `already at num_iterations, nothing to do` and exit cleanly with a success code.
    This lets a slot N+1 in an LSF array no-op safely when slot N happened to reach the final iteration.

## CLI additions to `scripts/train_srpo.py`

- `--resume-from <path>` — directory containing `latest/state.pt` from a prior run.
  Default `None` (fresh start).
- `--checkpoint-out-dir <path>` — where to write `latest/`, `best/`, `best_rollout/`, `snapshots/`.
  Defaults to the existing `save_dir` derived from `CHECKPOINTS_DIR / mode / run_tag` so current behaviour is unchanged.
  On resume, the caller is expected to pass the same path.
- `--checkpoint-keep-every <N>` — optional snapshot retention.
  Default 0 (snapshots disabled).

Note: resume uses `--resume-from` to locate prior state.
The new `<ckpt-dir>/latest/` directory *is* `<ckpt-dir>` combined with `latest/`, so `--resume-from` and `--checkpoint-out-dir` point at the same `<ckpt-dir>` for a chained run.

## Validation gate

The SIGTERM test is the pass/fail criterion for this work.
It requires a GPU and a working LIBERO environment, so the gate is executed on `voltash` or `a100sh`, not on the dev laptop.

### Harness

`scripts/validate_resume.sh` automates the full procedure in one shell invocation:

1. Baseline 2-iteration run in `$BASELINE_DIR`.
2. A second 2-iteration run in `$RESUME_DIR`, killed with `SIGTERM` once the iter-1 entry appears in `metrics.jsonl`.
3. A resume of `$RESUME_DIR` via `--resume-from`.
4. An inline diff of iter-2 metrics across the two `metrics.jsonl` files against the `1e-5` relative tolerance.

### Procedure

```bash
# on DTU, reserve an interactive GPU session first:
linuxsh -q voltash -W 1:00 -gpu "num=1:mode=exclusive_process"

cd src/vla
export SFT_CHECKPOINT=$HOME/smolvla_libero/spatial/best
export OUT_ROOT=$WORK3/resume_validation_$(date +%Y%m%d_%H%M%S)

bash scripts/validate_resume.sh
```

Environment overrides supported by the harness: `SFT_CHECKPOINT`, `OUT_ROOT`, `MODE`, `UPDATE_METHOD`, `SEED`, `TASK_IDS`, `TRAJS_PER_TASK`, `NUM_ROLLOUT_ENVS`, `MAX_STEPS`.

### Pass criterion

- `fpo_loss` / `awr_loss` / `ppo_loss`, `kl_penalty`, `step_kl_penalty`, `mean_ratio`, `raw_kl`, `max_log_ratio` match within `1e-5` relative tolerance (floating-point ordering noise).
- `total_successes`, `rollout_successes`, `replay_successes` match exactly.
- When iter 2 coincides with an eval step (`iteration == num_iterations` guarantees this with the harness), `eval/success_rate` matches the baseline exactly.

Any divergence larger than the tolerance means something in the state dict is missing, loaded in the wrong order, or the RNG advance differs.
Fix the implementation and rerun.

### Recorded result

_Pending execution on voltash / a100sh._
The harness prints a per-key diff table at the end; paste the output below once the gate has run.

| Metric | Baseline (iter 2) | Resumed (iter 2) | Rel diff | Within tolerance? |
| --- | --- | --- | --- | --- |
| `fpo_loss` |  |  |  |  |
| `kl_penalty` |  |  |  |  |
| `mean_ratio` |  |  |  |  |
| `raw_kl` |  |  |  |  |
| `total_successes` |  |  |  |  |
| `rollout_successes` |  |  |  |  |
| `replay_successes` |  |  |  |  |
| `eval/success_rate` |  |  |  |  |
