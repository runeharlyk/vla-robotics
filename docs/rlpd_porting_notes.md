# Porting RLPD/HIL-SERL ideas to SmolVLA + LIBERO

This note records what we deliberately ported from the HIL-SERL paper
(Luo et al. 2024, arXiv:2410.21845) and its underlying RL algorithm
RLPD (Ball et al. 2023) into our SmolVLA + LIBERO setup, what we
deliberately skipped, and why.

The first concrete experiment driven by this note is
`v12_multi_success_bc_rlpd5050_utd.yaml`.

## Why look at HIL-SERL at all

Our multi-task RL experiments on LIBERO-spatial keep landing in two failure modes:

1. **FPO drift** (v10b): `raw_sft_kl` doubles every iteration; saturated tasks (t1, t3, t6, t9) regress 5-10pp on the rollout EMA while hard tasks (t5) gain little.
2. **success_bc plateau** (v11b): `raw_sft_kl` stays at ~5e-5 (the policy barely moves); suite-avg is essentially flat vs SFT init at iter 6 (74.4 -> 73.7), with one real per-task signal (+8pp on t5).

Both failure modes are about *data composition*, not loss-function tuning. The SFT-KL coefficient is six orders of magnitude smaller than the BC loss in v11b; raising it is theatre.
HIL-SERL's central trick is that it never uses an explicit KL anchor at all, and yet it stably trains a policy from a sparse binary reward in 1-2.5 hours on a real robot.
That worked with a backbone that is comparable in scale to ours (HIL-SERL is roughly a 5-10M trainable head on top of a frozen ImageNet ResNet-10; SmolVLA is ~500M in the trainable VLA), so the *algorithmic* lessons should at least be cheap to test.

## What HIL-SERL actually does

Three layers stacked together:

1. **HIL** (the human-in-the-loop part).
   A SpaceMouse operator can take over control at any time during a rollout.
   The intervention bytes are written to *both* the demo replay buffer and the on-policy RL buffer.
   This is the part responsible for the headline "near-perfect in 1-2.5h on a real robot".

2. **SERL = RLPD applied to robotics**.
   RLPD (Ball et al. 2023) is the underlying algorithm.
   It does three things differently from a vanilla SAC fine-tune:

   a. **Symmetric prior+online sampling**.
   Each gradient batch is exactly 50% prior data (demos) + 50% on-policy data.
   No KL penalty in the loss; the demo data acts as an implicit anchor through the sampler.

   b. **Off-policy Q-learning** with high update-to-data (UTD) ratio.
   The actor maximises an entropy-regularised SAC objective on top of a Q-function trained from off-policy data.
   The Q-function is trained for many gradient steps per environment step (UTD typically 16-20).

   c. **Layer-norm critic + ensembled min-target**.
   Stabilises the high-UTD critic against value-overestimation collapse.

3. **System architecture**.
   Distributed actor/learner over gRPC with a *frozen* ImageNet ResNet-10 vision backbone.
   Only a small MLP head is trained for both actor and critic.
   Pretrained binary success classifier provides sparse rewards.

## What we ported (and why)

Implemented as `success_bc.balanced_demo_sampling: true` with `epochs: 6`, `minibatch_trajs: 8` in `v12_multi_success_bc_rlpd5050_utd.yaml`.

### 1. Symmetric demo + online minibatch sampling (RLPD #2a)

Rationale.
v11b's success_bc batch is dominated by saturated-task successes because those tasks generate ~92 successful rollouts per iter while t5 generates ~30.
With the existing uniform shuffle, an iteration's gradient is structurally biased toward whatever task happens to have the most recent successes.
RLPD's solution -- enforce a fixed prior:online ratio per minibatch -- directly addresses this without requiring a Q-function or any new loss term.

Implementation.
`Trajectory` now carries an `is_demo: bool` flag.
The trainer marks demo trajectories at insertion time.
`success_bc_update` builds minibatches via `_build_balanced_minibatches` (`src/vla/rl/policy_update/success_bc.py`) which emits batches whose composition matches `success_bc.demo_sampling_ratio` (default 0.5).
When one pool empties before the other, the sampler falls back to the still-populated pool to avoid stalling the optimiser mid-epoch.

We log `sparse_rl/success_bc/demo_fraction` and `sparse_rl/success_bc/online_fraction` so we can verify the sampler is doing what we asked.

### 2. Higher update-to-data ratio (RLPD #2b, with caveats)

Rationale.
v11b runs a single epoch with `minibatch_trajs=4`, giving roughly 40 gradient steps per iteration (160 successful trajectories / 4).
Across 24,000 environment steps per iter that's a UTD of ~0.0017 -- five orders of magnitude smaller than RLPD's typical UTD=20.
We can't match that; SmolVLA is 500M parameters, every forward+backward through the flow-matching action decoder is expensive, and at UTD=20 a 24h L40s slot would finish ~3 iterations.

Compromise.
`success_bc.epochs: 6` + `minibatch_trajs: 8` lifts grad steps per iter to roughly 150-200 (depends on success count).
That's a 4-5x UTD lift, achievable inside a 24h L40s wall clock based on v11b's rough iter timings.
If the wall clock blows, we can drop epochs back to 3.

### 3. Drop the SFT-KL anchor entirely

Rationale.
Empirically `raw_sft_kl * sft_coeff = 4.4e-5 * 0.02 = 9e-7` on v11b at iter 8, while the success_bc loss term is ~0.29.
The KL is a no-op in the loss surface but pollutes the diagnostics.
Conceptually, the 50/50 demo batch *is* the anchor in RLPD: every gradient step explicitly sees demo data, which prevents drift far more strongly than a KL term whose magnitude is dwarfed by the BC objective.

`v12` sets `kl.coeff: 0.0` and `kl.sft_coeff: 0.0`.
If the policy drifts (`raw_sft_kl > 0.05`) without the KL term, that tells us the demo-fraction sampling alone wasn't enough; we'd add it back at a magnitude that actually contributes to the loss.

### 4. Demo-replay validation + init-state-matched replay

Rationale (and a separate concern raised during the design).
LIBERO ships pre-recorded init-states under `init_files/<problem_folder>/<task>.pruned_init` (50 per task on the spatial / object / goal suites, 50 per long task).
The upstream `LIBERO/collect_demonstrations.py` records demo `k` against `init_state[k]` -- a 1:1 mapping that the `lerobot/libero_*_image` parquet conversion preserves in row order.
Our previous `replay_demo_rollouts` ignored that mapping and called `env.reset(seed=seed + spec_idx*1000 + demo_idx)`, which routes through `np.random.RandomState(seed).randint(num_init_states)` -- a uniform random pick over the 50 init-states.
For contact-rich tasks the demo's open-loop actions almost certainly fail when the cube starts a few centimetres from where it was during demo recording.

Failed demo replays don't corrupt the gradient when `success_bc` filters to `success=True` before the update (trainer.py:1082), but they *do* silently halve the effective demo data when the replay rate is low, **and** they pollute the reward-model demo set, the success-buffer pre-seed, and the RLPD-style 50/50 sampler -- which all consume the unfiltered replayed dict.

Implementation.

1. `Trajectory` carries an `init_state_id: int | None`.
2. `_load_libero_v2_from_hf` returns the within-task rank (= init_state_id) of every kept episode; `LiberoSFTDataset.episodes_as_trajectories(task_id=...)` attaches it.
3. `LiberoEnv.reset(seed, init_state_id=None)` accepts an explicit init-state-id that takes precedence over the seed-derived choice; the underlying LeRobot attribute name (`_init_state_id` on v0.4.x, `init_state_id` on main) is set with `_set_underlying_init_state_id` so we are forward-compatible with PR #2832.
4. `_replay_single_demo` forwards the demo's `init_state_id` to `env.reset` (try/except so non-LIBERO envs that don't accept the kwarg still work).
5. `replay_demo_rollouts(..., drop_failed_replays=True)` filters failed replays out of the dict before the trainer sees them and logs the kept-vs-dropped count.
6. The replay cache key incorporates `init_state_id`, so old caches are auto-invalidated.

Diagnostics on wandb:

- `sparse_rl/{tid}/demo_replay_success_rate` -- fraction of *attempted* replays that succeeded (computed pre-filter, so it stays meaningful).
- `sparse_rl/demo_replay_success_rate` -- suite average of the above.
- `sparse_rl/{tid}/demos_kept_after_replay` -- per-task kept count post-filter.
- `sparse_rl/demos_kept_after_replay_total` -- suite total post-filter.

If `demo_replay_success_rate` is still < 0.8 on a task with the init-state-matched replay, that points at a different bug (e.g. action normalisation drift or a LeRobot dataset action-frame mismatch) and should block the experiment.

## What we did not port (and why)

### Q-function head + DQN/SAC critic update

Reason.
SmolVLA is a flow-matching action sampler.
There is no tractable `log pi(a|s)` to plug into SAC's actor loss.
RLPD's sample efficiency comes from off-policy Q-learning that re-uses transitions for many gradient updates, but adding a Q-head to SmolVLA is real engineering: define the action embedding, define the Q architecture, decide chunk-vs-single-action, train the critic stably with bfloat16 features.
This is the right *next* experiment after v12 if v12 doesn't move the needle, but it's not a 1-week change.

A cheap pilot: train a Q-head on top of frozen-ish SmolVLA features for a single task (t5, the only one with measurable RL signal) and see if the Q learns anything sensible in 24h.
If yes, plan a full Q-head experiment.
If no, stay in the policy-gradient/BC family.

### Frozen pretrained vision backbone

Reason.
HIL-SERL freezes ResNet-10 and only trains an MLP head.
SmolVLA's vision tower (the VLM backbone) is entwined with action prediction through the flow-matching decoder; you can't naively freeze just the vision part without re-architecting the action decoder.
Plus, if we're not training a Q-function, the "frozen backbone" trick has no purpose.

### Layer-norm critic + ensembled min-target

Reason.
Only relevant once a Q-function exists.
Defer alongside the Q-head experiment.

### Human interventions

Reason.
Sim-only LIBERO has no human in the loop.
The closest analogue is "have more demos in the dataset", which we already do.

### Replacing the SmolVLA flow-matching action sampler with SAC's tanh-Gaussian

Reason.
That's a different policy class entirely; it would require dropping the SmolVLA pretrained weights (which is what made our SFT 74.3% baseline strong in the first place) and starting over with a much smaller policy.
Not on the table.

## Predicted outcomes

| Run | Expected suite-avg @ iter 12 | Reasoning |
|-----|------------------------------|-----------|
| v11b | 73.5 ± 2pp | trajectory says iter-12 is in the noise band of iter-6 |
| v12  | 74-77% | the anchor is now real (50/50) so the policy actually moves; UTD lift gives more chances per iter to find a better minimum |

Failure modes to watch for in v12:

- `raw_sft_kl` climbs > 0.05 within 6 iters: the demo-sampling anchor wasn't enough, add a small `sft_coeff` back.
- `demo_replay_success_rate` < 0.5 for any task even with init-state-matched replay: the "demos" being mixed in still aren't reaching the goal. Investigate action normalisation / action-space mismatch in the LeRobot dataset before treating them as anchors.
- `demos_kept_after_replay_total` falls more than 30% below `num_demos * num_tasks`: drop_failed_replays is silently downsampling the demo pool below what `success_bc.demo_sampling_ratio` assumes.
- Wall clock per iter > 30 min: epochs=6 is too aggressive, drop to 3.

## Submission

```
invoke submit-train --experiment v12_multi_success_bc_rlpd5050_utd --profile l40s-16 --submit
```

Run on the cluster login node (Windows can generate the script but cannot `bsub`).
The same v11b L40s slot will work; v12 is structurally identical to v11b in rollout cost and adds ~4x gradient cost.
