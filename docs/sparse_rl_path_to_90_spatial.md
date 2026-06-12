# Sparse-RL path to ≥ 90% on LIBERO spatial @ `n_action_steps=1`

This document captures the analysis from the late-April 2026 review session.
It answers the specific question "how do we get above 90 % on LIBERO spatial under `n_action_steps=1` eval, given that SRPO/dense reward does not work for us in practice".

The decisions and numbers here supersede the earlier `--mode srpo` recommendations for the immediate next 2-3 weeks.
The dense-reward path remains documented and reversible; see §6 below.

## Table of contents

1. [Goal and current measured state](#1-goal-and-current-measured-state)
2. [Why we are not yet at 90 %](#2-why-we-are-not-yet-at-90-)
3. [Is SmolVLA SFT saturated on LIBERO?](#3-is-smolvla-sft-saturated-on-libero)
4. [Eval-protocol fairness — our eval has the upper hand](#4-eval-protocol-fairness--our-eval-has-the-upper-hand)
5. [Why our chunks degrade faster than the paper's, and what to do about it](#5-why-our-chunks-degrade-faster-than-the-papers-and-what-to-do-about-it)
6. [Why SRPO/dense reward is parked](#6-why-srpodense-reward-is-parked)
7. [How the FPO loss target is constructed (`full_chunk_target`)](#7-how-the-fpo-loss-target-is-constructed-full_chunk_target)
8. [Sparse-RL path to 90 %](#8-sparse-rl-path-to-90-)
9. [Decision rules and risk register](#9-decision-rules-and-risk-register)

---

## 1. Goal and current measured state

The objective is to take SmolVLA on LIBERO spatial from the current SFT baseline to ≥ 90 % suite-average success rate, evaluated at `--n-action-steps 1` over 100 episodes per task (1000 episodes total) under the current env reset/init-state randomization that landed at commit `3c291f23` / merged at `ef79c3b5` (2026-04-20).

Current measured state, all under the same eval protocol:

| Checkpoint | n=1 spatial | task-5 | Source |
| --- | ---: | ---: | --- |
| SFT `HuggingFaceVLA/smolvla_libero` | 74.3 % | 37 % | `eval_sft_spatial_current_seed42_28242119` |
| RL `28188629/best` (single-task task-5, chunk-1) | 72.0 % | 46 % | `eval_28188629_current_seed42_28254851` |
| RL `28263586/best` (single-task task-5, chunk-5) | TBD (eval submitted) | TBD | pending |
| RL `28327579/best` (single-task task-5, chunk-5, sft_kl=0.005) | TBD (eval submitted) | TBD | pending |
| RL `28335554/best` (single-task task-5, chunk-5, sft_kl=0.005) | TBD (eval submitted) | TBD | pending |

Gap to goal from SFT baseline: `90 − 74.3 = +15.7 pp` on the harder env distribution.

---

## 2. Why we are not yet at 90 %

Three causes are simultaneously true and together explain the entire gap.

### 2a. The 74.3 % SFT baseline is ~15 pp below the published SmolVLA SFT, due to env distribution shift, not training quality

The SmolVLA paper ([arXiv 2506.01844](https://arxiv.org/pdf/2506.01844)) Table 13 reports SFT on `HuggingFaceVLA/smolvla_libero` at `chunk_size=50` with `n_action_steps=1` as **89 % spatial**, identical to `n_action_steps=10` (also 89 %).

Our baseline is 74.3 % at the same setting on the same checkpoint.

The 14.7 pp gap is not eval noise (our 1000-episode CI is ±2.8 pp, the paper's 100-episode CI is ±6.3 pp; the means are still distinguishable).

The most likely cause is the new init-state randomization that landed in the codebase after the published checkpoint was trained.
Under the new randomization the policy is moved partially off-distribution from the very first observation of every episode.

### 2b. All recent RL was single-task, but the goal is a suite average

Every recent RL training run (`28188629`, `28263586`, `28327579`, `28335554`, plus the two `fpo_t5_chunk5_*` configs from April 2026) trained on `task_ids=5` only.

Single-task RL is structurally incapable of moving the suite average:
even a perfect +20 pp on task-5 contributes only +2 pp to suite-avg, and any forgetting on the other 9 tasks erodes that gain.

The empirical pattern from `28188629` confirms this: under the new env it improved task-5 by +9 pp (37 → 46) but lost suite avg by 2.3 pp (74.3 → 72.0).
Net contribution to the actual objective was negative.

To hit suite-avg 90 %, multi-task training is mandatory, with per-task advantage normalization to prevent saturated tasks from dominating gradients.

### 2c. The single highest-leverage missing ingredient (dense reward) does not work for us in practice

The SRPO V-JEPA2 dense-progress-reward path is fully implemented in `src/vla/rl/srpo_reward.py::MultiTaskWorldProgressReward` and is wired through FPO via `Mode.SRPO` at `src/vla/rl/trainer.py:993-1004`.

Empirically the cluster-discovery step collapses on LIBERO: V-JEPA2 mean-pooled per-frame embeddings of "robot doing things in the same kitchen" land in a tight ball, DBSCAN finds either one giant cluster or all-noise, and the resulting per-trajectory reward `g_i` is approximately uniform across failures.

When `g_i` is uniform, the advantage normalisation degenerates to noise and FPO has nothing to learn from.

This is documented in [srpo_reward_study_and_online_progress_clusters.md](./srpo_reward_study_and_online_progress_clusters.md).

The decision is to park the dense-reward line for the immediate next 2–3 weeks of jobs and pursue sparse RL exclusively; see §6 for the documented preconditions to revisit it.

---

## 3. Is SmolVLA SFT saturated on LIBERO?

Mostly yes on the published eval distribution, partly no on ours.

Two distinct claims:

**On the paper's eval distribution** the 450M backbone with the public training recipe sits at 89 spatial / 94 object / 85 goal / 53 long (paper Table 13, n=1).
On the easy three suites this is within ~1–3 pp of larger published SFTs and there is little headroom.
Long has more room (53 → 80+ is what bigger backbones achieve), but Long is not in scope.

**On our harder eval distribution** the same checkpoint sits at 74 spatial.
Some of the 15 pp gap could in principle be recovered by SFT-finetuning on the new reset distribution, but we do not have demos under the new reset distribution.
The two routes to recover it via SFT would be (a) re-collect demos under the new reset, or (b) self-distill from RL rollouts (W6 in the thesis plan), and (b) requires first having a working RL policy on this distribution.

**Bigger backbones are not the bottleneck on spatial.**
The 450M model already hit 89 spatial under the paper's eval.
The published 99 % numbers come from RL on bigger backbones, but RL on the 450M smol backbone with the right ingredients should comfortably reach ≥ 90 % on a harder env.
RL is the right next step, not a bigger SFT.

---

## 4. Eval-protocol fairness — our eval has the upper hand

The published SmolVLA paper evaluates at **10 episodes per task** (paper §6, "evaluates with 10 trials per task").
The standard LIBERO benchmark ([Liu et al. 2023](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/liu_zhu_NeurIPS2023.pdf)) ships 50 default init states per task.
The 10-episode protocol uses init states **0–9** deterministically, which are the init states demonstrations were collected from.

Our eval at `src/vla/envs/libero.py:70-84` uses `np.random.RandomState(seed).randint(num_init_states)` per episode seed.
At 100 episodes per task across seeds 42, 43, …, 141 we sample roughly all 50 init states ~2× each.

Statistical reliability comparison at the suite level:

| Protocol | Episodes | Wilson 95 % CI at p=0.80 |
| --- | ---: | ---: |
| Paper (10/task × 10 tasks) | 100 | ±7.8 pp |
| Ours (100/task × 10 tasks) | 1000 | ±2.5 pp |

Init-state coverage comparison:

| Protocol | Distinct init states / task | Init states unseen at training time |
| --- | ---: | ---: |
| Paper | 10 | 0 (matches demo distribution) |
| Ours | ~50 (with replacement) | ~40 |

This is a defensible thesis-grade framing:

> "The published SmolVLA LIBERO numbers evaluate on a fixed subset of 10 init states per task — the same init states demonstrations were collected from.
> Our protocol samples uniformly across all 50 init states per task at 100 episodes/task, so the policy is tested on init states it has not seen during demo collection.
> This is exactly the criticism that motivated [LIBERO-PRO arXiv 2510.03827](https://arxiv.org/html/2510.03827v1)."

The implication is that our 74.3 % under broad init-state sampling is more informative than the published 89 % under memorised init-state sampling.
A meaningful portion of the 15 pp gap should be attributed to init-state over-fitting at training time, which is a now-published failure mode of LIBERO (LIBERO-PRO), not a defect of our setup.

---

## 5. Why our chunks degrade faster than the paper's, and what to do about it

Measured chunk degradation on the SFT checkpoint, both on spatial:

| n | Paper (10 ep/task, init states 0-9) | Ours (100 ep/task, init states random) | Δ |
| ---: | ---: | ---: | ---: |
| 1 | 89 | 74.3 | −14.7 |
| 2 | — | 69.5 | — |
| 5 | — | 61.8 | — |
| 10 | 89 | ~58 (interp. n=15: 58.3) | ~−31 |
| 30 | 76 | — | — |
| 50 | 54 | 45.4 | −8.6 |

The shape difference is the diagnostic, not the absolute level.

The paper shows a flat plateau from n=1 to n=10 (89 → 89) followed by a collapse at n=30+.
Ours shows ~5 pp loss per added open-loop step starting from n=2, with no flat region.

The mechanical chunk path is correct: see `collect_single_episode_chunked` at [src/vla/rl/rollout.py:228-340](../src/vla/rl/rollout.py) and `predict_action_chunk_batch` at [src/vla/models/smolvla.py:478-497](../src/vla/models/smolvla.py).
The policy emits `chunk_size=50` denormalised actions in one flow-matching sample, the rollout executes the first `n_action_steps`, then re-observes.

The cause is therefore behavioural drift between the SFT policy's learned chunk dynamics and the env's actual dynamics on our distribution.
Two contributing factors, in order of likelihood:

1. The new init-state randomization pushes the policy off-distribution from step 1 (same shift that explains 74.3 vs 89 at n=1).
   When the prefix observation is OOD, the policy's predicted chunk is internally consistent with the *training* distribution but not with the env's actual response after action[0].
   Errors compound from step 2 onwards, hence the linear decay.
2. We have **no temporal ensembling** (verified: zero matches for `temporal_ensembl|action_queue|TemporalEnsemble` in `vla-robotics`, and the standard SmolVLA / π0 eval path in lerobot is also synchronous-without-ensembling per [lerobot issue #1005](https://github.com/huggingface/lerobot/issues/1005)).
   Lerobot's `temporal_ensemble_coeff` is ACT-only and explicitly disabled when `n_action_steps>1`.
   So temporal ensembling does not explain the paper's flat plateau either; the plateau is a genuine OOD-vs-in-distribution chunk-coherence effect.

The lerobot bug at [issue #3312](https://github.com/huggingface/lerobot/issues/3312) (relative-action drift in queued steps) is ACT-specific and does **not** affect us, because `LiberoEnv.step` at `src/vla/envs/libero.py:98-99` forwards each action directly to the lerobot LIBERO env, which applies it as a single delta per env step.

**Practical implication**:
training at `n_action_steps>1` is acceptable as a wall-clock optimisation but the credit assignment becomes noisier as chunk drift grows.
Eval should always be at `n_action_steps=1` for this thesis, both because it is what the goal targets and because it is the fairest comparison protocol.

---

## 6. Why SRPO/dense reward is parked

The SRPO V-JEPA2 reward is the single largest unrealised lever per the published RL-on-VLA literature (SRPO went from 49 % to 99 % on LIBERO-spatial in 79 RL steps using exactly this reward).
We have it implemented but it does not work on our setup.

Failure mode: the V-JEPA2 + per-frame distance aggregation in `MultiTaskWorldProgressReward._encode_trajectories_per_frame` collapses on LIBERO.
LIBERO scenes are visually homogeneous (same kitchen, same camera, same objects), per-frame V-JEPA2 mean-pooled embeddings cluster too tightly, DBSCAN cannot find clusters, and the reward `g_i` is approximately uniform across all failed trajectories.

Concretely, the siiRL production code (the official SRPO implementation, see [srpo_paper_overview.md](./srpo_paper_overview.md)) uses three details we do not match:

1. Per-trajectory clip embeddings of 64 evenly-spaced frames passed through V-JEPA2 as one video, not per-frame independent encoding.
2. `StandardScaler` before DBSCAN with a fixed `eps=0.5` on the standardised space.
3. A min-max normalisation followed by `0.6 * sigmoid(10 * (0.5 − d_norm))` cap, not the paper's z-score.

The minimal debug to reopen the dense-reward line is:

- Switch to the per-trajectory clip encoding path (siiRL recipe).
- Add `StandardScaler` before DBSCAN.
- Re-evaluate cluster diagnostics (silhouette ratio, num_clusters per task) on a single-iteration smoke run.

Decision rule: dense reward returns to the workplan only if either (a) the per-traj fix unblocks DBSCAN in a smoke run, or (b) the sparse-RL path stalls at < 85 % suite avg with no further sparse-side levers available.

---

## 7. How the FPO loss target is constructed (`full_chunk_target`)

Code path: [src/vla/rl/policy_update/base.py:89-139](../src/vla/rl/policy_update/base.py).

Every job in `results/training/` to date uses **`full_chunk_target=True`**.
This is the default in `FPOConfig`, in `train_srpo.py`, and in `configs/train_srpo/base.yaml`.

### `full_chunk_target=True` (current)

For each decision point `t ∈ [0, T_dec)`:

1. Take the flat sequence of actually-executed actions across the whole episode: `flat_actions = executed_chunks[chunk_mask]` of shape `(T_dec * H, action_dim)`.
2. The loss target at obs_t is `flat_actions[t*H : t*H + chunk_size]` — the next 50 actions that were actually executed in the env, padded with zeros and masked beyond what's available.
3. Mask is `True` for the populated positions, `False` for the trailing pad.

At H=5 this means: at decision point `t=0` the target is the 50 actions executed at env-steps 0..49 (produced by 10 separate chunk inferences, only their first 5 each kept).
At `t=1` the target is env-steps 5..54.
This is the **SFT-style sliding-window target**.
Every chunk position gets gradient.
The advantage scalar weights the FM-loss positively or negatively.

### `full_chunk_target=False` (never tried)

For each decision point `t`:

1. Target is `executed_chunks[t]` of shape `(H, action_dim)` — only the H actions actually executed at this decision point.
2. Mask is `True` only for the H executed positions, `False` for chunk positions `H..49`.
3. The FM loss multiplies by the mask before reducing, so chunk positions H..49 receive **zero gradient**.

### Comparison

| Aspect | `full_chunk_target=True` (current) | `full_chunk_target=False` |
| --- | --- | --- |
| Loss positions getting gradient | 50 (full chunk) | H (executed only) |
| Where target comes from | Next 50 *executed* env actions across the next ~50/H decision points | The H actions actually taken at this decision point |
| Implicit credit model | Treat 50-action target as "what to do" at obs_t, weight by advantage | Treat only the H executed actions as "what to do" at obs_t, weight by advantage |
| Chunk coherence | Maintained — supervises full chunk shape every iter | Degrades — positions H..49 drift toward the prior |
| Gradient signal volume | 50/H × larger per decision point | 1× |
| Off-policy bias | Higher — positions H..49 of obs_t's target came from chunks predicted at later observations | Lower — positions 0..H-1 came from obs_t's actual chunk |

**Why we keep `full_chunk_target=True` for chunked training and n=1 eval**:
flow matching solves the whole chunk jointly, so chunk[0] at eval time is part of a denoising trajectory that depends on all 50 positions being in-distribution.
If positions 1..49 drift untrained (Mode B), the FM denoising path that produces chunk[0] is polluted.
Mode A is also what the SFT baseline used, so the KL anchor to SFT is well-defined.

**Worth a one-time A/B**: pick one of the upcoming runs and flip `--no-fpo.full-chunk-target` to compare.
The result is a thesis-able row: "Mode B is X pp worse, confirming dense full-chunk supervision is needed even when only H actions are executed".
If H=5 + Mode B beats H=5 + Mode A, defaults change.

---

## 8. Sparse-RL path to 90 %

Sparse-reward RL has a published ceiling well above 90 % on LIBERO spatial:

- SimpleVLA-RL (binary outcome + DAPO ingredients on OpenVLA-OFT 7B): **99.1 %** spatial.
- RIPT-VLA (binary outcome + REINFORCE-LOO on OpenVLA-OFT 7B): **97.5 %** LIBERO avg.

Our current sparse setup (FPO + LOO + SFT-KL) is in the same family.
Deltas to RIPT/SimpleVLA on the recipe side: backbone size (450M vs 7B), single-task vs multi-task, no rollout temperature, narrow symmetric clip, untuned trajs-per-task.
Backbone size is fixed; everything else is on the table.

### Strategic decisions for the next 2–3 weeks

1. **Multi-task chunk-aware FPO from now on.**
   The single-task task-5 line is a hyperparameter scout; it cannot move suite avg by construction.
   Stop using it as a path to the goal.
2. **Train at `n_action_steps>1` (e.g. 5) for wall-clock, eval at `n_action_steps=1` for fairness.**
   `full_chunk_target=True` makes this safe by keeping chunk coherence supervised across all 50 positions.
3. **Apply DAPO-style sparse-RL ingredients one at a time, in priority order**:
   - **a. Clip-higher**: widen upper FPO clip from `0.08` to `0.16` (lower stays at `0.05`).
     Designed from the sparse-RL recipe notes.
   - **b. Strong SFT-KL anchor**: `sft_kl_coeff=0.02` (vs `0.005` we drifted to).
     The audit showed `raw_sft_kl` at 1–3 vs target 0.01 in recent runs — the anchor was non-binding.
     Without dense reward we need it more, not less.
   - **c. Demos-in-update + success replay**: both already implemented; recent single-task runs disabled them.
     Re-enable for multi-task.
   - **d. Higher rollout temperature** (W9): not exposed today; ~30 min plumbing.
     Defer until first multi-task chunk-aware result lands.
4. **WiSE-FT interpolation as the cheap final lever** (W5).
   Half a day of code; routinely worth +1–3 pp at zero training cost.
5. **Distillation from RL rollouts** (W6) as the path from ~85 to ~90 if (a)–(d) plateau.
   PLD-Stage-3 without the residual actor.
   Two days of code.

### Realistic ceiling estimate

| Increment | Expected ceiling on harder env, n=1 eval |
| --- | ---: |
| Multi-task chunk-aware FPO with v4/v7 stability recipe | 80–84 % |
| + clip-higher 0.05/0.16 | +1–2 pp |
| + WiSE-FT interpolation with SFT | +1–3 pp |
| + rollout temperature 1.4–1.6 | +1–3 pp |
| + per-task rebalancing (W4) | +1–2 pp |
| + distillation from RL rollouts (W6) | +2–5 pp |

Stacking with normal compounding loss: **86–92 %** band.
This gives a real shot at the 90 % goal.
If after all five levers we plateau at ~85, that is the genuine 450M sparse-RL ceiling on the harder env, and the thesis story shifts to "matched the published 80.9 % SFT under a harder-distribution eval and added X pp via post-hoc weight interpolation".

### Sequencing

```
NOW       :  finish the 4 chunk-5 jobs already submitted
             (2 trains: fpo_t5_chunk5_sftkl002, fpo_t5_chunk1_v28_redo
              2 evals:  best of 28327579 and 28335554 at n=1)
+ ~24 h   :  evals report → answer "does chunk-5 training transfer to n=1 eval?"
             - if yes : multi-task chunk-5 is justified for v8
             - if no  : multi-task must train at chunk-1 for v8
+ ~36-72 h:  trains report → pick best single-task hyperparameter recipe as v8 base
+ next    :  v8a = multi-task all-spatial + clip 0.05/0.08 (control)
             v8b = multi-task all-spatial + clip 0.05/0.16 (clip-higher A/B)
+ +1-2 wk :  WiSE-FT merge utility + α sweep across SFT and best v8 ckpt
+ +2-3 wk :  if still < 90: distillation from v8 rollouts, OR rollout-temperature run
```

---

## 9. Decision rules and risk register

Hard decision rules that should govern the next 2–3 weeks of jobs.

| Rule | Trigger | Action |
| --- | --- | --- |
| Drop chunk-5 training | Both pending evals (28327579, 28335554) at n=1 are worse than SFT (74.3) | Multi-task v8 trains at `n_action_steps=1`, accept the 5× wall-clock cost |
| Keep chunk-5 training | At least one pending eval at n=1 is ≥ 75 | Multi-task v8 trains at `n_action_steps=5`, with `full_chunk_target=True` |
| Reopen dense reward | All five sparse levers exhausted and best ckpt still < 85 | Implement per-traj V-JEPA + StandardScaler fix, smoke test cluster diagnostics |
| Skip clip-higher A/B | First multi-task chunk-aware run lands ≥ 88 | Move directly to WiSE-FT, save the slot |
| Add rollout-temperature plumbing | First multi-task chunk-aware run plateaus at < 80 with `clip_frac < 0.05` | Implement W9, run v8c with T=1.4 |
| Promote a checkpoint to thesis figure | Suite avg ≥ 90 % at 100 ep/task on n=1 | Eval cross-suite (object, goal, long), then write up |
| Treat any training as "real" | Eval episodes ≥ 100 per task | < 100 ep/task is scout-only, never a thesis number |

Risk register:

- **Eval noise mistaken for signal** at < 100 ep/task: chronically misled the previous round of decisions; enforce 100 ep/task for every "best ckpt" eval.
- **`raw_sft_kl` drifts to 1–3 vs target 0.01**: SFT anchor is non-binding; always bring `sft_kl_coeff` to 0.02 unless explicitly ablating.
- **`fixed_noise_seed=42` everywhere in eval**: makes evals reproducible but couples them to a specific noise realisation.
  At 100 ep/task this is acceptable; at < 50 ep/task it adds material variance to the "best ckpt" choice.
- **Single-task RL hurts other tasks**: confirmed empirically at task-5; do not let any single-task scout displace a multi-task production run from the queue.
- **Dense reward debt**: the SRPO line is parked, not killed.
  If we publish a "no dense reward needed for SmolVLA-class models" claim, run the per-traj V-JEPA fix once before publication so the negative result has a controlled ablation behind it.

---

## Provenance

- Research session: 2026-04-26 to 2026-05-02.
- Anchor docs: [fpo_hyperparameter_experiments.md](./fpo_hyperparameter_experiments.md), [srpo_reward_study_and_online_progress_clusters.md](./srpo_reward_study_and_online_progress_clusters.md).
- Key external refs: SmolVLA paper [arXiv 2506.01844](https://arxiv.org/pdf/2506.01844), SRPO paper [arXiv 2511.15605](https://arxiv.org/abs/2511.15605), SimpleVLA-RL [arXiv 2509.09674](https://arxiv.org/abs/2509.09674), LIBERO-PRO [arXiv 2510.03827](https://arxiv.org/html/2510.03827v1), lerobot [issue #1005](https://github.com/huggingface/lerobot/issues/1005), lerobot [issue #3312](https://github.com/huggingface/lerobot/issues/3312).
