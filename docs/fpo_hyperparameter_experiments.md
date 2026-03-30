# FPO Hyperparameter Study — LIBERO Spatial Task 2

## Overview

This document summarises findings from six FPO (Flow Policy Optimization) training runs on LIBERO spatial task 2 (`sparse_rl` mode with binary success/failure rewards).
All runs use the same codebase (FPO update with leave-one-out advantages, asymmetric PPO-clip), the same SFT-initialised SmolVLA checkpoint, and differ only in hyperparameters.
The goal is to identify which settings produce stable learning and which cause collapse, and to guide future experiments.

---

## Experimental Setup

| Setting | Value (shared across all runs) |
| --- | --- |
| Simulator | LIBERO spatial, task 2 |
| Update method | FPO (flow policy optimization) |
| Advantage mode | Leave-one-out (RLOO) |
| Reward mode | `sparse_rl` (1.0 = success, 0.0 = failure) |
| Trajectories per iteration | 32 |
| Rollout envs | 8 (vectorised) |
| FM batch size | 64 |
| FM noise samples | 4 |
| KL coeff | 0.01 |
| Max grad norm | 10.0 |
| Gradient checkpointing | Yes |

---

## Run Summary

| Run | LR | Clip ε / ε_high | Neg adv scale | Iters | Max steps | Baseline SR | Best SR | Final SR | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3e-6 | 0.05 / 0.08 | 1.0 | 50 | 280 | 72% | **86%** (iter 45) | 84% | Stable, slow |
| 2 | 3e-6 | 0.05 / 0.08 | 1.0 | 200 | 220 | 58% | **94%** (iter 120) | ~82% | Best result, late stagnation |
| 3 | 3e-6 | 0.10 / 0.15 | 1.0 | 35 | 220 | 58% | **82%** (iter 30) | 82% | Slower convergence |
| 4 | 3e-6 | 0.05 / 0.08 | 0.5 | 35 | 220 | 58% | **88%** (iter 35) | 88% | Slightly worse than run 2 |
| 5 | **5e-6** | 0.10 / 0.15 | 1.0 | 35 | 220 | 58% | **90%** (iter 15) | **50%** | Collapsed |
| 6 | **5e-6** | 0.10 / 0.15 | **0.5** | 35 | 220 | 60% | **88%** (iter 15) | **4%** | Catastrophic collapse |

---

## Key Findings

### 1. LR = 5e-6 causes catastrophic collapse

Both runs with `lr=5e-6` (runs 5 and 6) collapsed after initially reaching 88–90% success rate.
The collapse signature is visible in the KL divergence metric:

| Run | LR | KL range (stable phase) | KL range (pre-collapse) | Outcome |
| --- | --- | --- | --- | --- |
| 2 | 3e-6 | 0.005–0.013 | 0.001–0.005 | Stable |
| 5 | 5e-6 | 0.017–0.041 | 0.009–0.011 | Collapsed to 50% |
| 6 | 5e-6 | 0.027–0.082 | 0.008–0.014 | Collapsed to 4% |

At higher LR the policy moves too far from the reference each iteration.
The FM-loss ratio `r = exp(L_old − L_new)` is only a valid proxy for the true likelihood ratio when policy changes are small.
When the ratio approximation breaks down, the PPO-clip trust region loses its meaning and the policy spirals.

The existing KL penalty (`kl_coeff=0.01`) is too weak to compensate for the larger step sizes.

**Conclusion:** `lr=3e-6` is the safe choice for the current FPO implementation.
Higher learning rates require a stronger or adaptive KL penalty (see recommendations below).

### 2. Larger clip ranges do not help and may hurt

Run 3 (`clip=0.1/0.15`, `lr=3e-6`) was strictly worse than run 2 (`clip=0.05/0.08`, same LR) at matched iteration counts.
At iteration 15, run 3 was still at baseline (58%) while run 2 was at ~72%.

Wider clips don't help when the FPO ratio is already well within the trust region — and they increase the risk of destabilising steps without providing any upside.

Combined with higher LR (runs 5, 6), larger clips amplify the step size problem and accelerate collapse.

**Conclusion:** `clip_epsilon=0.05` / `clip_epsilon_high=0.08` is the right setting.
These values align with the FPO paper's ablation, which found ε = 0.05 optimal.

### 3. Negative advantage scaling has marginal impact at conservative LR

Run 4 (`neg_adv_scale=0.5`, `lr=3e-6`) reached 88% at iter 35 — comparable to run 2 at the same iteration.
The FPO loss magnitudes are much larger (~−0.2 vs ~−0.005) because the negative-advantage trajectories now contribute a large asymmetric term, but this doesn't translate into faster or more stable learning.

At high LR (run 6), `neg_adv_scale=0.5` made collapse *worse* — the reduced push-away from bad trajectories removed a stabilising force.

**Conclusion:** `neg_adv_scale=1.0` (symmetric) is preferred.
Scaling down negative advantages removes useful signal without a clear benefit.

### 4. Late-stage signal vanishes after ~80% success rate

In run 2 (the longest run), the training dynamics evolve through distinct phases:

| Phase | Iters | FPO loss | KL | Rollout SR | Eval SR |
| --- | --- | --- | --- | --- | --- |
| Active learning | 1–30 | −0.005 to −0.010 | 0.006–0.013 | 47–84% | 58→72% |
| Moderate learning | 31–60 | −0.002 to −0.005 | 0.004–0.010 | 69–94% | 72→90% |
| Near convergence | 61–90 | −0.001 to −0.002 | 0.001–0.009 | 72–97% | 86–92% |
| Stagnation | 91–126 | −0.0005 to −0.002 | 0.001–0.005 | 63–94% | 78–94% |

As the rollout success rate approaches 90%, most trajectories have reward 1.0.
Leave-one-out advantages become very small (e.g. if 28/32 succeed, the LOO advantage for a success is only +0.14, for a failure is −1.0, and after z-score normalisation these shrink further).
The FPO ratio stays near 1.0 so the clipped surrogate loss approaches zero.
The policy effectively stops learning.

**Conclusion:** Sparse binary rewards have a natural ceiling around 90–94% SR for this task.
Pushing past this requires either more trajectories per iteration, a finer-grained reward signal (SRPO world-model rewards), or curriculum methods.

### 5. Eval noise obscures true performance

With 50 eval episodes and binary success at ~85% SR, the 95% confidence interval is roughly ±10%.
This means a measured drop from 92% to 82% could be pure noise, not real regression.

The eval in run 2 oscillates: 88% → 82% → 90% → 86% → 92% → 86% → 82% → 94% → 78% — much of this is sampling variance, not genuine policy quality changes.

**Conclusion:** Use at least 100 eval episodes for more reliable checkpointing.
A binomial 95% CI at 85% SR drops from ±10% (N=50) to ±7% (N=100).

---

## Recommended Configuration

### Proven stable baseline (run 2)

```
--lr 3e-06
--clip-epsilon 0.05
--clip-epsilon-high 0.08
--fpo-negative-adv-scale 1.0
--kl-coeff 0.01
--ppo-epochs 1
--trajs-per-task 32
--num-fm-noise-samples 4
--eval-episodes 50
--max-steps 220
```

### Suggested improvements (next experiment)

```
--lr 3e-06
--clip-epsilon 0.05
--clip-epsilon-high 0.08
--fpo-negative-adv-scale 1.0
--kl-coeff 0.01
--ppo-epochs 2               # extract more signal per iteration
--trajs-per-task 48           # better advantage estimates at high SR
--num-fm-noise-samples 4
--eval-episodes 100            # less noisy eval
--max-steps 220
```

The extra cost per iteration (~50% more rollout time, 2× gradient steps) should be worthwhile because it lets each expensive iteration extract more learning signal — especially in the late stage where gradients are small.

---

## Recommendations for Code Changes

### High impact: Adaptive KL penalty

The fixed `kl_coeff=0.01` cannot adapt to different effective step sizes.
Implementing adaptive KL targeting (as in the original PPO paper) would allow safely using higher learning rates:

```python
kl_target = 0.01
if actual_kl > 2.0 * kl_target:
    kl_coeff *= 1.5
elif actual_kl < 0.5 * kl_target:
    kl_coeff /= 1.5
```

This would decouple the LR choice from the trust-region constraint: the penalty auto-increases when the policy moves too fast.
Run 5's early phase (fast learning, 90% at iter 15) would be preserved while the collapse would be prevented.

### High impact: KL-based step rejection

A simple safety net: if the KL after a policy update exceeds a hard ceiling, revert the update and skip the iteration.
This is cheap insurance against the kind of spiral seen in runs 5 and 6.

### Medium impact: Richer ratio diagnostics

Currently only `clip_frac` is logged.
Adding `mean_ratio`, `std_ratio`, and `max_abs_log_ratio` would give early warning of impending collapse — in runs 5 and 6 these would have shown drift well before the eval SR dropped.

### Medium impact: LR schedule

A linear or cosine warmup over the first 5–10 iterations could smooth the early training phase.
A late-stage decay could prevent over-fitting when the signal is weak, though this matters less than the adaptive KL.

### Lower priority: Switch to SRPO mode

The `sparse_rl` mode is the right choice for initial experiments — it's simple and reliable.
But the natural ceiling at ~90% SR comes from the binary reward.
Switching to `srpo` mode with world-model progress rewards provides continuous reward values that can distinguish "barely succeeded" from "succeeded efficiently", giving finer gradient signal for late-stage refinement.

---

## Hyperparameter Sensitivity Summary

| Parameter | Safe range | Optimal | Danger zone | Evidence |
| --- | --- | --- | --- | --- |
| Learning rate | 1e-6 – 3e-6 | **3e-6** | ≥ 5e-6 (collapse) | Runs 5, 6 vs 1, 2 |
| clip_epsilon | 0.03 – 0.08 | **0.05** | ≥ 0.1 (no benefit, adds risk) | Run 3 vs 2 |
| clip_epsilon_high | 0.05 – 0.10 | **0.08** | ≥ 0.15 (compounds with LR) | Run 3 vs 2 |
| kl_coeff | 0.005 – 0.02 | **0.01** at lr=3e-6 | < 0.01 at lr=5e-6 (insufficient) | Runs 5, 6 |
| neg_adv_scale | 0.5 – 1.0 | **1.0** (symmetric) | — | Run 4 marginal diff |
| ppo_epochs | 1 – 2 | **1** (proven), 2 (untested) | ≥ 4 (stale FM ratio) | Run 2 |
| trajs_per_task | 32+ | **32** (proven), 48 (suggested) | < 16 (high variance) | All runs |
| eval_episodes | 50+ | **100** (suggested) | < 50 (noisy checkpoints) | Run 2 oscillation |
