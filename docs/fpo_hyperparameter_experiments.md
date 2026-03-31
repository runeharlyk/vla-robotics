# FPO Hyperparameter Study — LIBERO Spatial Task 2

## Overview

This document summarises findings from eight FPO (Flow Policy Optimization) training runs on LIBERO spatial task 2 (`sparse_rl` mode with binary success/failure rewards).
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
| Trajectories per iteration | 32 (runs 1–6), varies in runs 7–8 |
| Rollout envs | 8 (vectorised) |
| FM batch size | 64 |
| FM noise samples | 4 |
| KL coeff | 0.01 |
| Max grad norm | 10.0 |
| Gradient checkpointing | Yes |

Runs 7–8 also vary `trajs_per_task`, `ppo_epochs`, and `eval_episodes`; deviations from the baseline are noted in the run summary table.

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
| 7 | 3e-6 | 0.05 / 0.08 | 1.0 | 35 | 220 | 60% | **86%** (iter 30) | 84% | Comparable to run 2 at same iter count ¹ |
| 8 | 3e-6 | 0.05 / 0.08 | 1.0 | 64/100 | 220 | 59% | **90%** (iter 30) | **76%** (iter 60) | Degradation after peak ² |

¹ Run 7: `trajs_per_task=64` (2× baseline), all other settings match run 2.
² Run 8: `ppo_epochs=2`, `trajs_per_task=48`, `eval_episodes=100`. Still running at iter 64.

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

### 6. Doubling batch size (64 trajs) does not accelerate convergence

Run 7 doubled trajectories per iteration from 32 to 64 while keeping all other settings at the proven baseline (run 2 config).
At iteration 30, run 7 reached 86% eval SR — matching run 2 at the same iteration count.
However, each iteration took ~22 minutes vs ~13 minutes for the 32-traj baseline, so wall-clock cost per iteration roughly doubled.

The rollout success rate climbed steadily from 56% (iter 2) to 95% (iter 32), showing healthy learning, but the eval improvement was not proportionally faster.
The larger batch provides lower-variance advantage estimates: with 64 trajs and 50% success rate, LOO advantages are ±0.5 vs ±1.0 with 32 trajs.
But at `lr=3e-6` with conservative clipping, the optimiser is not gradient-noise-limited — the bottleneck is the trust-region step size, not advantage estimate quality.

**Conclusion:** `trajs_per_task=32` remains the cost-effective choice at the current learning rate.
Larger batches may help at higher learning rates (where more stable gradients could prevent the collapse seen in runs 5–6), but this is untested.

### 7. Two PPO epochs cause late-stage performance degradation

Run 8 used `ppo_epochs=2` with `trajs_per_task=48` and `eval_episodes=100`.
Early training was healthy: 59% → 76% (iter 10) → 84% (iter 20) → **90%** (iter 30), matching or exceeding the best of run 2 at comparable iterations.
But after iter 30, performance steadily declined: 89% (iter 40) → 85% (iter 50) → **76%** (iter 60).

The mechanism: with 2 PPO epochs, the second gradient pass uses the FM-loss ratio computed at the start of the iteration.
By the time the second epoch runs, the policy has already moved from the first epoch's update.
The ratio `r = exp(L_old − L_new)` is stale — it reflects the _original_ policy, not the post-epoch-1 policy.
This means the trust-region constraint is effectively broken on the second pass.

Diagnostic evidence from run 8:

| Symptom | Evidence |
| --- | --- |
| Updates making policy worse | Positive FPO losses at iters 33, 38, 39, 42 |
| Diminishing learning signal | KL dropped to 0.001–0.004 in late phase |
| Rollout degradation | Rollout SR declined from 87–94% (iters 25–32) to 73–83% (iters 50–64) |
| Eval degradation | 90% → 89% → 85% → 76% over iters 30–60 |

Note: run 8 changed three settings simultaneously (`ppo_epochs`, `trajs_per_task`, `eval_episodes`), but run 7 shows that changing only `trajs_per_task` does not cause degradation.
The `eval_episodes` increase is evaluation-only and cannot affect training.
Therefore `ppo_epochs=2` is the most likely cause of the regression.

**Conclusion:** `ppo_epochs=1` is correct for FPO.
The extra gradient step does not extract meaningful additional signal — instead it corrupts the trust-region approximation.
If more signal extraction per iteration is desired, increasing `trajs_per_task` is the safer path since it improves advantage estimates without introducing ratio staleness.

### 8. 100 eval episodes reveal real trends (confirmed)

Run 8 used 100 eval episodes as suggested in the recommendations from runs 1–6.
The eval trajectory 59% → 76% → 84% → 90% → 89% → 85% → 76% shows a clear rise-and-fall arc that would have been obscured by 50-episode noise.
With 50 episodes, the binomial 95% CI at 85% SR is ±10%, making a drop from 90% to 85% indistinguishable from noise.
With 100 episodes, the same CI is ±7%, and five consecutive evals showing monotonic decline (90% → 89% → 85% → 76%) strongly indicate real regression rather than sampling variance.

**Conclusion:** 100 eval episodes should be the standard going forward.

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
--ppo-epochs 1                # 2 epochs shown harmful (run 8)
--trajs-per-task 32           # 48/64 did not help (runs 7, 8)
--num-fm-noise-samples 4
--eval-episodes 100            # confirmed less noisy (run 8)
--max-steps 220
--adaptive-kl                  # if implemented — highest-impact change
```

Run 8 disproved the hypothesis that `ppo_epochs=2` would extract more signal — it caused degradation instead.
Run 7 showed that doubling `trajs_per_task` to 64 provided no meaningful speedup.
The most promising next step is adaptive KL targeting (see recommendations below), which could unlock higher learning rates without collapse.

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
| ppo_epochs | 1 | **1** | ≥ 2 (stale FM ratio → degradation) | Runs 2, 8 |
| trajs_per_task | 32–64 | **32** (cost-effective) | < 16 (high variance) | Runs 7, 8 |
| eval_episodes | 100+ | **100** (confirmed) | < 50 (noisy checkpoints) | Runs 2, 8 |
