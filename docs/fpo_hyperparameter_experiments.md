# FPO Hyperparameter Study — LIBERO Spatial Task 2

## Overview

This document summarises findings from ten FPO (Flow Policy Optimization) training runs on LIBERO spatial task 2 (`sparse_rl` mode with binary success/failure rewards).
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
| Trajectories per iteration | 32 (runs 1–6, 9–10), varies in runs 7–8 |
| Rollout envs | 8 (vectorised) |
| FM batch size | 64 |
| FM noise samples | 4 |
| KL coeff | 0.01 |
| Max grad norm | 10.0 |
| Gradient checkpointing | Yes |

Runs 7–8 also vary `trajs_per_task`, `ppo_epochs`, and `eval_episodes`.
Runs 9–10 add `adaptive_kl=True` with `kl_target=0.01` and `kl_adapt_factor=1.5` at `lr=5e-6` (see run summary footnotes).
Deviations from the baseline are noted in the run summary table.

---

## Run Summary

| Run | LR | Clip ε / ε_high | Neg adv scale | Iters | Max steps | Baseline SR | Best SR | Final SR | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3e-6 | 0.05 / 0.08 | 1.0 | 50 | 280 | 72% | **86%** (iter 45) | 84% (iter 50) | Stable, slow |
| 2 | 3e-6 | 0.05 / 0.08 | 1.0 | 126 ⁶ | 220 | 58% | **94%** (iter 120) | 94% (iter 120) ⁷ | Best result, late oscillation 78–94% |
| 3 | 3e-6 | 0.10 / 0.15 | 1.0 | 35 | 220 | 58% | **82%** (iter 30) | 82% (iter 35) | Slower convergence |
| 4 | 3e-6 | 0.05 / 0.08 | 0.5 | 35 | 220 | 58% | **88%** (iter 35) | 88% (iter 35) | Slightly worse than run 2 |
| 5 | **5e-6** | 0.10 / 0.15 | 1.0 | 35 | 220 | 58% | **90%** (iter 15) | **50%** (iter 35) | Collapsed |
| 6 | **5e-6** | 0.10 / 0.15 | **0.5** | 35 | 220 | 60% | **88%** (iter 15) | **4%** (iter 35) | Catastrophic collapse |
| 7 | 3e-6 | 0.05 / 0.08 | 1.0 | 35 | 220 | 60% | **86%** (iter 30) | 84% (iter 35) | Comparable to run 2 at same iter count ¹ |
| 8 | 3e-6 | 0.05 / 0.08 | 1.0 | 67 ³ | 220 | 59% | **90%** (iter 30) | **76%** (iter 60) | Degradation after peak; stopped early ² |
| 9 | **5e-6** | 0.05 / 0.08 | 1.0 | 100 | 220 | 60% | **91%** (iter 30) | **12%** (iter 100) | Adaptive KL; late collapse ⁴ |
| 10 | **5e-6** | **0.10 / 0.15** | 1.0 | 100 | 220 | 60% | **88%** (iter 20) | **16%** (iter 100) | Adaptive KL + wide clip; faster collapse ⁵ |

¹ Run 7: `trajs_per_task=64` (2× baseline), all other settings match run 2.
² Run 8: `ppo_epochs=2`, `trajs_per_task=48`, `eval_episodes=100`.
³ Run 8 stopped at training iter 67 (external signal); last eval at iter 60 was 76%.
⁴ Run 9: `adaptive_kl=True`, `kl_target=0.01`, `ppo_epochs=1`, `trajs_per_task=32`, `eval_episodes=100`.
⁵ Run 10: same as run 9 except `clip_epsilon=0.10`, `clip_epsilon_high=0.15`.
⁶ Run 2 was configured for 200 iters but stopped at iter 126 (external signal); the last logged eval is at iter 120.
⁷ Run 2's last logged eval was 94% at iter 120; evals between iters 40–120 oscillate 78–94% (see §5 for the full sequence).

---

## Key Findings

### 1. LR = 5e-6 causes catastrophic collapse

Runs with `lr=5e-6` and fixed `kl_coeff` (runs 5 and 6) collapsed after initially reaching 88–90% success rate.
Runs 9 and 10 used adaptive KL at `lr=5e-6` and also collapsed after strong mid-training peaks (see §9).
The collapse signature is visible in the KL divergence metric:

| Run | LR | KL range (early/learning phase) | KL range (late phase) | Outcome |
| --- | --- | --- | --- | --- |
| 2 | 3e-6 | 0.005–0.013 (iters ~5–30) | 0.001–0.005 (iters 91–126, late stagnation, no collapse) | Stable |
| 5 | 5e-6 | 0.017–0.041 (iters 1–25) | 0.009–0.014 (iters 30–35, pre/at collapse) | Collapsed to 50% |
| 6 | 5e-6 | 0.027–0.082 (iters 1–20) | 0.008–0.028 (iters 26–35, during collapse) | Collapsed to 4% |

At higher LR the policy moves too far from the reference each iteration.
The FM-loss ratio `r = exp(L_old − L_new)` is only a valid proxy for the true likelihood ratio when policy changes are small.
When the ratio approximation breaks down, the PPO-clip trust region loses its meaning and the policy spirals.

The existing KL penalty (`kl_coeff=0.01`) is too weak to compensate for the larger step sizes.

**Conclusion:** `lr=3e-6` is the safe choice for the current FPO implementation.
Higher learning rates need a KL scheme that actually constrains cumulative drift; the adaptive rule tested in runs 9–10 did not (§9).

### 2. Larger clip ranges do not help and may hurt

Run 3 (`clip=0.1/0.15`, `lr=3e-6`) was strictly worse than run 2 (`clip=0.05/0.08`, same LR) at matched iteration counts.
At iteration 15, run 3 was still at baseline (58%) while run 2 was at ~72%.

Wider clips don't help when the FPO ratio is already well within the trust region — and they increase the risk of destabilising steps without providing any upside.

Combined with higher LR (runs 5, 6), larger clips amplify the step size problem and accelerate collapse.
Run 10 (`clip=0.1/0.15`, `lr=5e-6`, adaptive KL) peaked at 88% by iter 20 but eval fell to 68% by iter 30 and to ~16% by iter 100 — worse mid-run than run 9’s conservative clips at the same LR.

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
Leave-one-out advantages become very small (e.g. if 28/32 succeed, the LOO advantage for a success is `1 − 27/31 ≈ +0.13`, for a failure is `0 − 28/31 ≈ −0.90`, and after z-score normalisation these shrink further).
The FPO ratio stays near 1.0 so the clipped surrogate loss approaches zero.
The policy effectively stops learning.

**Conclusion:** Sparse binary rewards have a natural ceiling around 90–94% SR for this task.
Pushing past this requires either more trajectories per iteration, a finer-grained reward signal (SRPO world-model rewards), or curriculum methods.

### 5. Eval noise obscures true performance

With 50 eval episodes and binary success at ~85% SR, the 95% confidence interval is roughly ±10%.
This means a measured drop from 92% to 82% could be pure noise, not real regression.

The eval in run 2 oscillates (iters 40 → 120 in steps of 10): 88% → 82% → 90% → 86% → 92% → 86% → 82% → 78% → 94% — much of this is sampling variance, not genuine policy quality changes.

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
| Rollout degradation | Rollout SR declined from 87–94% (iters 25–32) to 73–83% (iters 50–67) |
| Eval degradation | 90% → 89% → 85% → 76% over iters 30–60 (last logged eval before early stop) |

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

### 9. Adaptive KL as implemented did not stabilise `lr=5e-6` (runs 9–10)

Runs 9 and 10 used `adaptive_kl=True`, initial `kl_coeff=0.01`, `kl_target=0.01`, and adapt factor 1.5 (halve coeff when `raw_kl < 0.5 * target`, multiply when above `2 * target`).
Logged `raw_kl` per iteration stayed far below 0.01 (often ~1e-3 to 1e-4), so the rule repeatedly *decreased* `kl_coeff` until it was effectively zero within the first few tens of iterations.
That removed KL regularisation while the optimiser still stepped at `5e-6`, so the policy could drift and the FM-loss ratio proxy could break down later — matching the late catastrophic eval drop.

Run 9 eval arc (N=100): 60% → 65% (10) → 82% (20) → **91%** (30) → 64% (40) → 21% (50) → 15% (60) → 10–16% (70–100); final **12%** at iter 100.
Run 10 eval arc: 60% → 72% (10) → **88%** (20) → 68% (30) → 29% (40) → 19% (50) → 18% (60) → 15–20% (70–90); final **16%** at iter 100.

**Conclusion:** Do not assume “adaptive KL” fixes high LR until the adaptation signal matches the quantity you care about (e.g. floor on `kl_coeff`, different target, or KL measured vs a fixed reference).
Runs 9–10 are a negative result for the specific rule and metric used in those logs.

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
```

Run 8 disproved the hypothesis that `ppo_epochs=2` would extract more signal — it caused degradation instead.
Run 7 showed that doubling `trajs_per_task` to 64 provided no meaningful speedup.
Runs 9–10 show that the adaptive KL rule logged in those runs does *not* unlock stable `lr=5e-6`; redesign the adaptation (floor, target, or KL definition) before retrying higher LR.

---

## Recommendations for Code Changes

### High impact: Adaptive KL penalty (needs redesign after runs 9–10)

The fixed `kl_coeff=0.01` cannot adapt to different effective step sizes.
A symmetric adapt rule (as in the original PPO paper) can still fail if the monitored `actual_kl` is almost always below `kl_target`: runs 9–10 then drove `kl_coeff` toward zero and later collapsed.

Any retry should at least: enforce a **floor** on `kl_coeff`; consider a `kl_target` matched to the scale of the logged KL; or adapt using a KL vs a **fixed** reference policy, not a quantity that stays tiny while the policy drifts.

```python
kl_target = 0.01
if actual_kl > 2.0 * kl_target:
    kl_coeff *= 1.5
elif actual_kl < 0.5 * kl_target:
    kl_coeff /= 1.5
kl_coeff = max(kl_coeff, kl_coeff_floor)
```

Run 5's fast early phase at `5e-6` is still desirable, but runs 9–10 show the naive decrease-only regime is dangerous for FPO at that LR.

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
| Learning rate | 1e-6 – 3e-6 | **3e-6** (single-task) / **2e-6** (multi-task) | ≥ 5e-6 (collapse) | Runs 5, 6, 9, 10 vs 1, 2; multi-task §13 |
| clip_epsilon | 0.03 – 0.08 | **0.05** | ≥ 0.1 (no benefit, adds risk) | Run 3 vs 2; run 10 vs 9 |
| clip_epsilon_high | 0.05 – 0.10 | **0.08** | ≥ 0.15 (compounds with LR) | Run 3 vs 2; run 10 vs 9 |
| kl_coeff | 0.005 – 0.02 | **0.01** at lr=3e-6 | Collapsing to ~0 at lr=5e-6 (adaptive) | Runs 9, 10 |
| adaptive_kl | — | **off** until rule fixed | Decreasing-only + tiny raw_kl | Runs 9, 10 |
| neg_adv_scale | 0.5 – 1.0 | **1.0** (symmetric) | — | Run 4 marginal diff |
| ppo_epochs | 1 | **1** | ≥ 2 (stale FM ratio → degradation) | Runs 2, 8 |
| trajs_per_task | 32–64 | **32** (cost-effective) | < 16 (high variance) | Runs 7, 8 |
| eval_episodes | 100+ | **100** (confirmed) | < 50 (noisy checkpoints) | Runs 2, 8; §15 |
| sft_kl_coeff | 0.005 – 0.02 | **0.02** when long single-task or multi-task | ≤ 0.005 (anchor non-binding past iter ~30) | Multi-task vs t5-chunk5 long runs §11 |
| n_action_steps (training) | 1–5 | **1** for n=1 eval; **5** acceptable for wall-clock | > 5 untested at this LR | §10, eval pending |
| full_chunk_target | true | **true** | false untested; expect chunk-coherence drift | §12, all jobs to date |
| dynamic_sampling | on/off | **on** as safety net | — at trajs=32 below saturation | Multi-task §13; rl_vla_paper_recipes §3 |

---

## Updates 2026-04: New env reset, task-5 chunked, multi-task all-spatial

This appendix captures findings from FPO runs after the LIBERO init-state/reset randomisation change at commit `3c291f23` / merged `ef79c3b5` (2026-04-20).
The original §1–§9 findings were collected on task-2 with the previous (easier) reset distribution.
Hyperparameter conclusions there still hold; the new findings below extend them to task-5, to multi-task `--task-ids all`, and to chunk-aware execution at `--n-action-steps > 1`.

For the strategic plan that uses these findings see [sparse_rl_path_to_90_spatial.md](./sparse_rl_path_to_90_spatial.md) and [libero_spatial_rl_experiment_plan.md](./libero_spatial_rl_experiment_plan.md).

### Run summary — task-5 / multi-task under new env

All runs use FPO with leave-one-out advantages, `clip 0.05/0.08`, `num_fm_noise_samples=4`, `ppo_epochs=1`, `full_chunk_target=true`, and `seed=42`.
"task-5 best" is the maximum eval success rate at the trained-task during training; "raw_sft_kl mean" is the mean of `sparse_rl/raw_sft_kl` across all logged iterations.
"eval_eps" is the episodes-per-task budget used for in-training evaluation.

| Job ID | Run name (short) | Tasks | n_act | LR | sft_kl_coeff | iters | eval_eps | raw_sft_kl mean | task-5 best | Suite-avg post-RL @ n=1 |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 28188629 | v28-t5-sft-kl-0.005 (chunk-1) | task-5 | 1 | 3e-6 | 0.005 | 30 | 10 | n/a | 0.80¹ | 0.720² (eval `28254851`) |
| 28267780 | v28-t5-na1 (chunk-1) | task-5 | 1 | 3e-6 | 0.005 | — | 10 | n/a | 0.70 | not measured |
| 28267778 | v28-t5-na2 (chunk-2) | task-5 | 2 | 3e-6 | 0.005 | — | 10 | n/a | 0.40 | not measured |
| 28263586 | v28-t5-na5 (chunk-5) | task-5 | 5 | 3e-6 | 0.005 | — | 10 | 0.003 | 0.80¹ | not measured |
| 28292409 | t5-chunk5 (eval-100, dynamic on) | task-5 | 5 | 3e-6 | 0.005 | — | 50 | — | 0.58 | pending |
| 28327579 | t5-chunk5-nodynamic-confirm | task-5 | 5 | 3e-6 | 0.005 | 100 | 100 | **1.397** | **0.64** | pending eval at n=1 |
| 28335554 | fpo-t5-chunk5-lowsft (sft_kl=0.001) | task-5 | 5 | 3e-6 | 0.001 | 100 | 100 | **1.413** | **0.65** | pending eval at n=1 |
| 28314273 | v28-control-current-seeded-t5-n1 | task-5 | 1 | 3e-6 | 0.005 | 30 (27 done) | 50 | n/a | 0.44 (CI ±13) | matched-protocol |
| 28246360 | spatial-all-v4 (multi-task) | all 10 | 1 | 2e-6 | 0.02 | — | 20 | — | — | suite-avg best 0.755 |
| 28249578 | spatial-all-v7 (multi-task + dynamic) | all 10 | 1 | 2e-6 | 0.02 | — | 20 | **0.004** | — | **suite-avg best 0.795** |

¹ Training-time eval at `eval_episodes=10` per iter (single task, normal-approx 95 % CI ±25 pp at p=0.80); the pre-env-change offline eval `28192830` (N=100, old env) reported task-5 = 0.84 for the same checkpoint, while the post-env-change eval `28254851` (N=100, new env) reports task-5 = 0.46.
The 0.84 vs 0.46 gap reflects the env reset randomisation change at commit `ef79c3b5`, not sampling noise — see §15.
Subsequent 100-episode evals under the new env show the actual SR at task-5 is ≤ 0.65, see runs 28327579 / 28335554.
² Eval at `n_action_steps=1` over 100 episodes per task × 10 tasks = 1000 total episodes; suite-avg fell from SFT 0.743 to 0.720 because single-task RL on task-5 hurt the other 9 tasks.

### 10. Chunk-aware training (`--n-action-steps > 1`) is a wall-clock optimisation, not a quality optimisation

Setting `--n-action-steps H > 1` switches the rollout loop to chunk execution: for each decision point the policy emits one 50-action chunk and the env executes the first H actions before the next obs is sampled (see [`src/vla/rl/rollout.py:228-340`](../src/vla/rl/rollout.py)).
Wall-clock per iteration drops sub-linearly with H — measured ~3× faster at H=5 on these workloads, **not** H×, because env step, FM-loss computation, replay, and eval are not chunk-rate.

Empirical wall-clock from `_runtime` deltas in the training-curve files on task-5 (single L40s):

| Run | n_act | min / iter | iters logged | Total wall-clock | Decision points / episode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 28267780 (chunk-1) | 1 | 12.7 | 30 | 6.4 h | 220 |
| 28188629 (chunk-1) | 1 | 11.6 | 30 | 5.8 h | 220 |
| 28263586 (chunk-5) | 5 | 3.8 | 30 | 1.9 h | 44 |
| 28327579 (chunk-5, nodynamic) | 5 | 4.9 | 100 | 8.2 h | 44 |

Effective speed-up chunk-5 vs chunk-1 is **~3.0–3.3×**, not 5×.
Plan future budgets from these measured per-iter times rather than the H× heuristic.

The eval transfer from chunk-aware training back to `n_action_steps=1` eval is **not yet established**.
Two pending evals (`28327579/best` and `28335554/best`, both at `n_action_steps=1` over 100 ep/task) are designed to answer this directly.
Decision rule: if either is < 0.74 (SFT baseline at n=1), chunk-aware training does not transfer and future multi-task runs go to `n_action_steps=1` despite the wall-clock cost.

Train at `n_action_steps=H` only if the eval target is also `n_action_steps=H` *or* the pending evals confirm transfer.
For the thesis goal of "n=1 eval at 90 %" the eval is fixed at n=1, so chunk-aware training is justified only by the wall-clock argument and only if transfer holds.

### 11. SFT-anchor binding behaviour (`raw_sft_kl`) — `sft_kl_coeff=0.005` is too loose for long single-task runs

The training-time KL of the policy vs the SFT reference (`sparse_rl/raw_sft_kl` in the logs) is the diagnostic for whether the SFT anchor is doing work.

Observed `raw_sft_kl` ranges:

| Run | sft_kl_coeff | iters logged (local jsonl) | raw_sft_kl range | Anchor status |
| --- | ---: | ---: | --- | --- |
| 28249578 (multi-task v7) | 0.02 | 2 ⁸ | 0.003 – 0.005 (n=2) | Binding at iters 1–2; longer-run behaviour not verified locally |
| 28263586 (t5 chunk-5, short) | 0.005 | 30 | 0.000 – 0.008 (mean 0.003) | Binding |
| 28327579 (t5 chunk-5, 100 iter) | 0.005 | 100 | 0.000 – **3.93**, mean **1.40** | Non-binding past iter ~30 |
| 28335554 (t5 chunk-5, 100 iter) | 0.001 | 100 | 0.000 – **4.16**, mean **1.41** | Non-binding immediately |

⁸ The local training-curve file for 28249578 (`spatial-all-v7-…_28249578.jsonl`) contains only 3 rows (iter 0 eval, iters 1–2 training); subsequent iterations are not synced locally.
The "binding" diagnosis here is therefore based on 2 observations and should be re-validated against the wandb run before being used to justify multi-task hyperparameter choices.

Pattern: at `sft_kl_coeff ≤ 0.005` and ≥ 50 iterations on a single task with no demos-in-update or success-replay regulariser, `raw_sft_kl` drifts from ~0.003 into the 1–4 range — i.e. the policy is several orders of magnitude further from the SFT anchor than the implicit `kl_target=0.01` would suggest.
`28263586` did not exhibit this only because it stopped before iter ~30.
For `28249578` the long-iteration anchor behaviour is **not verified from local data** — the conjecture is that (a) `sft_kl_coeff=0.02` is 4× stronger and (b) demos-in-update + success-replay add their own SFT-aligned gradient, but this requires confirmation against the full wandb history before being treated as established.

**Conclusion for sparse RL on the new env**: prefer `sft_kl_coeff = 0.02`.
For single-task scouts ≥ 50 iterations, `0.005` is too loose; expect raw_sft_kl > 1 by iter 60 and treat any "best" past that point as off-anchor.
For ablating "is the SFT anchor what's holding things together?" use `0.001` deliberately and document — runs 28335554 vs 28327579 already provide one such pair.

### 12. `full_chunk_target=true` is the default and right choice — and `false` is the natural ablation row

`FPOConfig.full_chunk_target` controls how the FM-loss target is built from a chunked rollout (see [`src/vla/rl/policy_update/base.py:89-139`](../src/vla/rl/policy_update/base.py)).

| Mode | Loss target at obs_t | Positions getting gradient | Off-policy bias |
| --- | --- | ---: | --- |
| `true` (default, all jobs) | Sliding window of next 50 *executed* env actions | 50 | Higher — positions H..49 came from later observations |
| `false` (untested) | Only the H actions executed at this decision point | H | Lower |

Mode `true` preserves SFT-style dense supervision across the full 50-position chunk and keeps chunk coherence trained, which matters because flow matching solves the whole chunk jointly — chunk[0] at eval depends on chunk[1..49] being in-distribution.
Mode `false` provides cleaner credit but lets chunk positions H..49 drift toward the prior; risky for any policy intended to be evaluated at a different `n_action_steps` than it was trained at.

Every result in this repo to date uses `true`.
A one-time A/B with `--no-fpo.full-chunk-target` on a planned chunk-5 run would produce a clean ablation row for the thesis.

### 13. Single-task RL on task-5 hurts suite average under the new env

`28188629` (single-task task-5, chunk-1, 30 iters, training-time `eval_episodes=10`) under the new env:

- task-5 success: SFT 0.37 → RL 0.46 (**+9 pp**, post-env-change offline eval `28254851`, 100 ep / task, n=1)
- spatial suite avg: SFT 0.743 → RL 0.720 (**−2.3 pp**, same eval `28254851`, 100 ep × 10 tasks = 1000 total, n=1)
- For reference, the pre-env-change offline eval `28192830` (old reset distribution at git `e3b72285`) reported the same checkpoint at task-5 = 0.84 / suite-avg = 0.788 — not directly comparable to the new-env numbers; see §15.

Net contribution of the single-task line to the actual objective is negative.

The mechanism is that the SFT KL anchor cannot fully prevent representation drift on the 9 untrained tasks while the policy is being pulled hard toward task-5 success; demos-in-update and success-replay help only on tasks present in the replay buffer.

**Caveat on the multi-task numbers in the table above**: the locally synced training-curve files for `28246360` and `28249578` contain only the iter-0 (pre-RL) suite eval at `eval_episodes=20` per task — 2 rows total for 28246360 and 3 rows for 28249578.
The "suite-avg best" values 0.755 and 0.795 in the run-summary table are therefore pre-RL baselines measured with N=20, not verified post-RL improvements.
Comparing 0.755 / 0.795 (N=20) against the SFT suite-avg 0.743 (N=100, eval `28242119`) is within the sampling noise of N=20 (95 % CI ≈ ±6 pp per task → ≈ ±2 pp suite-avg) and does **not** demonstrate post-RL improvement on its own.
Treat the "0.743 → 0.755 → 0.795 without forgetting" claim as a working hypothesis and re-validate against a post-RL offline suite eval at N=100 (matching protocol with `28254851`/`28242119`) before relying on it for planning.

**Conclusion**: single-task FPO is a hyperparameter scout, not a path to suite-avg goals.
Promote any single-task winner to multi-task (`--task-ids all`) before reporting suite-avg numbers, and require a post-RL offline suite eval at `--eval-episodes 100` per task before claiming multi-task improvement over SFT.

### 14. Demos-in-update + success replay — directional preference, not statistically proven from local data

Multi-task runs that included demos and success replay outperform those that did not (per the run-summary table):

| Run | demos in update | success_replay total | success_replay max ratio | Suite-avg best ¹⁰ |
| --- | :---: | ---: | ---: | ---: |
| spatial-all-v2 (28236602) | yes | 320 | 0.5 | 0.730 |
| spatial-all-v4 (28246360) | yes | 320 | 0.5 | 0.755 |
| spatial-all-v7 (28249578) | yes | 320 | 0.5 | **0.795** |
| spatial-all-v5 | yes | 320 | 1.0 | regressed |
| Single-task chunk-5 long (28327579) | no | 0 | — | task-5 only 0.64 |

¹⁰ As noted in §13, the locally synced training-curve files for `28236602`, `28246360`, and `28249578` contain only the iter-0 (pre-RL) suite eval at `eval_episodes=20`.
The "suite-avg best" 0.730 / 0.755 / 0.795 values are therefore pre-RL baselines, not verified post-RL improvements, and the apparent monotonic increase is within sampling noise at N=20.
The local iter-0 values are 0.78 (v2), 0.755 (v4), 0.795 (v7) — note v2's 0.78 does not match the 0.730 reported in this table, suggesting at least one number here comes from a different snapshot or source that has not been re-validated.
Treat the comparisons in this row of the document as **directional** until N=100 post-RL offline suite evals are produced for each of these runs.

Two corroborating observations that are still useful even without the suite-avg comparison:

- `success_replay_max_ratio=1.0` (run v5) regressed — the replay batch dominated and over-fit to early successes.
This is a behavioural observation independent of the suite-avg comparison.
- Single-task long runs without demos/replay drifted off-anchor (§11, verified at iter ~30 onward in `28327579` and `28335554`).

**Conclusion (tentative until re-validated)**: keep `--include-demos-in-update`, `--success-replay-total-size 320`, `--success-replay-max-ratio 0.5`, `--success-replay-alpha 1.0`, `--success-replay-ema-decay 0.8` for multi-task FPO unless explicitly ablating.
For long single-task runs (≥ 50 iters), enable them too as a regulariser against off-anchor drift — this part is supported by the `raw_sft_kl` evidence in §11.
Before basing further multi-task experimental design on the "0.730 → 0.755 → 0.795" chain, produce post-RL offline N=100 suite evals for `28236602`, `28246360`, and `28249578` and update this table.

### 15. Eval reliability under the new env: 10/50 ep are scout-only, 100 ep is the standard

All "best @ iter X" claims using `eval_episodes ≤ 50` per task are scout-only and not directly comparable to SFT or to other RL ckpts.
Normal-approximation 95 % CIs (`±1.96·√(p(1−p)/N)`) at the relevant operating points:

| Eval episodes / task | Per-task p=0.50 CI | Per-task p=0.80 CI | Suite-avg (10 tasks pooled) p=0.74 CI ⁹ |
| ---: | --- | --- | --- |
| 10 | ±31 pp | ±25 pp | ±9 pp |
| 50 | ±14 pp | ±11 pp | ±4 pp |
| 100 | ±10 pp | ±8 pp | ±3 pp |

⁹ Suite-avg pooled CI assumes 10·N independent trials at the suite mean.
If between-task variance dominates (e.g. one task at 0.40 and another at 0.99 as in eval `28192830`), the practical suite-avg CI is wider than the pooled binomial; use task-level CIs when deciding between recipes.
The previous version of this table mis-labelled these as "Wilson 95 % CIs" — Wilson is asymmetric and at small N narrower than the normal approximation; the numbers in the table are normal-approximation, not Wilson.

`28188629`'s pre-env-change eval (`28192830`, **N=100 ep**, git `e3b72285`, old LIBERO init-state distribution) reported task-5 SR of 0.84.
The post-env-change eval (`28254851`, N=100 ep, git `ef79c3b5`, new init-state distribution) reports task-5 SR of 0.46.
Both evals have 95 % CIs of roughly ±7–10 pp at N=100, so the 0.84 → 0.46 gap is **not** sampling noise — it is driven by the LIBERO init-state randomisation change at commit `ef79c3b5` (see §16).
Any "best ckpt" number produced under the pre-`ef79c3b5` env is not directly comparable to a current eval; re-eval the checkpoint under the new env before drawing conclusions.

(For completeness: the training-time eval of `28188629` at `eval_episodes=10` per iteration reported task-5 best = 0.80 — that 10-ep number has a 95 % CI of ±25 pp at p=0.80, i.e. it was never statistically distinguishable from any value in [0.55, 1.00] and is *not* the source of the 0.84 reported above.)

**Conclusion**: any eval used to choose between hyperparameter recipes or to claim "best ckpt" must be at `eval_episodes ≥ 100` per task **and** under the current env (post-`ef79c3b5`).
Below that, the evaluation is a pre-screen, not a measurement, and across-env comparisons are invalid even at N=100.

### 16. Eval-protocol notes vs published SmolVLA numbers

Worth recording so future write-ups can frame the comparison correctly:

- Paper protocol: 10 episodes per task on the LIBERO default init states 0–9 (which match the demo distribution).
- Our protocol: 100 episodes per task with `np.random.RandomState(seed).randint(num_init_states)` per episode seed (sample roughly all 50 init states ~2× each).
- See [`src/vla/envs/libero.py:70-84`](../src/vla/envs/libero.py) for the init-state randomisation and [sparse_rl_path_to_90_spatial.md §4](./sparse_rl_path_to_90_spatial.md) for the full discussion.

Practical implication: SFT n=1 = 89 % (paper) and SFT n=1 = 74.3 % (us) are not directly comparable.
The ~15 pp gap is partly explained by (a) the new env reset randomisation reaching a broader init-state distribution and (b) the paper's protocol effectively evaluating on memorised init states.
This is a feature of our setup, not a defect — see [LIBERO-PRO arXiv 2510.03827](https://arxiv.org/html/2510.03827v1) for the now-published critique of the standard protocol.

### Updated recommended configuration

For **single-task task-5 scouts** (hyperparameter A/Bs only, not for suite-avg claims):

```bash
--task-ids 5
--update-method fpo
--advantage-mode leave_one_out
--lr 3e-06
--clip-epsilon 0.05
--clip-epsilon-high 0.08
--num-fm-noise-samples 4
--fpo-negative-adv-scale 1.0
--ppo-epochs 1
--trajs-per-task 32
--kl-coeff 0.01
--sft-kl-coeff 0.02              # was 0.005, see §11
--include-demos-in-update        # new default for ≥ 50 iter runs, see §11/§14
--success-replay-total-size 320
--success-replay-max-ratio 0.5
--success-replay-alpha 1.0
--success-replay-ema-decay 0.8
--dynamic-sampling
--dynamic-sampling-max-retries 2
--n-action-steps 1               # set 5 only if eval also at 5
--eval-episodes 100              # mandatory for "best ckpt" claims, see §15
--max-steps 220
--gradient-checkpointing
```

For **multi-task all-spatial production runs** (the path to suite-avg goals):

```bash
--task-ids all
--update-method fpo
--advantage-mode leave_one_out
--lr 2e-06                       # multi-task is more sensitive; 3e-6 untested at this scale
--clip-epsilon 0.05
--clip-epsilon-high 0.08         # see clip-higher A/B in sparse_rl_path_to_90_spatial.md §8
--num-fm-noise-samples 4
--fpo-negative-adv-scale 1.0
--ppo-epochs 1
--trajs-per-task 32
--kl-coeff 0.01
--sft-kl-coeff 0.02
--include-demos-in-update
--success-replay-total-size 320
--success-replay-max-ratio 0.5
--success-replay-alpha 1.0
--success-replay-ema-decay 0.8
--dynamic-sampling
--dynamic-sampling-max-retries 2
--n-action-steps 1               # or 5 only if pending evals confirm chunk-5 transfer
--eval-episodes 100
--max-steps 220
--gradient-checkpointing
```

The two block differences from the original §"Suggested improvements" (still valid for task-2 with the old env) are: lower LR for multi-task, stronger SFT anchor, demos+replay re-enabled, dynamic sampling on by default, and `n_action_steps` made an explicit choice rather than left at the default.

---

## Data-validation log

This section records which claims in the document have been reconciled against the raw training and eval artefacts in this repo, so future experiments do not get planned on top of unchecked numbers.

### Method

Each numeric claim was checked against one or both of:

- `results/training/*.json` — per-run training summaries (config, best/final eval iter and value, history points).
- `results/training_curves/*.jsonl` — per-iteration metric streams (`sparse_rl/iteration`, `sparse_rl/eval/success_rate`, `sparse_rl/fpo_loss`, `sparse_rl/kl_penalty`, `sparse_rl/raw_sft_kl`, `sparse_rl/kl_coeff`, `_runtime`, etc.).
- `results/evals/*.json` — offline N=100 evaluation results, including the suite-avg and per-task success rates and the `training_git_commit` field that distinguishes pre- vs post-`ef79c3b5` env reset distributions.
- Source code at `src/vla/rl/rollout.py`, `src/vla/rl/policy_update/base.py`, and `src/vla/envs/libero.py` for mechanism claims.

### Last reconciliation

- Date: 2026-05-13.
- Scope: every numeric value in §1–§16, the run-summary table, and the appendix run-summary table.

### Corrections applied in this revision

The following claims were inconsistent with the underlying data and have been corrected in this revision (see git diff for exact locations):

1. **Run 2 iters and final SR**: configured 200 but actually stopped at iter 126; the last logged eval is 94 % at iter 120, not "~82 %".
The late-phase evals oscillate 78–94 %.
2. **Run 2 oscillation sequence** in §5: the last two values (iters 110, 120) are 78 % then 94 %, not 94 % then 78 %.
3. **§1 KL-range table**: Run 2's late-phase column was previously labelled "pre-collapse" — Run 2 did not collapse, the late phase is stagnation.
Column headers are now "early/learning phase" and "late phase" with per-row clarification.
4. **§4 LOO-advantage worked example**: with 28/32 successes, the LOO advantage for a failure is `0 − 28/31 ≈ −0.90`, not −1.0.
5. **§10 wall-clock claim**: chunk-5 is ~3× faster per iteration than chunk-1 on these workloads, not H×.
The §10 table now reports measured `min / iter` and total wall-clock for each run from `_runtime` deltas.
6. **§10 table row for 28267780**: previously "≤ 25 in 24 h"; actually 30 iters in 6.4 h (~12.7 min/iter).
7. **§11 row for 28249578**: the local `spatial-all-v7-…_28249578.jsonl` contains only 3 rows (iter-0 eval plus iters 1–2), so the "binding" diagnosis at "~25 iters" rested on 2 observations.
This is now disclosed in a footnote and the longer-iteration claim is marked unverified-locally.
8. **§13 multi-task improvement narrative**: the "0.755 / 0.795 suite-avg best" values in the run-summary table are pre-RL evals at iter 0 with `eval_episodes=20`, not post-RL improvements, because the local training-curve files for `28246360` and `28249578` contain only iter-0 eval data.
The "0.743 → 0.755 → 0.795 without forgetting" claim is now flagged as a working hypothesis pending an N=100 post-RL offline suite eval.
9. **§15 "Wilson 95 % CIs" table**: relabelled as normal-approximation (which is what the numbers are).
Suite-avg column re-derived as the 10×N pooled binomial CI, with a caveat that between-task variance can widen it in practice.
10. **§15 28188629 old-eval framing**: the 0.84 task-5 number is from offline eval `28192830` at **N=100** under the **pre-`ef79c3b5`** env, not "10 ep × 10 tasks".
The 0.84 → 0.46 gap is driven by the env reset randomisation change, not sampling noise; both numbers have 95 % CIs of roughly ±7–10 pp at N=100.
The training-time `eval_episodes=10` reading of the same checkpoint was 0.80 (95 % CI ±25 pp at p=0.80), kept as a separate, less reliable measurement.
11. **Appendix row 28188629 iters**: 25 → 30 (`history_points=31`, `final_eval_iteration=30`).
Run 2 footnotes (⁶, ⁷) and §11 footnote (⁸) and §15 footnote (⁹) were added.

### Claims that remain in the document and were confirmed against the data

- All 10 task-2 runs' hyperparameters (LR, clip ε / ε_high, neg-adv scale, iters, max_steps, trajs, ppo_epochs, eval_episodes).
- All 10 task-2 runs' best-iter and final-iter eval values.
- Run 5 / 6 / 9 / 10 collapse signatures: final 50 %, 4 %, 12 %, 16 %.
- Run 7 rollout SR climb 56 % (iter 2) → 95 % (iter 32) and ~22 min/iter wall-clock.
- Run 8 "positive FPO losses at iters 33, 38, 39, 42" (actual values +0.00022, +0.00327, +0.00119, +0.00085).
- Run 8 eval trajectory 90 → 89 → 85 → 76 over iters 30–60.
- Run 9 / 10 `raw_kl` stays far below 0.01 (~1e-4 to 5e-3) and `kl_coeff` collapses from 6.67e-3 to 1e-6 within a few tens of iterations.
- `raw_sft_kl` means for 28327579 (1.397), 28335554 (1.413), and 28263586 (0.003).
- Eval `28254851`: suite-avg 0.72, task-5 0.46; SFT eval `28242119` (git `ef79c3b5`): suite-avg 0.743, task-5 0.37.
- Source-code references `src/vla/envs/libero.py:70-84` (init-state randomisation), `src/vla/rl/rollout.py:228-340` (chunk execution, function actually 230–342), and `src/vla/rl/policy_update/base.py:89-139` (`_actions_and_mask_for_loss`, function actually 98–148).

### Claims that could not be verified from this repo and require external evidence before being used to plan experiments

- §6 wall-clock figure "~13 min" for the 32-traj task-2 baseline (Run 2 measures 11.3 min/iter — close, but if you want the exact number for budgeting, use 11.3).
- §13 "multi-task setting monotonically improves suite avg from SFT (0.743 → 0.755 → 0.795) without forgetting" — needs a post-RL N=100 offline suite eval of `28246360` and `28249578` to confirm.
- §16 SmolVLA paper protocol (10 eps per task on init states 0–9, 89 % suite-avg) — sourced externally; the ~15 pp gap to our 74.3 % SFT is reported here for context only and is not independently verifiable from this repo.
- §7 code-level mechanism for why `ppo_epochs=2` corrupts the trust region — the empirical degradation is verified, but the "stale FM-loss ratio on the second pass" explanation should be confirmed against the current FPO update implementation before being treated as established.

### How to extend this log

When adding a new run or claim:

1. Cite the `lsf_job_id` and the exact `results/training/...json` / `results/training_curves/...jsonl` / `results/evals/...json` files used.
2. For any eval comparison, record the `training_git_commit` of each eval — pre- vs post-`ef79c3b5` evals are not directly comparable.
3. For any "best ckpt" or "suite-avg" claim, require `eval_episodes ≥ 100` per task at the current env commit; flag scout-only numbers (≤50 ep) explicitly.
4. For any wall-clock figure, derive it from `_runtime` deltas in the jsonl rather than from a configured iteration count.
