# Visual Diagnostics — Ranked Image Analysis

All 18 plots from the SmolVLA visual noise-robustness experiment, ranked from most informative to least. Each image is accompanied by a description of what it reveals about the policy's sensitivity to visual corruptions.

---

## 🥇 1. Severity Degradation by Suite (Faceted Line Plot)

![Severity by suite](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/severity_by_suite.png)

**Why #1**: This is the single most informative plot because it reveals the full story — how each noise type degrades performance across severity levels, broken down per suite. Key findings:

- **Object suite is the most fragile**: Starting from a ~91% clean baseline, success crashes to near-zero by severity 3 for all noise types except fog. The object-recognition tasks are devastatingly sensitive to any visual perturbation.
- **Fog is consistently the least disruptive** corruption across all three suites — the blue line always sits highest. This makes sense: fog preserves object shapes while only reducing contrast.
- **Zoom blur is the most resilient at s1** (especially spatial: ~87%) but **collapses dramatically** from s1→s3, making it the most "cliff-like" corruption — the policy has a sharp perceptual threshold for zoom distortion.
- **Glass blur is devastating even at s1** — particularly in goal and object suites where it immediately drops success to ~25% and ~5% respectively. The local pixel-shuffling clearly destroys the fine spatial features the policy relies on.
- **Spatial suite retains the most residual capability** at higher severities, likely because spatial tasks rely more on coarse relative positions than fine object features.

---

## 🥈 2. Task Sensitivity Heatmap — Severity 3

![Task heatmap s3](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/task_noise_drop_heatmap_s3.png)

**Why #2**: This is the most granular view, showing exactly which tasks break under which corruptions. At the standard test point (severity 3):

- **Near-total failure is the norm**: Most cells show drops of 0.9+ (dark red), meaning the policy goes from functional to almost completely broken under moderate noise.
- **"Pick up the bbq sauce" (O3) is the most fragile task**: 0.96–0.98 drop across all noise types — the task essentially fails 100% under any corruption.
- **"Put the bowl on top of the cabinet" (G4) shows anomalous fog resilience**: The fog drop is -0.07 (i.e. noise *helps*), while glass_blur and motion_blur still cause 0.45 drops. This suggests the task may benefit from the contrast reduction fog provides.
- **"Pick up the black bowl from table center" (S2) has the most variable response**: fog=0.22, glass_blur=0.40, zoom_blur=-0.04. The negative zoom_blur drop means the zooming corruption actually improves performance on this task, possibly by centering the relevant object.
- **Goal tasks (G0–G4) generally show lower drops than object tasks**, suggesting multi-step goal tasks may rely on coarser visual signals than precise object grasping.

---

## 🥉 3. Success vs Severity by Noise Type (Aggregate)

![Success vs severity](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/success_vs_severity_by_noise.png)

**Why #3**: The clearest summary of the overall trend. Key observations:

- **The clean baseline is 80%**, and *every* noise type at *every* severity falls below it — there is no corruption the policy is robust to.
- **All curves converge toward ~5% at severity 5**, indicating near-total policy failure under extreme corruption regardless of noise type.
- **The ranking at s1 (zoom > fog > gaussian > motion > glass) completely inverts by s5**, demonstrating that severity sensitivity varies non-linearly across noise types. Zoom blur goes from best (72.7%) to worst (2.7%).
- **Fog degrades the most gracefully** — its curve is the shallowest, maintaining 35% success at s3 when everything else is below 13%.
- **Confidence intervals widen notably at mid-severities** (s2–s3), reflecting high task-to-task variance at the "interesting" degradation point.

---

## 4. Noise Robustness Radar — Severity 1 vs 3 vs 5

Comparing across severities reveals how the robustness "shape" collapses:

````carousel
![Radar s1](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/noise_robustness_radar_s1.png)
<!-- slide -->
![Radar s3](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/noise_robustness_radar_s3.png)
<!-- slide -->
![Radar s5](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/noise_robustness_radar_s5.png)
````

**At severity 1**: The radar shows differentiated, spread-out polygons — spatial (green) dominates with large area, object (orange) follows, goal (blue) is the smallest. Zoom_blur and fog axes extend furthest, confirming the policy handles these mild corruptions relatively well.

**At severity 3**: The polygons collapse dramatically. The only remaining "spike" is the fog axis, particularly for the spatial and object suites — fog is the only noise the policy shows any residual tolerance for at moderate severity.

**At severity 5**: Near-total collapse. All three suite polygons shrink to a tiny cluster near the origin. The spatial suite (green) retains a barely-visible edge on fog (~20%), but all other noise×suite combinations are essentially zero. The suite differences that were so prominent at s1 have been wiped out.

---

## 5. Success Rate Drop by Noise Type (Δ Bars) — Severity 1 vs 3 vs 5

````carousel
![Delta bars s1](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/success_delta_bars_s1.png)
<!-- slide -->
![Delta bars s3](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/success_delta_bars_s3.png)
<!-- slide -->
![Delta bars s5](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/success_delta_bars_s5.png)
````

**At severity 1**: The most revealing — significant variation exists across noise types. Glass_blur causes the largest drop (~0.63–0.67) across goal and object suites, while zoom_blur is near-zero or even slightly *negative* for goal and spatial (the policy performs about the same or marginally better). The object suite (orange) consistently shows the highest drops.

**At severity 3**: Drops are uniformly massive (0.6–0.9). Object suite shows ~0.9 drops across all noise types. Fog still stands out as the mildest (0.3 for goal and spatial).

**At severity 5**: Everything has converged to maximum damage. The bars are nearly uniform at 0.65 (goal), 0.91 (object), and 0.62–0.75 (spatial). Noise type differentiation has essentially disappeared — at extreme corruption, it doesn't matter *how* the image is corrupted.

---

## 6. Suite Sensitivity Heatmaps — Severity 1 vs 3 vs 5

````carousel
![Suite heatmap s1](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/suite_noise_drop_heatmap_s1.png)
<!-- slide -->
![Suite heatmap s3](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/suite_noise_drop_heatmap_s3.png)
<!-- slide -->
![Suite heatmap s5](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/suite_noise_drop_heatmap_s5.png)
````

A compact 3×5 grid showing the drop from clean for each suite × noise type.

- **s1**: Clear gradient from zoom_blur (least disruptive, spatial even shows -0.02 = slight improvement) to glass_blur (most disruptive, 0.42–0.67 drop). Object suite already in deep trouble.
- **s3**: Object suite shows uniformly catastrophic drops (0.74–0.91). Fog remains the mildest corruption (0.30 for both goal and spatial).
- **s5**: Near-uniform dark red. Object at 0.91 everywhere, noise-type differentiation essentially gone. The heatmap has become monotone — all corruptions are equally destructive at extreme severity.

---

## 7. Noise Ranking (Bar Charts) — Severity 1 vs 3 vs 5

````carousel
![Ranking s1](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/noise_ranking_s1.png)
<!-- slide -->
![Ranking s3](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/noise_ranking_s3.png)
<!-- slide -->
![Ranking s5](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/noise_ranking_s5.png)
````

These rank the noise types by absolute success rate:

- **s1**: Zoom_blur (72.7%) is closest to baseline (80%), while glass_blur (22.3%) is worst. Clear separation between noise types.
- **s3**: Fog (35.1%) is the clear leader, everything else below 13%. The gap between fog and the rest has widened dramatically.
- **s5**: Total collapse. Fog barely leads at 6.7%, all others at 2.7–4.3%. The ranking becomes meaningless when nothing works.

---

## 8. Task Sensitivity Heatmaps — Severity 1 and 5

````carousel
![Task heatmap s1](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/task_noise_drop_heatmap_s1.png)
<!-- slide -->
![Task heatmap s5](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/task_noise_drop_heatmap_s5.png)
````

**At severity 1**: The most visually interesting heatmap — a mix of blue (negative drops = noise helps), light peach, and dark red. The bottom row (S0: "pick up the black bowl between the plate and...") shows *negative* drops for fog, gaussian_blur, glass_blur, and zoom_blur, meaning mild noise actually *improves* the policy's success on this task. This suggests possible overfitting in the clean condition or beneficial regularization from mild perturbation.

**At severity 5**: Almost uniformly dark red. Even the most robust tasks (G3: "open the top drawer") show 0.33 drops. The task-level structure visible at s1 has been completely obliterated — extreme noise is a universal equalizer.

---

## 9. Episode Length Distribution

![Episode length](C:/Users/canic/.gemini/antigravity-ide/brain/7fe21ad3-521c-4b94-82e1-91c65d66168b/episode_length_distribution.png)

**Ranked last** not because it's unimportant, but because the finding is straightforward: the box plot is almost entirely green (success) with red (failure) boxes virtually invisible — there are so few successes under noise that the failure boxes dominate. Key observations:

- **Median episode length is ~110–125 steps** across all noise types, regardless of success/failure. This means failed episodes run for about the same duration as successful ones — the policy doesn't "fail fast," it **keeps trying** until timeout.
- **Max-steps (280) timeout line** shows scattered outliers reaching it, confirming some episodes exhaust the entire budget.
- **Glass_blur and motion_blur have slightly higher median lengths**, potentially indicating the policy "struggles" more before eventually failing.
- The uniformity across noise types suggests that episode length is not a useful diagnostic for distinguishing corruption types — success rate is the cleaner signal.

---

## Summary of Key Findings

| Finding | Evidence |
|---------|----------|
| **Glass blur is the most devastating corruption** | Lowest success at s1, causes the largest drops even at mild severity |
| **Fog is the most tolerable corruption** | Highest residual success at all severity levels, preserves object shapes |
| **Object suite is disproportionately fragile** | 0.91 drop at s3 across all noise types, vs 0.65 for goal and 0.70 for spatial |
| **Zoom blur has a cliff between s1 and s3** | Goes from best (72.7%) to near-worst (2.7%) — brittle perceptual threshold |
| **Some tasks benefit from mild noise** | Negative drops at s1 for certain spatial tasks (S0, S3) — possible overfitting |
| **At extreme severity, noise type doesn't matter** | All corruptions converge to ~3–7% success at s5 |
