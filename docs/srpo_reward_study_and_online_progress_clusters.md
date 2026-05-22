# SRPO Reward Study and Online Progress-Cluster Reward

## Purpose

The current SRPO path should not be judged only by online training curves. If
the world-model reward does not already separate success from failure and does
not track progress offline, online RL will likely amplify reward noise.

This note records the proposed reward study and a more plausible online reward
variant: comparing cumulative sliding-window rollout prefixes to progress
clusters built from successful trajectories.

## Why Study The Reward First

The most important question is:

> Does the reward separate successful behavior from failed behavior, and does it
> increase monotonically with task progress?

If the answer is no, SRPO/FPO does not have a valid dense learning signal yet.
More online training would mostly optimize embedding artifacts.

The `cluster-viz` branch contains useful diagnostic scaffolding for this:

- `scripts/reward_study.py`
- `scripts/visualize_clusters.py`
- `scripts/visualize_demo_distances.py`
- `src/vla/diagnostics/reward_analysis.py`
- `src/vla/diagnostics/clustering.py`
- `src/vla/diagnostics/collect_trajectories.py`

That branch is too divergent to merge directly, but those diagnostics are worth
porting or reimplementing on the current branch.

## Offline Reward Study

Start with fixed trajectory groups per task:

- Demonstrations.
- SFT successes.
- SFT failures.
- Random-policy failures.
- Synthetic progress rollouts at 0%, 10%, ..., 100%, made by replaying a
  reference trajectory prefix and then switching to random or policy actions.

Recommended initial tasks:

- `spatial:2`: sparse FPO already looked relatively strong here.
- `spatial:5`: SRPO/sparse training has shown weaker behavior here.
- `object:0`: cross-suite sanity check.

Compare at least these reward variants:

- Current SRPO reward implementation.
- siiRL-faithful reward:
  - 64 sampled frames per trajectory.
  - V-JEPA/V-JEPA2 trajectory embedding path.
  - `StandardScaler` before DBSCAN.
  - DBSCAN with `eps=0.5`, `min_samples=2`.
  - Mean-success fallback if DBSCAN returns no clusters.
  - Failure reward from nearest-center distance.
- Current z-score failure reward versus min-max normalized failure reward.
- Terminal-trajectory reward versus sliding-window progress reward.

Offline metrics:

- Success/failure AUROC.
- Cohen's d between success and failure distance distributions.
- Histogram overlap.
- Spearman correlation between synthetic progress and reward.
- DBSCAN diagnostics:
  - number of clusters,
  - number of noise points,
  - singleton/all-noise collapse,
  - mean intra-cluster and inter-cluster distances.

Go/no-go criteria before expensive online SRPO:

- Success/failure AUROC is clearly above chance, ideally around `0.8+`.
- Reward versus synthetic progress has clear positive Spearman correlation.
- Random failures score below SFT failures.
- SFT failures score below successes/demos.
- DBSCAN does not collapse into all noise or one center per reference without
  reporting that fallback explicitly.

## Online Sliding-Window Reward Idea

The paper overview describes a cumulative sliding-window reward: prefixes of a
trajectory are encoded and compared against successful behavior. This is likely
more useful than scoring only the final trajectory embedding.

However, the online version should not simply compare arbitrary failed prefixes
to terminal success clusters. Early useful behavior may still be far from the
final success state. Reaching, grasp alignment, contact, and partial transport
can be valuable progress even when the final image is still far away.

The more plausible online formulation is a progress-cluster reward.

## Progress-Cluster Reward

For each successful demo or online success:

1. Build cumulative prefixes, for example:
   - `0:10`
   - `0:16`
   - `0:32`
   - ...
   - `0:T`
2. Encode each prefix with the same world-model path used by SRPO.
3. Assign each prefix a progress label, such as `0.1`, `0.2`, ..., `1.0`.
4. Cluster embeddings per progress bin, or store centers per progress bin.

For each current rollout:

1. Build cumulative prefixes from the rollout.
2. Encode each prefix.
3. Compare each prefix embedding to the progress-bin centers.
4. Estimate the highest progress bin that the rollout reached.
5. Convert that progress estimate into a scalar trajectory reward.

Conceptually:

```text
successful references
  -> cumulative prefix embeddings
  -> progress-bin DBSCAN centers

current rollout
  -> cumulative prefix embeddings
  -> nearest progress-bin center
  -> estimated progress curve
  -> scalar g_i for SRPO/FPO
```

Initial scalar reward candidates:

```text
g_i = alpha * max_progress_seen
success => 1.0
```

Alternatives to compare:

```text
g_i = alpha * final_progress
g_i = alpha * mean(top_k_progress)
g_i = alpha * (final_progress + progress_slope_bonus)
```

The first candidate, `max_progress_seen`, is the safest starting point because a
LIBERO rollout can make real progress and still fail late. A pure
`final_progress` reward may punish useful exploration too harshly.

## Important Implementation Details

The study should keep the embedding path fixed across variants whenever
possible. Otherwise it becomes unclear whether a result came from the reward
formula or from the encoder/preprocessing path.

Key details to verify:

- Use the same image normalization/cropping path for references and rollouts.
- Use `StandardScaler` consistently when matching the siiRL-style DBSCAN path.
- Store the fitted scaler alongside cluster centers.
- Report fallback behavior explicitly:
  - no clusters,
  - all noise,
  - too few references,
  - singleton-heavy clusters.
- Keep task-specific reward models and clusters isolated by task.
- Log reward distributions for successes and failures separately.

## Online Training Gate

Do not use the online progress-cluster reward for full SRPO training until an
offline study shows:

- Demo prefixes order correctly by progress.
- Synthetic progress rollouts produce mostly monotonic progress curves.
- Successful final prefixes land near `0.9-1.0`.
- Random rollouts remain low.
- Failed SFT rollouts show partial progress when they visually approach the
  task goal.

If those checks fail, fix the reward before running more online SRPO. The
reward study is cheaper and more diagnostic than another training run.

