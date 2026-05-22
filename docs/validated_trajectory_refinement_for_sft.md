# Validated Trajectory Refinement For SFT

## Purpose

This note records a data-generation idea for improving SmolVLA without relying
on an online RL reward signal.

The idea is to treat successful policy rollouts like collected demonstrations:
collect successful trajectories from the current SFT baseline, optimize their
motion offline, replay the optimized trajectories in the simulator, record fresh
observations and actions, and then train SFT as if these were newly collected
episodes.

This is better described as validated trajectory refinement or synthetic
demonstration generation, not pure offline RL.

## Core Pipeline

```text
current SFT baseline
  -> collect many successful rollouts
  -> optionally perturb/reset initial states
  -> optimize trajectory commands
  -> replay from chosen initial states
  -> record fresh observation/action episodes
  -> train SFT on accepted episodes
```

The key rule is:

> If the actions are optimized, the observations must be recorded again during
> replay.

Do not pair optimized actions with the original images. The optimized path will
usually produce a different visual trajectory.

## Why This Could Help

The current SFT policy may already know how to solve some tasks, but its
successful trajectories can still be noisy:

- jerky arm motion,
- excessive acceleration,
- stop-start hesitation,
- action saturation,
- redundant recovery movements,
- unstable gripper timing,
- long paths,
- inconsistent action chunks.

If successful rollouts are cleaned up and replay-validated, they can become
higher-quality behavior cloning data.

The framing is similar to collecting additional teleoperated data:

```text
human demos
+ baseline successful rollouts
+ optimized validated replays
```

The generated trajectories should be stored and trained exactly like collected
successful demonstrations, with metadata describing how they were produced.

## Candidate Optimization Targets

Start with action-space refinement because it is easiest to implement from the
current trajectory buffers.

Useful quality terms:

- action smoothness: penalize large `a_t - a_{t-1}`,
- acceleration: penalize large second differences,
- jerk: penalize large third differences,
- action saturation: penalize values close to action limits,
- path length: prefer shorter successful paths,
- stop-start oscillation: penalize repeated near-zero / large / near-zero motion,
- gripper toggles: penalize unnecessary open-close changes,
- action-chunk consistency: prefer chunks that agree with later executed motion.

If reliable end-effector or object state is available, later versions can add:

- end-effector velocity smoothness,
- end-effector acceleration,
- end-effector jerk,
- object velocity/acceleration smoothness,
- contact stability,
- distance-to-waypoint constraints around grasp/lift/place events.

## Gripper Handling

The gripper should not be blindly smoothed like arm actions.

Treat gripper behavior as discrete events:

- open,
- close,
- hold.

The optimizer may shift event timing slightly, but it should avoid creating
intermediate gripper values unless the action space explicitly expects them.
Preserving grasp and release timing is more important than making the gripper
signal numerically smooth.

## Simple First Optimizer

A minimal first version:

```text
successful rollout actions
  -> smooth arm action dimensions with Savitzky-Golay or cubic spline
  -> preserve gripper events
  -> clamp actions to valid bounds
  -> replay from same initial state
  -> record fresh observations/actions
  -> keep only if replay succeeds and quality improves
```

Example objective:

```text
J =
  w_anchor * waypoint_deviation
+ w_vel    * mean(||v_t||^2)
+ w_acc    * mean(||a_t||^2)
+ w_jerk   * mean(||j_t||^2)
+ w_sat    * action_saturation_penalty
+ w_grip   * gripper_toggle_penalty
```

The anchor term prevents the optimizer from destroying important task events.
Important anchors include contact, grasp, lift, transport, and placement.

## Replay Validation

Replay validation is mandatory.

Accept an optimized trajectory only if:

```text
task_success == true
simulator_stable == true
actions_within_bounds == true
gripper_timing_plausible == true
motion_quality_after >= motion_quality_before
trajectory_deviation <= allowed_limit
```

Rejected candidates are still useful diagnostics, but they should not be added
to the SFT dataset.

## Perturbations And Diversity

The first milestone should use the exact same initial state as the original
successful rollout. This validates that saved seeds/states/actions can be
replayed reliably.

After exact replay works, add diversity:

- object pose jitter,
- robot initial pose jitter,
- camera/light/background perturbations if supported,
- small waypoint perturbations before smoothing,
- timing variation,
- action noise followed by smoothing,
- slightly delayed or advanced gripper events,
- different smoothing strengths.

The goal is to avoid producing many nearly identical copies of the same
trajectory. The generated data should improve both quality and coverage.

## Recommended Development Order

1. Replay exact original successes.
2. Smooth exact successes and replay them.
3. Record fresh observation/action episodes from successful smoothed replays.
4. Train SFT with the same hyperparameters as the current SFT baseline.
5. Compare baseline SFT versus generated-data SFT.
6. Add initial-state perturbations only after exact replay is reliable.
7. Generate multiple validated variants per source trajectory.

## Dataset Metadata

Each generated episode should carry enough metadata to reproduce and audit it:

```json
{
  "source": "sft_rollout_success",
  "source_checkpoint": "...",
  "task": "spatial:5",
  "init_seed": 123,
  "initial_state_id": "...",
  "perturbation": {},
  "optimizer": "savgol_action_smoothing_v1",
  "quality_before": {
    "smoothness": 0.0,
    "acceleration": 0.0,
    "jerk": 0.0,
    "saturation": 0.0,
    "gripper_toggles": 0
  },
  "quality_after": {
    "smoothness": 0.0,
    "acceleration": 0.0,
    "jerk": 0.0,
    "saturation": 0.0,
    "gripper_toggles": 0
  },
  "replay_success": true
}
```

## Training Use

The generated data can be used as standard successful demonstrations:

```text
original demos
+ accepted optimized replays
```

Two simple training variants:

```text
plain SFT:
  all accepted generated episodes have normal BC weight

weighted SFT:
  optimized validated success weight = 1.0
  original noisy success weight      = 0.7
```

The first ablation should keep the same SFT hyperparameters as the current
baseline. That makes the effect of the generated trajectories easier to
measure.

## Evaluation

Compare:

- current SFT baseline,
- SFT on original demos plus policy successes,
- SFT on original demos plus optimized validated replays,
- optionally SFT on original demos plus both original and optimized successes.

Primary metric:

- held-out LIBERO success rate.

Secondary metrics:

- average episode length,
- action smoothness,
- acceleration,
- jerk,
- action saturation,
- gripper toggles,
- failure mode changes.

The expected benefit is not just more data. The expected benefit is cleaner,
more canonical successful behavior that the policy can imitate more reliably.

