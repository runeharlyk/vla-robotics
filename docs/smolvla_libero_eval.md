# SmolVLA Baseline Evaluation — LIBERO

**Model:** `HuggingFaceVLA/smolvla_libero`  
**Run directory:** `outputs/eval/2026-03-25/22-32-46_libero_smolvla`  
**Eval finished (log):** 2026-03-26 06:50:23 (`bot_eval.py`)  
**Protocol:** 10 tasks × 100 episodes = 1000 episodes (`libero_spatial`)

## Summary

| Suite | `pc_success` | `eval_s` | `eval_ep_s` |
| ----- | ------------ | -------- | ----------- |
| `libero_spatial` | **74.1%** | 29678 s (~8.2 h) | 29.7 s |
| `libero_object` | **86.3%** | 24585 s (~6.8 h) | 24.6 s |

## libero_spatial

### Command

```sh
uv run lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.control_mode=relative \
  --eval.batch_size=8 \
  --eval.n_episodes=100 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --env.max_parallel_tasks=1 \
  --seed=42
```

This aggregate run used **1000** episodes (100 per task).

### Per-Task Results

| Task ID | Success Rate |
| ------- | ------------ |
| 0 | 71.0% |
| 1 | 85.0% |
| 2 | 87.0% |
| 3 | 61.0% |
| 4 | 83.0% |
| 5 | 52.0% |
| 6 | 80.0% |
| 7 | 81.0% |
| 8 | 83.0% |
| 9 | 58.0% |

### Aggregate Metrics

| Metric | Value |
| ------ | ----- |
| `pc_success` | **74.1%** |
| `avg_sum_reward` | 0.741 |
| `avg_max_reward` | 0.741 |
| `n_episodes` | 1000 |
| `eval_s` (total) | 29678.33 s (~494.6 min) |
| `eval_ep_s` (per episode) | 29.678 s |

Recorded videos: 10 episodes per task (100 MP4 files total) under `outputs/eval/2026-03-25/22-32-46_libero_smolvla/videos/libero_spatial_<task_id>/eval_episode_*.mp4`.
Aggregate metrics cover all 1000 episodes.

---

## libero_object

**Run directory:** `outputs/eval/2026-03-26/11-03-15_libero_smolvla`  
**Eval finished (log):** 2026-03-26 17:55:45 (`bot_eval.py`)  
**Protocol:** 10 tasks × 100 episodes = 1000 episodes (`libero_object`)

On this suite, the same policy reaches **86.3%** success versus **74.1%** on `libero_spatial`.
Wall time per episode is lower (~24.6 s vs ~29.7 s).

### Command

```sh
uv run lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero \
  --env.task=libero_object \
  --env.control_mode=relative \
  --eval.batch_size=8 \
  --eval.n_episodes=100 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --env.max_parallel_tasks=1 \
  --seed=42
```

### Per-Task Results

| Task ID | Success Rate |
| ------- | ------------ |
| 0 | 69.0% |
| 1 | 94.0% |
| 2 | 94.0% |
| 3 | 97.0% |
| 4 | 84.0% |
| 5 | 71.0% |
| 6 | 98.0% |
| 7 | 81.0% |
| 8 | 97.0% |
| 9 | 78.0% |

### Aggregate Metrics

| Metric | Value |
| ------ | ----- |
| `pc_success` | **86.3%** |
| `avg_sum_reward` | 0.863 |
| `avg_max_reward` | 0.863 |
| `n_episodes` | 1000 |
| `eval_s` (total) | 24585.22 s (~409.8 min) |
| `eval_ep_s` (per episode) | 24.585 s |

Recorded videos: 10 episodes per task (100 MP4 files total) under `outputs/eval/2026-03-26/11-03-15_libero_smolvla/videos/libero_object_<task_id>/eval_episode_*.mp4`.
Aggregate metrics cover all 1000 episodes.

---

## Raw metrics (eval log)

`libero_spatial` — overall aggregated metrics:

```text
{'avg_sum_reward': 0.741, 'avg_max_reward': 0.741, 'pc_success': 74.1, 'n_episodes': 1000, 'eval_s': 29678.333701610565, 'eval_ep_s': 29.678333701848985, 'video_paths': [...]}
```

`libero_object` — overall aggregated metrics:

```text
{'avg_sum_reward': 0.863, 'avg_max_reward': 0.863, 'pc_success': 86.3, 'n_episodes': 1000, 'eval_s': 24585.224433660507, 'eval_ep_s': 24.585224434375764, 'video_paths': [...]}
```

The log also includes full `per_task` and `per_group` structures.
Those entries include `sum_rewards`, `max_rewards`, `successes`, and `video_paths` per task.

---

## Configuration

| Parameter | Value |
| --------- | ----- |
| VLM backbone | `HuggingFaceTB/SmolVLM2-500M-Instruct` |
| `chunk_size` | 50 |
| `n_action_steps` | 10 |
| `n_obs_steps` | 1 |
| `attention_mode` | `cross_attn` |
| `self_attn_every_n_layers` | 2 |
| `num_expert_layers` | -1 (all) |
| `expert_width_multiplier` | 0.5 |
| `freeze_vision_encoder` | True |
| `train_expert_only` | True |
| `resize_imgs_with_padding` | (512, 512) |
| Input image resolution (policy) | 256 × 256 |
| Observation image resolution (env) | 360 × 360 |
| State dim | 8 |
| Action dim | 7 |
| Cameras | `agentview_image`, `robot0_eye_in_hand_image` |
| Control mode | relative |
| `use_amp` | False |
| `use_peft` | False |
| Seed | 1000 |
