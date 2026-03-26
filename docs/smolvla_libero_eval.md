# SmolVLA Baseline Evaluation — LIBERO

**Model:** `HuggingFaceVLA/smolvla_libero`  
**Run directory:** `outputs/eval/2026-03-25/22-32-46_libero_smolvla`  
**Eval finished (log):** 2026-03-26 06:50:23 (`bot_eval.py`)  
**Protocol:** 10 tasks × 100 episodes = 1000 episodes (`libero_spatial`)

## Summary

| Suite | `pc_success` | `eval_s` | `eval_ep_s` |
| ----- | ------------ | -------- | ----------- |
| `libero_spatial` | **74.1%** | 29678 s (~8.2 h) | 29.7 s |

## Command

```sh
uv run lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.batch_size=1 \
    --eval.n_episodes=100 \
    --policy.n_action_steps=10
```

Adjust `eval.n_episodes` and task wiring if your CLI differs.
This aggregate run used **1000** episodes on `libero_spatial` (100 per task).

---

## libero_spatial

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

## Raw metrics (eval log)

Overall aggregated metrics:

```text
{'avg_sum_reward': 0.741, 'avg_max_reward': 0.741, 'pc_success': 74.1, 'n_episodes': 1000, 'eval_s': 29678.333701610565, 'eval_ep_s': 29.678333701848985, 'video_paths': [...]}
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
