# SmolVLA Baseline Evaluation — LIBERO

**Model:** `HuggingFaceVLA/smolvla_libero`  
**Date:** 2026-03-09  
**Protocol:** 10 tasks × 20 episodes = 200 episodes per suite

## Summary

| Suite | `pc_success` | `eval_s` | `eval_ep_s` |
| ----- | ------------ | -------- | ----------- |
| `libero_spatial` | **71.5%** | 3721 s (~62 min) | 18.6 s |
| `libero_object` | **90.0%** | 4060 s (~68 min) | 20.3 s |

## Command

```sh
uv run lerobot-eval \
    --policy.path=HuggingFaceVLA/smolvla_libero \
    --env.type=libero \
    --env.task=<suite> \
    --eval.batch_size=1 \
    --eval.n_episodes=20 \
    --policy.n_action_steps=10
```

---

## libero_spatial

### Per-Task Results

| Task ID | Success Rate |
| ------- | ------------ |
| 0 | 80.0% |
| 1 | 85.0% |
| 2 | 55.0% |
| 3 | 70.0% |
| 4 | 85.0% |
| 5 | 30.0% |
| 6 | 75.0% |
| 7 | 70.0% |
| 8 | 95.0% |
| 9 | 70.0% |

### Aggregate Metrics

| Metric | Value |
| ------ | ----- |
| `pc_success` | **71.5%** |
| `avg_sum_reward` | 0.715 |
| `avg_max_reward` | 0.715 |
| `n_episodes` | 200 |
| `eval_s` (total) | 3721 s (~62 min) |
| `eval_ep_s` (per episode) | 18.6 s |

---

## libero_object

### Per-Task Results

| Task ID | Success Rate |
| ------- | ------------ |
| 0 | 85.0% |
| 1 | 90.0% |
| 2 | 100.0% |
| 3 | 70.0% |
| 4 | 100.0% |
| 5 | 85.0% |
| 6 | 100.0% |
| 7 | 95.0% |
| 8 | 100.0% |
| 9 | 75.0% |

### Aggregate Metrics

| Metric | Value |
| ------ | ----- |
| `pc_success` | **90.0%** |
| `avg_sum_reward` | 0.900 |
| `avg_max_reward` | 0.900 |
| `n_episodes` | 200 |
| `eval_s` (total) | 4060 s (~68 min) |
| `eval_ep_s` (per episode) | 20.3 s |

---

## Configuration

Identical across both runs.

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
