# LIBERO Spatial RL Experiment Plan

This is the tactical plan for the next LIBERO spatial RL runs.
The broader thesis framing remains in [thesis_research_plan.md](./thesis_research_plan.md).

## Current State

Target metric:

- suite: `libero_spatial`
- final protocol: `10 tasks x 100 episodes = 1000 episodes`
- old documented SFT baseline: **80.9%**

Important provenance:

- `spatial_task_5_seed42_28188629` was trained at commit `e3b72285` on 2026-04-11.
- LIBERO reset/init-state randomization landed at `3c291f23` and was merged as `ef79c3b5` on 2026-04-20.
- Therefore `28188629` was **not trained** under the new initial-state behavior. If it beats SFT in current eval, that is a new-eval robustness result, not proof the training distribution already used the new reset behavior.

Current local evals under the newer reset/init-state behavior:

| Checkpoint | Full-suite SR | Task 5 SR | Notes |
| --- | ---: | ---: | --- |
| SFT `HuggingFaceVLA/smolvla_libero`, eval `28242119` | 74.3% | 37% | current SFT reference in local JSONs |
| RL `28188629/best`, eval `28254851` | 72.0% | 46% | better on trained task, not better full-suite |
| RL `28188629/best`, old eval `28192830` | 78.8% | 84% | old eval/protocol; do not compare directly to current reset eval |

## What We Learned

- `lr=5e-6` is still unsafe. It repeatedly produced late collapse, including adaptive-KL variants.
- Keep FPO at `ppo_epochs=1`, `clip_epsilon=0.05`, `clip_epsilon_high=0.08`, `num_fm_noise_samples=4`.
- `num_fm_noise_samples=1` was worse: task-5 run `28247219` peaked at 70% and ended at 50%.
- `trajs_per_task=8` is too small for full-suite training. Run `28254656` regressed from 78.5% to 75.0%; that is a small-group issue, not evidence against dynamic sampling itself.
- Full-suite `trajs_per_task=32` without chunking often times out before useful post-RL eval.
- `--n-action-steps 5` is the main new lever. Run `28263586` reached 80% task-5 eval by iter 30 with only 10 eval episodes and much better wall-clock.
- Dynamic sampling should not be treated as a major experimental factor at normal batch sizes. With `trajs_per_task=32` and per-trajectory success `p=0.72`, the probability of drawing an all-success task batch is `0.72^32 = 2.72e-5` (0.0027%), and all-failure is effectively zero. Even across `10 tasks x 45 iterations = 450` task-batches, the chance of seeing any uniform batch is only about 1.2%. It starts to matter near saturation (`p=0.90` gives ~3.4% uniform per task-batch) or when `trajs_per_task` is small (`p=0.72`, `n=8` gives ~7.2% uniform per task-batch).
- Therefore do not submit explicit "dynamic vs non-dynamic" comparison jobs at `trajs_per_task=32` unless success is already near 90% or the rollout batch is intentionally small. Treat dynamic sampling as a low-impact safety net, not as a likely explanation for current performance differences.

## Decision

Drop the old v1-v7 backlog.

Do not submit these now:

- old full-suite v1-v5 jobs that only produced pre-RL best checkpoints
- replay-heavy `success_replay_max_ratio=1.0`
- noise-samples-8 follow-up
- `trajs_per_task=8` full-suite run
- dynamic-vs-non-dynamic full-suite comparisons at `trajs_per_task=32`; the probability of dynamic sampling changing the batch is too small unless tasks are already near saturation
- any `lr=5e-6` run
- any `ppo_epochs>1` run

Submit the jobs below instead.

## Submit Now

Run these in order. If queue capacity allows, submit all three now.

### Job A - Matched Current-Protocol Eval

Purpose: resolve whether SFT, old RL `28188629`, or chunked RL `28263586` is actually best under the current reset behavior.

Submit as `jobs/eval_spatial_current_protocol_l40s.sh`:

```sh
#!/bin/sh

#BSUB -J eval_spatial_current_protocol
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/eval_spatial_current_protocol/%J.out

. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

uv run python scripts/evaluate.py \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --simulator libero \
  --suite spatial \
  --num-episodes 100 \
  --num-envs 8 \
  --max-steps 220 \
  --seed 42 \
  --fixed-noise-seed 42 \
  --wandb-name "eval_sft_spatial_current_seed42" \
  --wandb

uv run python scripts/evaluate.py \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --checkpoint-dir /work3/s234814/vla-robotics/checkpoints/sparse_rl/spatial_task_5_seed42_28188629/best \
  --simulator libero \
  --suite spatial \
  --num-episodes 100 \
  --num-envs 8 \
  --max-steps 220 \
  --seed 42 \
  --fixed-noise-seed 42 \
  --wandb-name "eval_rl_spatial_28188629_best_current_seed42" \
  --wandb

uv run python scripts/evaluate.py \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --checkpoint-dir /work3/s234814/vla-robotics/checkpoints/sparse_rl/spatial_task_5_seed42_28263586/best \
  --simulator libero \
  --suite spatial \
  --num-episodes 100 \
  --num-envs 8 \
  --max-steps 220 \
  --seed 42 \
  --fixed-noise-seed 42 \
  --wandb-name "eval_rl_spatial_28263586_best_current_seed42" \
  --wandb
```

Expected output:

- full-suite SR for all three checkpoints
- per-task table showing whether task 5 remains the main RL win

### Job B - Task-5 Chunked Confirmation

Purpose: verify that `--n-action-steps 5` is genuinely improving task 5 with low-noise eval, not just 10-episode variance.
Dynamic sampling is enabled by default; it should only do work when task-5 groups become uniform.

Submit as `jobs/sparse_fpo_t5_chunk5_confirm_l40s.sh`:

```sh
#!/bin/sh

#BSUB -J sparse_fpo_t5_chunk5_confirm
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/sparse_fpo_t5_chunk5_confirm/%J.out

. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

uv run python scripts/train_srpo.py \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --simulator libero \
  --suite spatial \
  --libero-suite spatial \
  --task-ids 5 \
  --mode sparse_rl \
  --update-method fpo \
  --advantage-mode leave_one_out \
  --seed 42 \
  --lr 3e-06 \
  --max-grad-norm 10.0 \
  --iterations 100 \
  --trajs-per-task 32 \
  --num-rollout-envs 8 \
  --fm-batch-size 64 \
  --ppo-epochs 1 \
  --clip-epsilon 0.05 \
  --clip-epsilon-high 0.08 \
  --num-fm-noise-samples 4 \
  --fpo-negative-adv-scale 1 \
  --kl-coeff 0.01 \
  --sft-kl-coeff 0.005 \
  --adv-eps 1e-8 \
  --adv-skip-threshold 1e-6 \
  --dynamic-sampling \
  --dynamic-sampling-max-retries 2 \
  --eval-every 20 \
  --eval-episodes 100 \
  --max-steps 220 \
  --n-action-steps 5 \
  --gradient-checkpointing \
  --wandb-name "t5-chunk5-lr3e6-sftkl005-eval100" \
  --wandb
```

Success criterion:

- task-5 eval is at least 80% with 100 eval episodes
- no late collapse by iter 80-100

### Job C - Full-Suite Chunked Dynamic Sampling

Purpose: first serious full-suite run under the current code using chunked rollouts plus dynamic sampling.
Dynamic sampling is treated as a default safety net: it is free when groups are mixed and useful when high-success tasks saturate.

Submit as `jobs/sparse_fpo_spatial_all_chunk5_dyn_l40s.sh`:

```sh
#!/bin/sh

#BSUB -J sparse_fpo_spatial_all_chunk5_dyn
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/sparse_fpo_spatial_all_chunk5_dyn/%J.out

. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

uv run python scripts/train_srpo.py \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --simulator libero \
  --suite spatial \
  --libero-suite spatial \
  --task-ids all \
  --mode sparse_rl \
  --update-method fpo \
  --advantage-mode leave_one_out \
  --seed 42 \
  --lr 2e-06 \
  --max-grad-norm 10.0 \
  --iterations 45 \
  --trajs-per-task 32 \
  --num-rollout-envs 8 \
  --fm-batch-size 64 \
  --ppo-epochs 1 \
  --clip-epsilon 0.05 \
  --clip-epsilon-high 0.08 \
  --num-fm-noise-samples 4 \
  --fpo-negative-adv-scale 1 \
  --kl-coeff 0.01 \
  --sft-kl-coeff 0.01 \
  --include-demos-in-update \
  --success-replay-total-size 320 \
  --success-replay-alpha 1.0 \
  --success-replay-ema-decay 0.8 \
  --success-replay-max-ratio 0.5 \
  --adv-eps 1e-8 \
  --adv-skip-threshold 1e-6 \
  --dynamic-sampling \
  --dynamic-sampling-max-retries 2 \
  --eval-every 15 \
  --eval-episodes 20 \
  --max-steps 220 \
  --n-action-steps 5 \
  --gradient-checkpointing \
  --wandb-name "spatial-all-chunk5-dyn-lr2e6-sftkl001-replay320-demos" \
  --wandb
```

Success criterion:

- post-RL eval beats iteration 0
- task 5 improves without large losses on tasks 0, 1, 4, 6, 8
- dynamic sampling retries remain modest until tasks approach saturation
- no regression like the `trajs_per_task=8` run

## Promotion Eval

For every promising training run:

1. Promote with `10 x 50`.
2. Finalize with `10 x 100`.
3. Compare against the matched current-protocol SFT eval from Job A, not the old 80.9% number unless using the exact old eval setup.

Use this command pattern:

```sh
uv run python scripts/evaluate.py \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --checkpoint-dir /work3/s234814/vla-robotics/checkpoints/sparse_rl/<RUN_DIR>/best \
  --simulator libero \
  --suite spatial \
  --num-episodes 100 \
  --num-envs 8 \
  --max-steps 220 \
  --seed 42 \
  --fixed-noise-seed 42 \
  --wandb-name "eval_rl_spatial_<RUN_DIR>_best_current_seed42" \
  --wandb
```

## Recording Rule

After jobs finish:

```sh
uv run python src/vla/utils/fetch_wandb.py <training-project> --type training --with-history
uv run python src/vla/utils/fetch_wandb.py <eval-project> --type eval
```

Future fetched JSONs should include W&B provenance and git/LSF metadata when available.
If `git_commit` is still missing, do not treat the reconstructed commit as ground truth.
