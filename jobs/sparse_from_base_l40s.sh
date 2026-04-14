#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J sparse_fpo_l40s
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/sparse_fpo_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

uv run python scripts/train_srpo.py \
  --simulator libero \
  --checkpoint lerobot/smolvla_base\
  --suite spatial \
  --libero-suite spatial \
  --task-ids 1 \
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
  --adv-eps 1e-8 \
  --adv-skip-threshold 1e-6 \
  --eval-every 25 \
  --eval-episodes 50 \
  --max-steps 220 \
  --include-demos-in-update \
  --success-replay-buffer-size 16 \
  --gradient-checkpointing \
  --wandb-name "v17-t-5" \
  --wandb