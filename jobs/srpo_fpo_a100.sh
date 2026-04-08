#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J fpo_t5_clean_srpo_a100
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/srpo_fpo_a100/%J.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

uv run python scripts/train_srpo.py \
  --simulator libero \
  --suite spatial \
  --libero-suite spatial \
  --task-ids 5 \
  --mode srpo \
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
  --fpo-negative-adv-scale 1.0 \
  --kl-coeff 0.01 \
  --adv-eps 1e-8 \
  --adv-skip-threshold 1e-6 \
  --eval-every 25 \
  --eval-episodes 100 \
  --max-steps 220 \
  --gradient-checkpointing \
  --wandb-name "v21-t5-srpo-clean" \
  --wandb
