#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J sparse_awr_l40s
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/sparse_awr_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

uv run python scripts/train_srpo.py \
  --simulator libero \
  --suite spatial \
  --libero-suite spatial \
  --task-ids 2 \
  --mode sparse_rl \
  --update-method awr \
  --advantage-mode leave_one_out \
  --seed 42 \
  --lr 5e-06 \
  --max-grad-norm 10.0 \
  --iterations 50 \
  --trajs-per-task 32 \
  --num-rollout-envs 8 \
  --fm-batch-size 64 \
  --awr-epochs 3 \
  --awr-temperature 0.5 \
  --kl-coeff 0.01 \
  --adv-eps 1e-8 \
  --adv-skip-threshold 1e-6 \
  --eval-every 5 \
  --eval-episodes 50 \
  --max-steps 280 \
  --gradient-checkpointing \
  --wandb-name "v2_awr_fixed" \
  --wandb