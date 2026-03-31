#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J eval_rl_spatial_l40s
#BSUB -q gpul40s
#BSUB -W 12:00
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/eval_rl_spatial_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

uv run python scripts/evaluate.py \
  --checkpoint-dir /work3/s234814/vla-robotics/checkpoints/sparse_rl/spatial_task_2_seed42_28117746/best \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --simulator libero \
  --suite spatial \
  --num-episodes 100 \
  --max-steps 220 \
  --seed 42 \
  --num-envs 8