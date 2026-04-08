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

CKPT_DIR="${1:-/work3/s234814/vla-robotics/checkpoints/sparse_rl/spatial_task_2_seed42_28123898/best}"

export LIBERO_PATH=/work3/s234814/libero
WANDB_PROJECT="${WANDB_PROJECT:-vla-libero-eval}"
CKPT_NAME="$(basename "$(dirname "$CKPT_DIR")")"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-eval_rl_spatial_l40s_${CKPT_NAME}_${LSB_JOBID}}"
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

uv run python scripts/evaluate.py \
  --checkpoint-dir "$CKPT_DIR"  \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --simulator libero \
  --suite spatial \
  --num-episodes 100 \
  --max-steps 220 \
  --seed 42 \
  --num-envs 8 \
  --fixed-noise-seed 42 \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-name "$WANDB_RUN_NAME"
