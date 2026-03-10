#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J srpo_mt_maniskill
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/srpo_mt_maniskill/%J.out
# -------------------------------------------------
. jobs/_env.sh

SFT_CKPT="${1:-$VLA_WORK3/checkpoints/sft/best}"
DATA_DIR="${2:-$VLA_WORK3/data/preprocessed}"
ITERS="${3:-100}"
TRAJS_PER_TASK="${4:-4}"
ENVS="${5:-8}"
SEED="${6:-42}"

uv run python scripts/train_srpo.py \
    --sft-checkpoint "$SFT_CKPT" \
    --checkpoint lerobot/smolvla_base \
    --simulator maniskill \
    --multitask \
    --data-dir "$DATA_DIR" \
    --mode srpo \
    --world-model dinov2 \
    --iterations "$ITERS" \
    --trajs-per-task "$TRAJS_PER_TASK" \
    --num-rollout-envs "$ENVS" \
    --fm-batch-size 128 \
    --ppo-epochs 1 \
    --gradient-checkpointing \
    --seed "$SEED" \
    --wandb
