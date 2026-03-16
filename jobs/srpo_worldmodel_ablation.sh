#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J srpo_wm_ablation
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/srpo_wm_ablation/%J.out
# -------------------------------------------------
. jobs/_env.sh

SFT_CKPT="${1:-$VLA_WORK3/checkpoints/sft/best}"
SIMULATOR="${2:-libero}"
SUITE="${3:-spatial}"
TASK_ID="${4:-0}"
ITERS="${5:-100}"
TRAJS="${6:-16}"
ENVS="${7:-8}"
SEEDS="42 123 456"

for WM in dinov2 vjepa2; do
  for SEED in $SEEDS; do
    echo "=== world_model=$WM  seed=$SEED ==="
    uv run python scripts/train_srpo.py \
        --sft-checkpoint "$SFT_CKPT" \
        --simulator "$SIMULATOR" \
        --suite "$SUITE" \
        --task-ids "$TASK_ID" \
        --world-model "$WM" \
        --iterations "$ITERS" \
        --num-rollout-envs "$ENVS" \
        --ppo-epochs 1 \
        --max-steps 280 \
        --fm-batch-size 128 \
        --gradient-checkpointing \
        --seed "$SEED" \
        --wandb
  done
done
