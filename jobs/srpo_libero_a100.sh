#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J srpo_libero
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/srpo_libero/%J.out
# -------------------------------------------------
. jobs/_env.sh

# ── Configurable via positional args ─────────────────────────────────────
SFT_CKPT="${1:-$VLA_WORK3/checkpoints/sft/best}"
SIMULATOR="${2:-libero}"
SUITE="${3:-spatial}"
TASK_ID="${4:-0}"
WORLD_MODEL="${5:-dinov2}"
ITERS="${6:-100}"
TRAJS="${7:-16}"
ENVS="${8:-8}"
SEED="${9:-42}"

echo "=== SRPO Training ==="
echo "  SFT checkpoint : $SFT_CKPT"
echo "  Simulator       : $SIMULATOR"
echo "  Suite           : $SUITE"
echo "  Task ID         : $TASK_ID"
echo "  World model     : $WORLD_MODEL"
echo "  Iterations      : $ITERS"
echo "  Trajs/iter      : $TRAJS"
echo "  Rollout envs    : $ENVS"
echo "  Seed            : $SEED"

uv run python scripts/train_srpo.py \
    --sft-checkpoint "$SFT_CKPT" \
    --simulator "$SIMULATOR" \
    --suite "$SUITE" \
    --task-id "$TASK_ID" \
    --world-model "$WORLD_MODEL" \
    --iterations "$ITERS" \
    --trajs-per-iter "$TRAJS" \
    --num-rollout-envs "$ENVS" \
    --fm-batch-size 32 \
    --seed "$SEED" \
    --wandb
