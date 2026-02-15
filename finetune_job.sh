#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J smolvla-finetune
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/%J.out
#BSUB -eo logs/%J.err
# -------------------------------------------------

set -e

export HF_HOME=/work3/s234814/.cache/huggingface
export WANDB_DIR=/work3/s234814/.cache/wandb
export WANDB_CACHE_DIR=/work3/s234814/.cache/wandb
export UV_CACHE_DIR=/work3/s234814/.cache/uv
export UV_PROJECT_ENVIRONMENT=/work3/s234814/.venvs/vla-robotics
export DATA_DIR=/work3/s234814/data

mkdir -p "$HF_HOME" "$WANDB_DIR" "$UV_CACHE_DIR" "$UV_PROJECT_ENVIRONMENT"

module load cuda/12.2

nvidia-smi

uv sync

uv run python src/vla/train_vla.py \
    --env PickCube-v1 \
    --epochs 50 \
    --batch-size 64 \
    --lr 2e-5 \
    --model-id lerobot/smolvla_base \
    --seq-len 8 \
    --device cuda \
    --no-freeze-vision \
    --weight-decay 0.01 \
    --grad-clip 1.0 \
    --warmup-steps 200 \
    --val-split 0.1 \
    --patience 10 \
    --amp \
    --compile \
    --num-workers 6 \
    --prefetch 4 \
    --wandb-project vla-smolvla
