#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J smolvla-libero
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

mkdir -p "$HF_HOME" "$WANDB_DIR" "$UV_CACHE_DIR" "$UV_PROJECT_ENVIRONMENT"

module load cuda/12.2

nvidia-smi

uv sync

uv run python src/vla/train_smolvla.py \
    --suite long \
    --steps 20000 \
    --batch-size 64 \
    --lr 1e-4 \
    --decay-lr 2.5e-6 \
    --warmup-steps 1000 \
    --decay-steps 30000 \
    --model-id lerobot/smolvla_base \
    --chunk-size 50 \
    --device cuda \
    --weight-decay 1e-10 \
    --grad-clip 10.0 \
    --val-split 0.1 \
    --val-every 500 \
    --amp \
    --num-workers 6 \
    --log-every 50 \
    --wandb-project vla-smolvla-libero
