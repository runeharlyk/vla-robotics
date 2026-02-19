#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J smolvla-libero90-finetune
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/%J.out
#BSUB -eo logs/%J.err
# -------------------------------------------------

set -e

cd "$LSB_SUBCWD"

exec 2>&1

export HF_HOME=/work3/s234814/.cache/huggingface
export WANDB_DIR=/work3/s234814/.cache/wandb
export WANDB_CACHE_DIR=/work3/s234814/.cache/wandb
export UV_CACHE_DIR=/work3/s234814/.cache/uv
export UV_PROJECT_ENVIRONMENT=/work3/s234814/.venvs/vla-robotics
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$WANDB_DIR" "$UV_CACHE_DIR" "$UV_PROJECT_ENVIRONMENT"

module load cuda/12.2

nvidia-smi

uv sync

uv run lerobot-train \
    --dataset.repo_id=lerobot/libero_90_image \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --steps=50000 \
    --batch_size=64 \
    --lr=1e-4 \
    --device=cuda \
    --wandb.project=vla-smolvla-libero90
