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

uv run lerobot_train \
    --dataset.repo_id=lerobot/libero_10_image \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --steps=20000 \
    --batch_size=64 \
    --lr=1e-4 \
    --device=cuda \
    --wandb.project=vla-smolvla-libero
