#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J smolvla-libero-finetune
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
# -------------------------------------------------

set -e

cd "$LSB_SUBCWD"

exec 2>&1

export HF_HOME=/work3/s234814/.cache/huggingface
export WANDB_DIR=/work3/s234814/.cache/wandb
export WANDB_CACHE_DIR=/work3/s234814/.cache/wandb
export UV_CACHE_DIR=/work3/s234814/.cache/uv
export UV_PROJECT_ENVIRONMENT=/work3/s234814/.venvs/vla-robotics
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1

mkdir -p "$HF_HOME" "$WANDB_DIR" "$UV_CACHE_DIR" "$UV_PROJECT_ENVIRONMENT" logs

module load cuda/12.2

nvidia-smi

uv sync --extra sim

uv run lerobot-train \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.load_vlm_weights=true \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.optimizer_lr=1e-4 \
    --env.type=libero \
    --env.task=libero_10 \
    --steps=50000 \
    --batch_size=64 \
    --eval_freq=5000 \
    --eval.n_episodes=10 \
    --eval.batch_size=1 \
    --output_dir=outputs/smolvla_libero \
    --save_checkpoint=true \
    --save_freq=5000 \
    --wandb.enable=true \
    --wandb.project=vla-smolvla-libero
