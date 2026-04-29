#!/bin/sh

_VLA_ENV_SOURCED=0
(return 0 2>/dev/null) && _VLA_ENV_SOURCED=1

if [ "$_VLA_ENV_SOURCED" -eq 0 ]; then
  set -e
fi

if [ -n "${LSB_SUBCWD:-}" ]; then
  cd "$LSB_SUBCWD"
fi

if [ "$_VLA_ENV_SOURCED" -eq 0 ]; then
  exec 2>&1
fi

export VLA_WORK3=/work3/s234814/vla-robotics

export HF_HOME=/work3/s234814/.cache/huggingface
export WANDB_DIR=/work3/s234814/.cache/wandb
export WANDB_CACHE_DIR=/work3/s234814/.cache/wandb
export UV_CACHE_DIR=/work3/s234814/.cache/uv
export UV_PROJECT_ENVIRONMENT=/work3/s234814/.venvs/vla-robotics

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
export PYTHONUNBUFFERED=1
export SAPIEN_DISABLE_RAY_TRACING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_HOME" "$WANDB_DIR" "$UV_CACHE_DIR" "$UV_PROJECT_ENVIRONMENT" \
       "$VLA_WORK3/data" "$VLA_WORK3/checkpoints" "$VLA_WORK3/outputs" "$VLA_WORK3/models" \
       "logs/${LSB_JOBNAME:-manual}"

if command -v module >/dev/null 2>&1; then
  if [ "$_VLA_ENV_SOURCED" -eq 1 ]; then
    module load cuda/12.2 || echo "Warning: failed to load cuda/12.2"
  else
    module load cuda/12.2
  fi
else
  echo "Warning: environment modules are not available in this shell"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "Warning: nvidia-smi is not available in this shell"
fi

if [ "$_VLA_ENV_SOURCED" -eq 1 ]; then
  uv sync || echo "Warning: uv sync failed; continuing because jobs/_env.sh was sourced interactively"
else
  uv sync
fi
