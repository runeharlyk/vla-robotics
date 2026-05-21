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

export VLA_MIN_WORK3_FREE_GIB="${VLA_MIN_WORK3_FREE_GIB:-10}"

_vla_quota_free_gib() {
  if ! command -v getquota_work3.sh >/dev/null 2>&1; then
    return 1
  fi

  getquota_work3.sh 2>/dev/null | awk -F'|' -v user="${USER:-}" '
    function trim(s) {
      gsub(/^[ \t]+|[ \t]+$/, "", s)
      return s
    }
    function to_gib(raw,   parts, value, unit) {
      raw = trim(raw)
      if (raw == "" || raw == "unlimited") {
        return -1
      }
      split(raw, parts, /[ \t]+/)
      value = parts[1] + 0
      unit = parts[2]
      if (unit == "Byte" || unit == "Bytes") {
        return value / 1024 / 1024 / 1024
      }
      if (unit == "KiB") {
        return value / 1024 / 1024
      }
      if (unit == "MiB") {
        return value / 1024
      }
      if (unit == "GiB") {
        return value
      }
      if (unit == "TiB") {
        return value * 1024
      }
      return -1
    }
    $1 ~ user && NF >= 5 {
      used = to_gib($4)
      hard = to_gib($5)
      if (hard >= 0 && used >= 0) {
        free = hard - used
        if (!seen || free < min_free) {
          min_free = free
          seen = 1
        }
      }
    }
    END {
      if (seen) {
        printf "%.2f\n", min_free
      } else {
        exit 1
      }
    }
  '
}

_vla_df_free_gib() {
  df -Pk "$VLA_WORK3" 2>/dev/null | awk 'NR == 2 { printf "%.2f\n", $4 / 1024 / 1024 }'
}

_vla_check_work3_space() {
  free_gib="$(_vla_quota_free_gib || _vla_df_free_gib || true)"
  if [ -z "$free_gib" ]; then
    echo "Warning: could not determine free space for $VLA_WORK3"
    return 0
  fi

  is_low="$(awk -v free="$free_gib" -v min="$VLA_MIN_WORK3_FREE_GIB" 'BEGIN { print (free < min) ? 1 : 0 }')"
  if [ "$is_low" -eq 1 ]; then
    msg="Warning: only ${free_gib} GiB free for $VLA_WORK3; need at least ${VLA_MIN_WORK3_FREE_GIB} GiB"
    echo "$msg"
  else
    echo "Work3 free-space check OK: ${free_gib} GiB available (minimum ${VLA_MIN_WORK3_FREE_GIB} GiB)"
  fi
}

_vla_check_work3_space

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
