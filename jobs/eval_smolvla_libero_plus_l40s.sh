#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J eval_smolvla_libero_plus_l40s
#BSUB -q gpul40s
#BSUB -W 12:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/eval_smolvla_libero_plus_l40s/%J.out
# -------------------------------------------------

# Baseline eval for HuggingFaceVLA/smolvla_libero on LIBERO-Plus.
# LIBERO-Plus replaces the `libero` Python package while keeping the normal
# suite names; perturbations are selected as explicit task ids from its
# task_classification.json manifest.

. jobs/_env.sh

CHECKPOINT="${CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
SUITE="${SUITE:-spatial}"
CATEGORY="${CATEGORY:-all}"
MAX_TASKS="${MAX_TASKS:-100}"
NUM_EPISODES="${NUM_EPISODES:-1}"
NUM_ENVS="${NUM_ENVS:-8}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-220}"
LIBERO_PLUS_ASSETS="${LIBERO_PLUS_ASSETS:-/work3/s234814/libero-plus/assets}"
WANDB_PROJECT="${WANDB_PROJECT:-vla-libero-plus-eval}"
WANDB_NAME="${WANDB_NAME:-eval_smolvla_plus_${SUITE}_${CATEGORY}_${MAX_TASKS}_${LSB_JOBID:-manual}}"

# Keep the LIBERO-Plus package replacement out of the normal project venv.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT_LIBERO_PLUS:-/work3/s234814/.venvs/vla-robotics-libero-plus}"
if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
  echo "Creating LIBERO-Plus venv at $UV_PROJECT_ENVIRONMENT"
  uv venv "$UV_PROJECT_ENVIRONMENT"
fi

echo "Syncing main project deps into LIBERO-Plus venv..."
# NEVER let base `libero` (a locked pyproject dep) into this venv: a regular
# site-packages package beats the PYTHONPATH namespace package below, silently
# swapping LIBERO-Plus for base LIBERO at import time. `uv pip uninstall` does
# not reliably see the sync-installed copy, so also remove leftovers on disk.
uv sync --no-install-package libero

echo "Installing LIBERO-Plus extras and removing base LIBERO package..."
if [ -f .libero-plus-src/extra_requirements.txt ]; then
  uv pip install -r .libero-plus-src/extra_requirements.txt
fi
rm -rf "$UV_PROJECT_ENVIRONMENT"/lib/python*/site-packages/libero \
       "$UV_PROJECT_ENVIRONMENT"/lib/python*/site-packages/libero-*.dist-info
export PYTHONPATH="$PWD/.libero-plus-src${PYTHONPATH:+:$PYTHONPATH}"

PACKAGE_ASSETS=.libero-plus-src/libero/libero/assets
if [ ! -e "$PACKAGE_ASSETS" ]; then
  if [ ! -d "$LIBERO_PLUS_ASSETS" ]; then
    echo "LIBERO_PLUS_ASSETS does not exist: $LIBERO_PLUS_ASSETS"
    echo "Download assets.zip from https://huggingface.co/datasets/Sylvest/LIBERO-plus and unzip it first."
    exit 1
  fi
  ln -s "$LIBERO_PLUS_ASSETS" "$PACKAGE_ASSETS"
fi

# Dedicated LIBERO config dir (see jobs/inv_eval_plus_spatial_l40s.sh): keeps plus
# paths out of the shared ~/.libero/config.yaml and always regenerates from scratch.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/work3/s234814/libero-plus/.libero-config}"
mkdir -p "$LIBERO_CONFIG_PATH"
rm -f "$LIBERO_CONFIG_PATH/config.yaml"

export LIBERO_PATH="${LIBERO_PATH:-/work3/s234814/libero-plus}"
mkdir -p "$LIBERO_PATH"
printf "Y\n%s\nY\n" "$LIBERO_PATH" | uv run --no-sync python -c "import libero.libero; print('LIBERO-Plus configured')"

TASK_IDS="$(uv run --no-sync python scripts/select_libero_plus_tasks.py \
  --suite "$SUITE" \
  --category "$CATEGORY" \
  --max-tasks "$MAX_TASKS" \
  --seed "$SEED" \
  --quiet)"

echo "Checkpoint        : $CHECKPOINT"
echo "Suite             : $SUITE"
echo "Category          : $CATEGORY"
echo "Selected task ids : $TASK_IDS"
echo "Episodes per task : $NUM_EPISODES"
echo "Num envs          : $NUM_ENVS"
echo "Git commit        : $(git rev-parse HEAD)"

uv run --no-sync python scripts/evaluate.py \
  --checkpoint "$CHECKPOINT" \
  --simulator libero \
  --suite "$SUITE" \
  --task-ids "$TASK_IDS" \
  --num-episodes "$NUM_EPISODES" \
  --max-steps "$MAX_STEPS" \
  --seed "$SEED" \
  --num-envs "$NUM_ENVS" \
  --n-action-steps 1 \
  --fixed-noise-seed "$SEED" \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-name "$WANDB_NAME"
