#!/bin/sh

# ---------------- LSF directives ----------------
# Queue: override at submit time with `bsub -q gpua100 < this_script` (CLI beats #BSUB).
# Training fits gpul40s/gpua100; evals also fit gpua10/gpua40. AVOID gpuv100 (V100 has
# no bf16 support and SmolVLA runs in bfloat16). Requeue pending jobs with `bmod -q`.
# LIBERO-Plus (perturbed) eval for the 5 invariance arms (one array element
# each), all perturbation categories. Mirrors jobs/eval_smolvla_libero_plus_l40s.sh
# but points CHECKPOINT at each arm's trained checkpoint. Run after the sweep.
#BSUB -J inv_eval_plus[1-6]
#BSUB -q gpul40s
#BSUB -W 12:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -Ne
#BSUB -env "all, LSB_JOB_REPORT_MAIL=N"
#BSUB -oo logs/inv_eval_plus/%J_%I.out
# -------------------------------------------------
. jobs/_env.sh

# SUFFIX selects the objective version: "" = v1, "_v3" = current.
# Baseline is only trained as v1, so eval it separately, e.g.:
#   SUFFIX=""    bsub -J "inv_eval_plus[1]"   < this_script
#   SUFFIX="_v3" bsub -J "inv_eval_plus[2-5]" < this_script
SUFFIX="${SUFFIX:-}"
# CKPT_NAME selects which saved weights to eval: last (default) or ema (Polyak average).
CKPT_NAME="${CKPT_NAME:-last}"
NAME_TAG=""
if [ "$CKPT_NAME" != "last" ]; then NAME_TAG="_${CKPT_NAME}"; fi
ARMS="baseline augment vision language both both_aug augment_full jepa"
ARM=$(echo "$ARMS" | cut -d' ' -f"$LSB_JOBINDEX")
CHECKPOINT="$VLA_WORK3/checkpoints/sft/spatial_${ARM}_seed42${SUFFIX}/${CKPT_NAME}"
SUITE="${SUITE:-spatial}"
CATEGORY="${CATEGORY:-all}"
MAX_TASKS="${MAX_TASKS:-100}"
NUM_EPISODES="${NUM_EPISODES:-5}"
NUM_ENVS="${NUM_ENVS:-8}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-220}"
LIBERO_PLUS_ASSETS="${LIBERO_PLUS_ASSETS:-/work3/s234814/libero-plus/assets}"
WANDB_PROJECT="${WANDB_PROJECT:-vla-libero-plus-eval}"
WANDB_NAME="inv_plus_${ARM}_seed42${SUFFIX}${NAME_TAG}"

# LIBERO-Plus replaces the base `libero` package; keep it in its own venv.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT_LIBERO_PLUS:-/work3/s234814/.venvs/vla-robotics-libero-plus}"
if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
  echo "Creating LIBERO-Plus venv at $UV_PROJECT_ENVIRONMENT"
  uv venv "$UV_PROJECT_ENVIRONMENT"
fi
# NEVER let base `libero` (a locked pyproject dep) into this venv: a regular
# site-packages package beats the PYTHONPATH namespace package below, silently
# swapping LIBERO-Plus (2402 spatial tasks) for base LIBERO (10) at import time.
# `uv pip uninstall` does not reliably see the sync-installed copy, so also
# remove any leftover directly on disk.
uv sync --no-install-package libero
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
    echo "Download assets.zip from https://huggingface.co/datasets/Sylvest/LIBERO-plus and unzip first."
    exit 1
  fi
  ln -s "$LIBERO_PLUS_ASSETS" "$PACKAGE_ASSETS"
fi

# Dedicated LIBERO config dir so LIBERO-Plus paths (bddl_files/assets/benchmark_root)
# resolve to the plus source tree, NOT the base-libero paths baked into the shared
# ~/.libero/config.yaml (which the clean/lang/probe jobs rely on). Always start from
# a MISSING config: libero only regenerates it when absent, so a config written by a
# previous bad run (e.g. base libero imported) would otherwise poison every later run.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/work3/s234814/libero-plus/.libero-config}"
mkdir -p "$LIBERO_CONFIG_PATH"
rm -f "$LIBERO_CONFIG_PATH/config.yaml"

export LIBERO_PATH="${LIBERO_PATH:-/work3/s234814/libero-plus}"
mkdir -p "$LIBERO_PATH"
printf "Y\n%s\nY\n" "$LIBERO_PATH" | uv run --no-sync python -c "import libero.libero; print('LIBERO-Plus configured')"

TASK_IDS="$(uv run --no-sync python scripts/select_libero_plus_tasks.py \
  --suite "$SUITE" --category "$CATEGORY" --max-tasks "$MAX_TASKS" --seed "$SEED" --quiet)"

echo "=== LIBERO-Plus eval: arm=$ARM ckpt=$CHECKPOINT category=$CATEGORY tasks=$TASK_IDS ==="

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
