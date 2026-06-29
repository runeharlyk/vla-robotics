#!/bin/sh

# ---------------- LSF directives ----------------
# LIBERO-Plus (perturbed) eval for the 5 invariance arms (one array element
# each), all perturbation categories. Mirrors jobs/eval_smolvla_libero_plus_l40s.sh
# but points CHECKPOINT at each arm's trained checkpoint. Run after the sweep.
#BSUB -J inv_eval_plus[1-5]
#BSUB -q gpul40s
#BSUB -W 12:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -Ne
#BSUB -env "LSB_JOB_REPORT_MAIL=N"
#BSUB -oo logs/inv_eval_plus/%J_%I.out
# -------------------------------------------------
. jobs/_env.sh

ARMS="baseline augment vision language both"
ARM=$(echo "$ARMS" | cut -d' ' -f"$LSB_JOBINDEX")
CHECKPOINT="$VLA_WORK3/checkpoints/sft/spatial_${ARM}_seed42/last"
SUITE="${SUITE:-spatial}"
CATEGORY="${CATEGORY:-all}"
MAX_TASKS="${MAX_TASKS:-100}"
NUM_EPISODES="${NUM_EPISODES:-5}"
NUM_ENVS="${NUM_ENVS:-8}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-220}"
LIBERO_PLUS_ASSETS="${LIBERO_PLUS_ASSETS:-/work3/s234814/libero-plus/assets}"
WANDB_PROJECT="${WANDB_PROJECT:-vla-libero-plus-eval}"
WANDB_NAME="inv_plus_${ARM}_seed42"

# LIBERO-Plus replaces the base `libero` package; keep it in its own venv.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT_LIBERO_PLUS:-/work3/s234814/.venvs/vla-robotics-libero-plus}"
if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
  echo "Creating LIBERO-Plus venv at $UV_PROJECT_ENVIRONMENT"
  uv venv "$UV_PROJECT_ENVIRONMENT"
fi
uv sync
if [ -f .libero-plus-src/extra_requirements.txt ]; then
  uv pip install -r .libero-plus-src/extra_requirements.txt
fi
uv pip uninstall -y libero || true
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
