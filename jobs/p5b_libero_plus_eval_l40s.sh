#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J p5b_libero_plus_eval_l40s
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/p5b_libero_plus_eval_l40s/%J.out
# -------------------------------------------------

# Phase 5b: Evaluate one anchor checkpoint on the LIBERO-Plus benchmark
# (Sylvest et al. 2025, https://arxiv.org/abs/2510.13626 -- in-depth
# robustness analysis for VLAs covering camera / robot / language /
# light / background / noise / layout perturbations applied at the
# init-state level within the four base suites).
#
# LIBERO-Plus installs by REPLACING the libero Python package
# (`pip install -e .libero-plus-src/`). To avoid contaminating the main
# vla-robotics venv used for everything else, we create a separate
# UV project environment for this job and install libero-plus there.
#
# Required overrides before submission:
#   - ANCHOR_CKPT_DIR   : the Phase-2/3/4 anchor ckpt being audited
#   - LIBERO_PLUS_ASSETS: path to the unzipped assets/ folder from
#                         https://huggingface.co/datasets/Sylvest/LIBERO-plus
#   - PERTURBATION_TYPE : one of the seven categories (camera, robot,
#                         language, light, background, noise, layout)
#                         or 'total' to run everything.

. jobs/_env.sh

ANCHOR_CKPT_DIR="${ANCHOR_CKPT_DIR:-REPLACE_WITH_ANCHOR_CKPT_DIR}"
LIBERO_PLUS_ASSETS="${LIBERO_PLUS_ASSETS:-/work3/s234814/libero-plus/assets}"
PERTURBATION_TYPE="${PERTURBATION_TYPE:-total}"
NUM_EPISODES="${NUM_EPISODES:-100}"
WANDB_NAME="${WANDB_NAME:-eval_p5b_libero_plus_${PERTURBATION_TYPE}_${LSB_JOBID}}"

# Use a dedicated venv so we don't pollute the main vla-robotics env.
export UV_PROJECT_ENVIRONMENT=/work3/s234814/.venvs/vla-robotics-libero-plus

if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
  echo "Creating LIBERO-Plus venv at $UV_PROJECT_ENVIRONMENT"
  uv venv "$UV_PROJECT_ENVIRONMENT"
fi

echo "Syncing main project deps into LIBERO-Plus venv..."
uv sync

echo "Installing LIBERO-Plus on top (replaces libero package)..."
uv pip install -e .libero-plus-src/

# LIBERO-Plus reads its assets from inside the package dir; set the
# environment variable so it picks the right perturbation manifest.
export LIBERO_PLUS_ASSETS_PATH="$LIBERO_PLUS_ASSETS"
export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero-Plus configured')"

echo "Anchor checkpoint  : $ANCHOR_CKPT_DIR"
echo "Perturbation type  : $PERTURBATION_TYPE"
echo "Episodes per task  : $NUM_EPISODES"
echo "Git commit         : $(git rev-parse HEAD)"

# LIBERO-Plus exposes the same four suite names (spatial/object/goal/
# long); the perturbation is selected via the PERTURBATION_TYPE env
# variable, which our env wrapper inspects.
for SUITE in spatial object goal long; do
  echo "=== LIBERO-Plus $PERTURBATION_TYPE :: $SUITE ==="
  uv run python scripts/evaluate.py \
    --checkpoint-dir "$ANCHOR_CKPT_DIR" \
    --checkpoint HuggingFaceVLA/smolvla_libero \
    --simulator libero \
    --suite "$SUITE" \
    --num-episodes "$NUM_EPISODES" \
    --max-steps 220 \
    --seed 42 \
    --num-envs 8 \
    --n-action-steps 1 \
    --fixed-noise-seed 42 \
    --wandb \
    --wandb-project vla-libero-plus-eval \
    --wandb-name "${WANDB_NAME}_${SUITE}"
done
