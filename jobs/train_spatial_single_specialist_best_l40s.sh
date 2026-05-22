#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J train_spatial_single_specialist_best_l40s
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/train_spatial_single_specialist_best_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

TASK_ID="${1:-4}"

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" \
  | uv run --no-sync python -c "import libero.libero; print('Libero configured')"

echo "Kind: train"
echo "Experiment: spatial_single_specialist_best"
echo "Task id: ${TASK_ID}"
echo "Profile: l40s-16"
echo "Git commit: $(git rev-parse HEAD)"
git status --short || true
git diff HEAD -- configs/train_srpo scripts/train_srpo_hydra.py scripts/train_srpo.py scripts/evaluate.py src/vla/rl || true

uv run --no-sync python scripts/train_srpo_hydra.py \
  experiment=spatial_single_specialist_best \
  task_ids="${TASK_ID}" \
  wandb_name="spatial-t${TASK_ID}-specialist-best-fpo-n1-lr1e6"
