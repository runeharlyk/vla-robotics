#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J p4b_distill_l40s
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/p4b_distill_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

# Phase 4b step 2: distill the SFT-init (or WiSE-FT-merged) student from
# the collected RL-teacher rollouts produced by p4b_collect_distill_rollouts_l40s.sh.

ROLLOUT_PT="${ROLLOUT_PT:-/work3/s234814/vla-robotics/data/collected/libero_spatial_distill_300_REPLACE_WITH_COLLECTION_JOBID.pt}"
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-HuggingFaceVLA/smolvla_libero}"
SAVE_DIR="${SAVE_DIR:-/work3/s234814/vla-robotics/checkpoints/distill/libero_spatial_seed42_${LSB_JOBID}}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LR="${LR:-2e-5}"
WANDB_NAME="${WANDB_NAME:-distill_libero_spatial_${LSB_JOBID}}"

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

echo "Rollout dataset: $ROLLOUT_PT"
echo "Student init   : $STUDENT_CHECKPOINT"
echo "Save dir       : $SAVE_DIR"
echo "LR / epochs    : $LR / $NUM_EPOCHS"
echo "Git commit     : $(git rev-parse HEAD)"

uv run python scripts/distill_from_rollouts.py \
  --rollouts "$ROLLOUT_PT" \
  --student-checkpoint "$STUDENT_CHECKPOINT" \
  --save-dir "$SAVE_DIR" \
  --lr "$LR" \
  --epochs "$NUM_EPOCHS" \
  --batch-size 32 \
  --micro-batch-size 4 \
  --eval-every 2 \
  --eval-episodes 20 \
  --eval-suite spatial \
  --simulator libero \
  --seed 42 \
  --wandb \
  --wandb-name "$WANDB_NAME"
