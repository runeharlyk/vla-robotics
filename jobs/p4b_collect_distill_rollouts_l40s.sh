#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J p4b_collect_distill_rollouts_l40s
#BSUB -q gpul40s
#BSUB -W 12:00
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/p4b_collect_distill_rollouts_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

# Phase 4b step 1: collect ~300 successful rollouts per spatial task from
# the Phase-2 / Phase-3 selected teacher checkpoint. Override
# TEACHER_CHECKPOINT_DIR with the actual ckpt path before submission.

TEACHER_CHECKPOINT_DIR="${TEACHER_CHECKPOINT_DIR:-/work3/s234814/vla-robotics/checkpoints/sparse_rl/REPLACE_WITH_PHASE2_WINNER/best}"
OUTPUT_PATH="${OUTPUT_PATH:-/work3/s234814/vla-robotics/data/collected/libero_spatial_distill_300_${LSB_JOBID}.pt}"
SUCCESSES_PER_TASK="${SUCCESSES_PER_TASK:-300}"
MAX_ATTEMPTS_PER_TASK="${MAX_ATTEMPTS_PER_TASK:-2000}"

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

echo "Teacher checkpoint: $TEACHER_CHECKPOINT_DIR"
echo "Output: $OUTPUT_PATH"
echo "Target successes/task: $SUCCESSES_PER_TASK (max attempts: $MAX_ATTEMPTS_PER_TASK)"
echo "Git commit: $(git rev-parse HEAD)"

uv run python scripts/collect_success_dataset.py \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --checkpoint-dir "$TEACHER_CHECKPOINT_DIR" \
  --suite spatial \
  --task-ids all \
  --successes-per-task "$SUCCESSES_PER_TASK" \
  --max-attempts-per-task "$MAX_ATTEMPTS_PER_TASK" \
  --num-envs 8 \
  --n-action-steps 1 \
  --max-steps 220 \
  --seed 42 \
  --output "$OUTPUT_PATH"

# After this completes, submit jobs/p4b_distill_l40s.sh with
#   ROLLOUT_PT="$OUTPUT_PATH"
# in the environment.
