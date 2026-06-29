#!/bin/sh

# ---------------- LSF directives ----------------
# Course-1 sweep: the 5-arm ladder for the latent-invariance objective.
# One array element per arm (baseline / augment / vision / language / both),
# each on its own L40s.  Trains on full LIBERO Spatial, seed 42, with in-loop
# sim eval DISABLED (--eval-episodes 0) -- evaluation is a separate, faster job
# (jobs/inv_eval_*.sh) run on the saved checkpoints afterwards.
#
# NOTE: --epochs / -W are starting values. Check the first element's per-epoch
# time in the log and resize before relying on the full run.
#BSUB -J inv_sweep[1-5]
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -Ne
#BSUB -env "LSB_JOB_REPORT_MAIL=N"
#BSUB -oo logs/inv_sweep/%J_%I.out
# -------------------------------------------------
. jobs/_env.sh

# Fall back to offline wandb if no credentials are present (sync later).
if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then
  export WANDB_MODE=offline
  echo "wandb: no credentials found -> WANDB_MODE=offline"
fi

ARMS="baseline augment vision language both"
ARM=$(echo "$ARMS" | cut -d' ' -f"$LSB_JOBINDEX")
echo "=== inv_sweep element $LSB_JOBINDEX -> arm=$ARM (seed 42) ==="

uv run python scripts/train_sft.py \
    --libero-suite spatial \
    --eval-suite spatial \
    --arm "$ARM" \
    --inv-target ema \
    --inv-lambda 1.0 \
    --epochs 15 \
    --eval-episodes 0 \
    --batch-size 32 \
    --micro-batch-size 4 \
    --seed 42 \
    --save-tag "spatial_${ARM}_seed42"
