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

# SUFFIX distinguishes objective versions.
#   v2 = direct alignment, no predictor (v1's predictor absorbed the mapping).
#   v3 = v2 + THREE code-review fixes: (a) encode_prefix_pooled now pools the
#        CONTEXTUAL prefix (through the VLM transformer) instead of raw input
#        embeddings — v1/v2's language arm never sent gradient into the LLM;
#        (b) templated paraphrase fallback — the variants JSON covered only 5
#        of 10 Spatial tasks, so half the language-arm data was silently clean;
#        (c) augment arm now trains on a fair 50/50 clean/nuisance mix (was
#        100% nuisance).
# Baseline (arm 1) has no invariance/nuisance component -> its v1 checkpoint
# stays valid. Submit v3 as: bsub -J "inv_sweep[2-5]" < this_script
SUFFIX="${SUFFIX:-_v3}"

ARMS="baseline augment vision language both"
ARM=$(echo "$ARMS" | cut -d' ' -f"$LSB_JOBINDEX")
echo "=== inv_sweep element $LSB_JOBINDEX -> arm=$ARM (seed 42, suffix=$SUFFIX) ==="

uv run python scripts/train_sft.py \
    --libero-suite spatial \
    --eval-suite spatial \
    --arm "$ARM" \
    --unfreeze-backbone \
    --inv-target ema \
    --inv-lambda 1.0 \
    --no-inv-predictor \
    --lr 2.5e-5 \
    --epochs 5 \
    --eval-episodes 0 \
    --batch-size 32 \
    --micro-batch-size 4 \
    --seed 42 \
    --save-tag "spatial_${ARM}_seed42${SUFFIX}"
