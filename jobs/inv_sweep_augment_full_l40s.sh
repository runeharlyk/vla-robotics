#!/bin/sh

# ---------------- LSF directives ----------------
# Augmentation-rate ablation: the augment arm (vision+language nuisances, no
# invariance loss) retrained with --nuisance-prob 1.0 (100% augmented views)
# instead of the default fair 50/50 mix. Same protocol as arm 2 of
# jobs/inv_sweep_spatial_l40s.sh otherwise. Saves to
# checkpoints/sft/spatial_augment_full_seed42_v3.
#BSUB -J inv_sweep_augment_full
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -Ne
#BSUB -env "all, LSB_JOB_REPORT_MAIL=N"
#BSUB -oo logs/inv_sweep/%J.out
# -------------------------------------------------
. jobs/_env.sh

if [ -z "${WANDB_API_KEY:-}" ] && [ ! -f "$HOME/.netrc" ]; then
  export WANDB_MODE=offline
  echo "wandb: no credentials found -> WANDB_MODE=offline"
fi

SUFFIX="${SUFFIX:-_v3}"
echo "=== inv_sweep augment_full (nuisance_prob=1.0, seed 42, suffix=$SUFFIX) ==="

uv run python scripts/train_sft.py \
    --libero-suite spatial \
    --eval-suite spatial \
    --arm augment \
    --nuisance-prob 1.0 \
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
    --save-tag "spatial_augment_full_seed42${SUFFIX}"
