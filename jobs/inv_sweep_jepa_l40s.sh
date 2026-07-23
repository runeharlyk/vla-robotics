#!/bin/sh

# ---------------- LSF directives ----------------
# v5 arm: action-conditioned temporal latent prediction (JEPA proper) on the
# fused conditioning features — L_SFT + lambda * d(P(z_t, a_t..t+k), EMA z_{t+k}).
# Motivation: the v3 ladder showed view-invariance Goodharts (drift falls 4x,
# behavior unmoved) while augmentation dominates behaviorally; JEPA constrains
# what the features must CONTAIN (controllable dynamics state) instead of what
# they must ignore. Same training protocol as the other arms otherwise.
# Saves to checkpoints/sft/spatial_jepa_seed42_v3.
#BSUB -J inv_sweep_jepa
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
JEPA_HORIZON="${JEPA_HORIZON:-8}"
echo "=== inv_sweep jepa (horizon=$JEPA_HORIZON, seed 42, suffix=$SUFFIX) ==="

uv run python scripts/train_sft.py \
    --libero-suite spatial \
    --eval-suite spatial \
    --arm jepa \
    --jepa-horizon "$JEPA_HORIZON" \
    --unfreeze-backbone \
    --inv-target ema \
    --inv-lambda 1.0 \
    --lr 2.5e-5 \
    --epochs 5 \
    --eval-episodes 0 \
    --batch-size 32 \
    --micro-batch-size 4 \
    --seed 42 \
    --save-tag "spatial_jepa_seed42${SUFFIX}"
