#!/bin/sh

# ---------------- LSF directives ----------------
# Smoke test for the latent-invariance objective (arm D = both modalities).
# Validates the full HPC pipeline: LIBERO data load, two-view nuisance
# construction, EMA target encoder, combined SFT+invariance loss, and the
# EGL-based LIBERO simulator eval -- before launching the full A/A'/B/C/D sweep.
#BSUB -J inv_smoke_l40s
#BSUB -q gpul40s
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -Ne
#BSUB -oo logs/inv_smoke_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

uv run python scripts/train_sft.py \
    --libero-suite spatial \
    --eval-suite spatial \
    --num-demos 5 \
    --arm both \
    --inv-target ema \
    --inv-lambda 1.0 \
    --epochs 2 \
    --eval-every 2 \
    --eval-episodes 10 \
    --batch-size 32 \
    --micro-batch-size 4 \
    --no-wandb
