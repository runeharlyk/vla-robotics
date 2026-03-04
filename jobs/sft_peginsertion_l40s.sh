#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J sft_peginsertion_l40s
#BSUB -q gpul40s
#BSUB -W 08:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/sft_peginsertion_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

uv run python scripts/train_sft.py \
    --data $VLA_WORK3/data/preprocessed/peginsertionside.pt \
    --env PegInsertionSide-v1 \
    --num-demos 1000 \
    --seed 42 \
    --checkpoint HuggingFaceVLA/smolvla_libero \
    --lr 1e-4 \
    --batch-size 64 \
    --micro-batch-size 32 \
    --epochs 50 \
    --warmup-steps 1000 \
    --decay-steps 30000 \
    --decay-lr 2.5e-6 \
    --grad-clip-norm 10.0 \
    --eval-every 5 \
    --eval-episodes 50 \
    --max-steps 100 \
    --simulator maniskill \
    --eval-suite all \
    --wandb
