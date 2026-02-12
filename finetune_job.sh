#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J smolvla-finetune
#BSUB -q gpul40s
#BSUB -W 04:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234834@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/%J.out
#BSUB -eo logs/%J.err
# -------------------------------------------------

set -e

module load cuda/12.2

nvidia-smi

uv run python src/vla/train_vla.py \
    --env PickCube-v1 \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-5 \
    --model-id lerobot/smolvla_base \
    --seq-len 8 \
    --device cuda \
    --freeze-vision \
    --weight-decay 0.01 \
    --grad-clip 1.0 \
    --warmup-steps 200 \
    --val-split 0.1 \
    --patience 10 \
    --wandb-project vla-smolvla
