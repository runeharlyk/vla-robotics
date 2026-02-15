#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J rt1-train
#BSUB -q gpua100
#BSUB -W 04:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[gpu40gb]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234834@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/%J.out
#BSUB -eo logs/%J.err
# -------------------------------------------------

set -e

module load cuda/11.8

source .venv/bin/activate

nvidia-smi

uv run python src/maniskill/train.py rt1 \
    --env PickCube-v1 \
    --epochs 200 \
    --batch-size 32 \
    --lr 3e-5 \
    --seq-len 6 \
    --pretrained \
    --wandb-project vla-rt1