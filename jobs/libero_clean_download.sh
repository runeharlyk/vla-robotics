#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J libero_clean_download
#BSUB -q gpua10
#BSUB -W 4:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234863@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/libero_clean_download/%J.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234863/libero
mkdir -p "$LIBERO_PATH"

uv run python smolvla_visual_pilot/download_libero_clean.py \
    --datasets libero_10 \
    --output-root /work3/s234863/libero_clean_data \
    --max-episodes 150 \
    