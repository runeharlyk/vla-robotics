#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J visual_pilot_eval_s5
#BSUB -q gpua10
#BSUB -W 4:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234863@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/visual_pilot_eval_s5/%J.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234863/libero
mkdir -p "$LIBERO_PATH"

uv run python smolvla_visual_pilot/run_evaluation.py \
    --device cuda \
    --rollout /work3/s234863/libero_clean_data/libero_10 \
    --noise-severity 5 \
    --timestep-batch-size  64 \
    --output-dir /work3/s234863/visual_pilot_eval_s5 \
