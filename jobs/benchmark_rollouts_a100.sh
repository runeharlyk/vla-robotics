#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J benchmark_rollouts_a100
#BSUB -q gpua100
#BSUB -W 08:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/benchmark_rollouts_a100/%J.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

uv run python scripts/benchmark_rollouts.py \
    --simulator all \
    --maniskill-env PickCube-v1 \
    --libero-suite spatial \
    --libero-task-id 0 \
    --max-steps 100 \
    --maniskill-env-counts 1,2,4,8,16,32,64,128 \
    --libero-env-counts 1,2,4,8,16 \
    --warmup-runs 1 \
    --repeats 3 \
    --continue-on-error
