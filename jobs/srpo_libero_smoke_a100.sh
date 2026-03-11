#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J srpo_libero_smoke_a100
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[gpu80gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/srpo_libero_smoke_a100/%J.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

uv run python scripts/train_srpo.py \
    --simulator libero \
    --suite spatial \
    --libero-suite spatial \
    --multitask \
    --mode srpo \
    --update-method awr \
    --num-demos 1 \
    --seed 42 \
    --lr 5e-06 \
    --max-grad-norm 10.0 \
    --iterations 200 \
    --trajs-per-task 4 \
    --trajs-per-iter 16 \
    --num-rollout-envs 8 \
    --fm-batch-size 128 \
    --awr-epochs 2 \
    --awr-temperature 5.0 \
    --awr-weight-clip 20.0 \
    --ppo-epochs 1 \
    --clip-epsilon 0.2 \
    --kl-coeff 0.01 \
    --eval-every 20 \
    --eval-episodes 10 \
    --max-steps 280 \
    --world-model vjepa2 \
    --dbscan-eps 60 \
    --dbscan-min-samples 2 \
    --subsample-every 5 \
    --gradient-checkpointing \
    --wandb
