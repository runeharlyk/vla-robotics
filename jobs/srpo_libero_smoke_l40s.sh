#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J srpo_libero_smoke_l40s
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/srpo_libero_smoke_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

uv run python scripts/train_srpo.py \
    --simulator libero \
    --suite object \
    --task-id 5 \
    --libero-suite object \
    --mode srpo \
    --num-demos 50 \
    --seed 42 \
    --lr 5e-06 \
    --iterations 200 \
    --trajs-per-iter 8 \
    --num-rollout-envs 8 \
    --fm-batch-size 64 \
    --ppo-epochs 1 \
    --clip-epsilon 0.2 \
    --kl-coeff 0.01 \
    --eval-every 20 \
    --eval-episodes 10 \
    --max-steps 280 \
    --world-model vjepa2 \
    --dbscan-eps 60 \
    --gradient-checkpointing \
    --wandb
