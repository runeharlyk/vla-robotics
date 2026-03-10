#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J srpo_libero_spatial_a100
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/srpo_libero_spatial_a100/%J.out
# -------------------------------------------------
. "$LSB_SUBCWD/jobs/_env.sh"

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

uv run python scripts/train_srpo.py \
    --sft-checkpoint /work3/s234814/vla-robotics/checkpoints/sft/best \
    --simulator libero \
    --suite spatial \
    --task-id 0 \
    --data /work3/s234814/vla-robotics/data/preprocessed/spatial.pt \
    --mode srpo \
    --num-demos 50 \
    --seed 42 \
    --lr 1e-05 \
    --iterations 100 \
    --trajs-per-iter 16 \
    --num-rollout-envs 8 \
    --fm-batch-size 128 \
    --gradient-checkpointing \
    --ppo-epochs 1 \
    --clip-epsilon 0.2 \
    --kl-coeff 0.01 \
    --eval-every 10 \
    --eval-episodes 50 \
    --max-steps 280 \
    --world-model vjepa2 \
    --wandb
