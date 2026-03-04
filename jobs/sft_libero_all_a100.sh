#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J sft_libero_all_a100
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/sft_libero_all_a100/%J.out
# -------------------------------------------------
. "$LSB_SUBCWD/jobs/_env.sh"

uv run lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=lerobot/libero \
    --output_dir=outputs/train/smolvla_libero_all \
    --job_name=sft_libero_all_a100 \
    --batch_size=64 \
    --steps=100000 \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --wandb.enable=true
