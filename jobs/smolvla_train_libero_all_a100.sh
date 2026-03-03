#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J smolvla_train_libero_all_a100
#BSUB -q gpua100
#BSUB -W 24:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/smolvla_train_libero_all_a100/%J.out
# -------------------------------------------------
. "$LSB_SUBCWD/jobs/_env.sh"

uv run lerobot-train \
    --dataset.repo_id=HuggingFaceVLA/libero \
    --policy.type=smolvla \
    --policy.pretrained_path=lerobot/smolvla_base \
    --policy.load_vlm_weights=true \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --policy.optimizer_lr=0.0001 \
    --env.type=libero \
    --env.task=libero_10 \
    --steps=20000 \
    --batch_size=64 \
    --eval_freq=5000 \
    --eval.n_episodes=10 \
    --eval.batch_size=1 \
    --output_dir=outputs/smolvla_train_libero_all_a100 \
    --save_checkpoint=true \
    --save_freq=5000 \
    --wandb.enable=true \
    --wandb.project=vla-smolvla-libero
