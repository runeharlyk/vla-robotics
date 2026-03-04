#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J smolvla_eval_maniskill_l40s
#BSUB -q gpul40s
#BSUB -W 04:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/smolvla_eval_maniskill_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

uv run python -m vla evaluate \
    --model smolvla \
    --checkpoint models/smolvla_train_libero_all_a100.pt \
    --simulator maniskill \
    --env-id PegInsertionSide-v1 \
    --num-episodes 20 \
    --device cuda \
    --compile \
    --wandb-project vla-smolvla-maniskill
