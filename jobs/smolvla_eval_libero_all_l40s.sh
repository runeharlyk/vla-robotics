#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J smolvla_eval_libero_all_l40s
#BSUB -q gpul40s
#BSUB -W 04:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/smolvla_eval_libero_all_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

uv run python -m vla evaluate \
    --model smolvla \
    --checkpoint HuggingFaceVLA/smolvla_libero \
    --simulator libero \
    --suite all \
    --num-episodes 20 \
    --device cuda \
    --compile \
    --wandb-project vla-smolvla-libero
