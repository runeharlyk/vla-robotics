#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J smolvla_eval_libero_all_a100
#BSUB -q gpua100
#BSUB -W 16:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/smolvla_eval_libero_all_a100/%J.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

uv run lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.control_mode=relative \
  --eval.batch_size=8 \
  --eval.n_episodes=100 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --env.max_parallel_tasks=1 \
  --seed=42
