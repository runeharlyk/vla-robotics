#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J visual_cor_suc_s1
#BSUB -q gpua40
#BSUB -W 04:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234863@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/visual_cor_suc_s1/%J.out
# -------------------------------------------------

. jobs/_env.sh

export LIBERO_PATH=/work3/s234863/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234863/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

# Run your script
uv run python -m visual_diagnostic.visual_sensitivity \
  --severity 1 \
  --output-dir /work3/s234863/visual_cor_suc_s1 \