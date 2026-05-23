#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J visual_cor_suc_s5
#BSUB -q gpua40
#BSUB -W 04:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234809@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/visual_diagnostic_s5/%J.out
# -------------------------------------------------

. jobs/_env.sh

export LIBERO_PATH=/work3/s234809/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

# Run your script
uv run python -m visual_diagnostic.visual_sensitivity \
  --severity 5 \
  --output-dir visual_diagnostic/outputs/s5
