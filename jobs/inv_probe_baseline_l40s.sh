#!/bin/sh

# Drift probe for the v1 baseline checkpoint (arm 1 of the invariance ladder).
# The arms 2-6 + public-reference probes are in results/probes/; this fills the
# baseline row so probe drift can be compared against plain SFT.
#BSUB -J inv_probe_baseline
#BSUB -q gpul40s
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -Ne
#BSUB -env "all, LSB_JOB_REPORT_MAIL=N"
#BSUB -oo logs/inv_probe/%J.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n%s\nY\n" "$LIBERO_PATH" | uv run python -c "import libero.libero; print('Libero configured')"

CKPT_DIR="$VLA_WORK3/checkpoints/sft/spatial_baseline_seed42/last"
echo "=== drift probe: baseline (v1) ckpt=$CKPT_DIR ==="
uv run python scripts/probe_drift.py --checkpoint-dir "$CKPT_DIR" --libero-suite spatial
