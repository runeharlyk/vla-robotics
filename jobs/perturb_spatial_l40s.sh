#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J perturb_spatial_l40s
#BSUB -q gpul40s
#BSUB -W 06:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/perturb_spatial_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

# Generate spatial_perturbed.pt: 5 perturbed variants per demo, with camera
# jitter, photometric distortion, motion blur, and instruction paraphrasing.
# Drops failed replays so labels remain physically valid (--require-success).
# Uses init_state_id from each episode (added by the export tool) to land in
# the exact recorded LIBERO init state — replay success rate should be high.

DATA_DIR="${VLA_WORK3:-.}/data/preprocessed"
INPUT="$DATA_DIR/spatial.pt"
OUTPUT="$DATA_DIR/spatial_perturbed.pt"

if [ ! -f "$INPUT" ]; then
  echo "ERROR: $INPUT not found. Run jobs/export_libero_to_pt_l40s.sh first." >&2
  exit 1
fi

uv run python scripts/replay_perturbed_dataset.py \
  --data "$INPUT" \
  --output "$OUTPUT" \
  --suite spatial \
  --variants 5 \
  --camera-pos-std 0.015 \
  --camera-fovy-std 0.05 \
  --brightness 0.10 \
  --contrast 0.15 \
  --noise-std 0.008 \
  --motion-blur-max 5 \
  --instruction-variants data/instruction_variants/libero_all.json \
  --require-success

echo "Done. $OUTPUT:"
ls -lh "$OUTPUT"
