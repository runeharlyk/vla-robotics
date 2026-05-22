#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J perturb_object_l40s
#BSUB -q gpul40s
#BSUB -W 06:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/perturb_object_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

DATA_DIR="${VLA_WORK3:-.}/data/preprocessed"
INPUT="$DATA_DIR/object.pt"
OUTPUT="$DATA_DIR/object_perturbed.pt"

if [ ! -f "$INPUT" ]; then
  echo "ERROR: $INPUT not found. Run jobs/export_libero_to_pt_l40s.sh first." >&2
  exit 1
fi

uv run python scripts/replay_perturbed_dataset.py \
  --data "$INPUT" \
  --output "$OUTPUT" \
  --suite object \
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
