#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J export_libero_to_pt_l40s
#BSUB -q gpul40s
#BSUB -W 04:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/export_libero_to_pt_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

# Materialise the four LIBERO suites into .pt files compatible with
# scripts/replay_perturbed_dataset.py and FewDemoDataset. Skips suites
# whose .pt already exists. Also runs locally on Windows (the patched
# LIBERO works there) — see scripts/export_libero_to_pt.py --help.

OUT_DIR="${VLA_WORK3:-.}/data/preprocessed"
mkdir -p "$OUT_DIR"

for suite in spatial object goal long; do
  if [ -f "$OUT_DIR/${suite}.pt" ]; then
    echo "Skipping ${suite}.pt — already exists at $OUT_DIR/${suite}.pt"
    continue
  fi
  echo "Exporting LIBERO suite '${suite}' to $OUT_DIR/${suite}.pt"
  uv run python scripts/export_libero_to_pt.py \
    --suite "${suite}" \
    --output-dir "$OUT_DIR" \
    --action-chunk-size 50
done

echo "Done. Listing $OUT_DIR:"
ls -lh "$OUT_DIR"
