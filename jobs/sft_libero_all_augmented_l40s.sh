#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J sft_libero_all_augmented_l40s
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/sft_libero_all_augmented_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

# Job B — AUGMENTED: continued SFT on all 4 LIBERO suites with:
#   * 4 perturbed datasets (5 variants/demo, sim+photometric+text perturbations)
#   * Online photometric augmentation (brightness/contrast/noise/crop)
#   * Online instruction paraphrasing via libero_all.json
#
# Pairs with Job A (jobs/sft_libero_all_control_l40s.sh) for the
# augmentation-vs-control comparison. All hyperparameters EXCEPT augmentation
# knobs and num_epochs match Job A exactly. num_epochs is reduced from 25 to
# 18 to land at the same ~100k-optimizer-step budget on the larger augmented
# dataset.
#
# Recipe-of-record: configs/sft_libero_all_augmented.yaml.

DATA_DIR="${VLA_WORK3:-.}/data/preprocessed"
for f in spatial.pt object.pt goal.pt long.pt \
         spatial_perturbed.pt object_perturbed.pt goal_perturbed.pt long_perturbed.pt; do
  if [ ! -f "$DATA_DIR/$f" ]; then
    echo "ERROR: $DATA_DIR/$f missing — run jobs/export_libero_to_pt_l40s.sh and jobs/perturb_*_l40s.sh first." >&2
    exit 1
  fi
done

if [ ! -f "data/instruction_variants/libero_all.json" ]; then
  echo "ERROR: data/instruction_variants/libero_all.json missing." >&2
  exit 1
fi

uv run python scripts/train_sft.py \
  --data "$DATA_DIR/spatial.pt" \
  --data "$DATA_DIR/object.pt" \
  --data "$DATA_DIR/goal.pt" \
  --data "$DATA_DIR/long.pt" \
  --data "$DATA_DIR/spatial_perturbed.pt" \
  --data "$DATA_DIR/object_perturbed.pt" \
  --data "$DATA_DIR/goal_perturbed.pt" \
  --data "$DATA_DIR/long_perturbed.pt" \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --lr 1e-4 \
  --warmup-steps 1000 \
  --decay-steps 30000 \
  --decay-lr 2.5e-6 \
  --grad-clip-norm 10.0 \
  --batch-size 64 \
  --micro-batch-size 4 \
  --epochs 18 \
  --eval-every 5 \
  --eval-episodes 100 \
  --max-steps 220 \
  --simulator libero \
  --eval-suite spatial \
  --action-chunk-size 50 \
  --augment.brightness 0.05 \
  --augment.contrast 0.05 \
  --augment.noise-std 0.005 \
  --augment.crop-scale 0.92 \
  --augment.repeats 1 \
  --instruction-variants data/instruction_variants/libero_all.json \
  --seed 42

echo "Done."
