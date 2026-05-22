#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J sft_libero_all_control_l40s
#BSUB -q gpul40s
#BSUB -W 24:00
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/sft_libero_all_control_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

# Job A — CONTROL: continued SFT on all 4 LIBERO suites with NO augmentation.
# Pairs with Job B (jobs/sft_libero_all_augmented_l40s.sh) which adds the
# perturbed datasets, online photometric augmentation, and online instruction
# paraphrasing under otherwise-identical hyperparameters.
#
# Recipe-of-record: configs/sft_libero_all_control.yaml.
#
# All optimizer / scheduler knobs below are the LeRobot SmolVLA defaults that
# trained HuggingFaceVLA/smolvla_libero (lr=1e-4, AdamW betas=(0.9, 0.95),
# wd=1e-10 [hard-coded in scripts/train_sft.py], grad_clip=10, warmup=1000,
# decay=30000, decay_lr=2.5e-6, batch=64, AMP off). Source: SmolVLA paper
# §4.3, lerobot/issues/3287, and the policy config.json shipped with the
# checkpoint. We only have to set the few knobs scripts/train_sft.py does NOT
# already default to (here: --batch-size which defaults to 32).

DATA_DIR="${VLA_WORK3:-.}/data/preprocessed"
for f in spatial.pt object.pt goal.pt long.pt; do
  if [ ! -f "$DATA_DIR/$f" ]; then
    echo "ERROR: $DATA_DIR/$f missing — run jobs/export_libero_to_pt_l40s.sh first." >&2
    exit 1
  fi
done

uv run python scripts/train_sft.py \
  --data "$DATA_DIR/spatial.pt" \
  --data "$DATA_DIR/object.pt" \
  --data "$DATA_DIR/goal.pt" \
  --data "$DATA_DIR/long.pt" \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --lr 1e-4 \
  --warmup-steps 1000 \
  --decay-steps 30000 \
  --decay-lr 2.5e-6 \
  --grad-clip-norm 10.0 \
  --batch-size 64 \
  --micro-batch-size 4 \
  --epochs 25 \
  --eval-every 5 \
  --eval-episodes 100 \
  --max-steps 220 \
  --simulator libero \
  --eval-suite spatial \
  --action-chunk-size 50 \
  --seed 42

echo "Done."
