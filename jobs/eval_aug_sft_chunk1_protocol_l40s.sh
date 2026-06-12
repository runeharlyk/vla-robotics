#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J eval_aug_sft_chunk1_l40s
#BSUB -q gpul40s
#BSUB -W 06:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/eval_aug_sft_chunk1_l40s/%J.out
# -------------------------------------------------
. jobs/_env.sh

# Eval protocol for the augmented-SFT plan: baseline vs Job A vs Job B at
# chunk-1 (primary) + chunk-2 (sanity), plus chunk-1 cross-suite (object/goal/long).
#
# BEFORE SUBMITTING: edit configs/evaluate/experiment/aug_sft_chunk1_protocol.yaml
# and replace FILL_IN_JOBA_RUN_ID / FILL_IN_JOBB_RUN_ID with the actual run IDs.

PROTOCOL=configs/evaluate/experiment/aug_sft_chunk1_protocol.yaml
if grep -q "FILL_IN_JOB[AB]_RUN_ID" "$PROTOCOL"; then
  echo "ERROR: $PROTOCOL still contains FILL_IN_* placeholders." >&2
  echo "Replace them with the best/ checkpoint dirs printed by Jobs A and B." >&2
  exit 1
fi

uv run python scripts/evaluate_hydra.py experiment=aug_sft_chunk1_protocol

echo "Done. See WandB project vla-libero-eval (entries with prefix eval_aug_sft_*)."
