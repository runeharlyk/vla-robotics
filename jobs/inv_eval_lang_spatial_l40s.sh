#!/bin/sh

# ---------------- LSF directives ----------------
# Queue: override at submit time with `bsub -q gpua100 < this_script` (CLI beats #BSUB).
# Training fits gpul40s/gpua100; evals also fit gpua10/gpua40. AVOID gpuv100 (V100 has
# no bf16 support and SmolVLA runs in bfloat16). Requeue pending jobs with `bmod -q`.
# Language-perturbation LIBERO Spatial eval for the invariance arms: identical
# calibrated protocol to jobs/inv_eval_clean_spatial_l40s.sh (MuJoCo 3.3.2,
# seeded init, n_action_steps=1, 100 episodes/task), but every task instruction
# is replaced by a HELD-OUT paraphrase type (sentence_structure / verbosity —
# training only ever sees politeness / verb_paraphrase). The paraphrase drop vs
# the arm's own clean eval is the language-axis robustness gap.
# Array layout: elements 1-6 = arms x sentence_structure, 7-12 = arms x verbosity.
#BSUB -J inv_eval_lang[1-12]
#BSUB -q gpul40s
#BSUB -W 12:00
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -Ne
#BSUB -env "all, LSB_JOB_REPORT_MAIL=N"
#BSUB -oo logs/inv_eval_lang/%J_%I.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n%s\nY\n" "$LIBERO_PATH" | uv run python -c "import libero.libero; print('Libero configured')"

# SUFFIX selects the objective version: "" = v1, "_v3" = current.
# Baseline is only trained as v1, so eval it separately, e.g.:
#   SUFFIX=""    bsub -J "inv_eval_lang[1,7]"        < this_script
#   SUFFIX="_v3" bsub -J "inv_eval_lang[2-6,8-12]"   < this_script
SUFFIX="${SUFFIX:-}"
# CKPT_NAME selects which saved weights to eval: last (default) or ema (Polyak average).
CKPT_NAME="${CKPT_NAME:-last}"
NAME_TAG=""
if [ "$CKPT_NAME" != "last" ]; then NAME_TAG="_${CKPT_NAME}"; fi

ARMS="baseline augment vision language both both_aug"
ARM_IDX=$(( (LSB_JOBINDEX - 1) % 6 + 1 ))
ARM=$(echo "$ARMS" | cut -d' ' -f"$ARM_IDX")
if [ "$LSB_JOBINDEX" -le 6 ]; then VARIANT=sentence_structure; else VARIANT=verbosity; fi
OVERRIDES="language_diagnostics/heldout_overrides/spatial_${VARIANT}.json"
CKPT="$VLA_WORK3/checkpoints/sft/spatial_${ARM}_seed42${SUFFIX}/${CKPT_NAME}"
echo "=== lang eval: arm=$ARM variant=$VARIANT suffix=$SUFFIX ckpt=$CKPT ==="

uv run python scripts/evaluate.py \
  --checkpoint "$CKPT" \
  --simulator libero \
  --suite spatial \
  --num-episodes 100 \
  --max-steps 220 \
  --seed 42 \
  --num-envs 8 \
  --n-action-steps 1 \
  --fixed-noise-seed 42 \
  --instruction-overrides "$OVERRIDES" \
  --variant-type "$VARIANT" \
  --wandb \
  --wandb-project vla-libero-eval \
  --wandb-name "inv_lang_${VARIANT}_${ARM}_seed42${SUFFIX}${NAME_TAG}"
