#!/bin/sh

# ---------------- LSF directives ----------------
# Queue: override at submit time with `bsub -q gpua100 < this_script` (CLI beats #BSUB).
# Training fits gpul40s/gpua100; evals also fit gpua10/gpua40. AVOID gpuv100 (V100 has
# no bf16 support and SmolVLA runs in bfloat16). Requeue pending jobs with `bmod -q`.
# WiSE-FT interpolation sweep (Wortsman et al. 2021): evaluate
#   theta(alpha) = (1-alpha)*public_SFT + alpha*fine-tuned_arm
# on clean LIBERO Spatial under the calibrated protocol. Targets the ~6pp
# forgetting tax: endpoints are known (alpha=0 -> public 80.4%, alpha=1 ->
# the arm's own eval), elements sweep alpha in {0.25, 0.5, 0.75}.
# Select the arm/version via env:  ARM=both SUFFIX=_v3 bsub < this_script
#BSUB -J inv_wiseft[1-3]
#BSUB -q gpul40s
#BSUB -W 12:00
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -Ne
#BSUB -env "all, LSB_JOB_REPORT_MAIL=N"
#BSUB -oo logs/inv_wiseft/%J_%I.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n%s\nY\n" "$LIBERO_PATH" | uv run python -c "import libero.libero; print('Libero configured')"

ARM="${ARM:-both}"
SUFFIX="${SUFFIX:-_v3}"
CKPT_NAME="${CKPT_NAME:-last}"
ALPHAS="0.25 0.5 0.75"
ALPHA=$(echo "$ALPHAS" | cut -d' ' -f"$LSB_JOBINDEX")
CKPT_DIR="$VLA_WORK3/checkpoints/sft/spatial_${ARM}_seed42${SUFFIX}/${CKPT_NAME}"
echo "=== WiSE-FT eval: arm=$ARM suffix=$SUFFIX alpha=$ALPHA ckpt_dir=$CKPT_DIR ==="

uv run python scripts/evaluate.py \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --checkpoint-dir "$CKPT_DIR" \
  --wise-ft-alpha "$ALPHA" \
  --simulator libero \
  --suite spatial \
  --num-episodes 100 \
  --max-steps 220 \
  --seed 42 \
  --num-envs 8 \
  --n-action-steps 1 \
  --fixed-noise-seed 42 \
  --wandb \
  --wandb-project vla-libero-eval \
  --wandb-name "inv_wiseft_${ARM}${SUFFIX}_a${ALPHA}"
