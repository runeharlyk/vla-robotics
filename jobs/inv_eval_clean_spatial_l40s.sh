#!/bin/sh

# ---------------- LSF directives ----------------
# Clean LIBERO Spatial eval for the 5 invariance arms (one array element each),
# under the calibrated protocol (MuJoCo 3.3.2, seeded init, n_action_steps=1,
# 100 episodes/task). Run after jobs/inv_sweep_spatial_l40s.sh produces
# checkpoints/sft/spatial_<arm>_seed42/last.
#BSUB -J inv_eval_clean[1-5]
#BSUB -q gpul40s
#BSUB -W 12:00
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -Ne
#BSUB -env "LSB_JOB_REPORT_MAIL=N"
#BSUB -oo logs/inv_eval_clean/%J_%I.out
# -------------------------------------------------
. jobs/_env.sh

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n%s\nY\n" "$LIBERO_PATH" | uv run python -c "import libero.libero; print('Libero configured')"

# SUFFIX selects the objective version: "" = v1, "_v3" = current.
# Baseline is only trained as v1, so eval it separately, e.g.:
#   SUFFIX=""    bsub -J "inv_eval_clean[1]"   < this_script
#   SUFFIX="_v3" bsub -J "inv_eval_clean[2-5]" < this_script
SUFFIX="${SUFFIX:-}"
ARMS="baseline augment vision language both"
ARM=$(echo "$ARMS" | cut -d' ' -f"$LSB_JOBINDEX")
CKPT="$VLA_WORK3/checkpoints/sft/spatial_${ARM}_seed42${SUFFIX}/last"
echo "=== clean eval: arm=$ARM suffix=$SUFFIX ckpt=$CKPT ==="

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
  --wandb \
  --wandb-project vla-libero-eval \
  --wandb-name "inv_clean_${ARM}_seed42${SUFFIX}"
