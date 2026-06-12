#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J language_sensitivity_object
#BSUB -q gpul40s
#BSUB -W 48:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234809@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/language_sensitivity_object_l40s/%J.out
# -------------------------------------------------

. jobs/_env.sh

mkdir -p logs/language_sensitivity_object_l40s

uv run python -m language_diagnostics.sensitivity_experiment.language_sensitivity \
  --rollout data/libero/libero_object_tasks5_rollouts50.h5 \
  --variants-json language_diagnostics/variant_prompt_plan_full.json \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --device cuda \
  --seed 0 \
  --output-dir language_diagnostics/sensitivity_experiment/outputs/object_l40s_$LSB_JOBID
