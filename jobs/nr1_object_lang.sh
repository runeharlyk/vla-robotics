#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J object_lang_eval_nr1
#BSUB -q gpua40
#BSUB -W 24:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234809@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/libero_prompt_eval_nr1/%J.out
# -------------------------------------------------

. jobs/_env.sh

export LIBERO_PATH=/work3/s234809/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

# Run your script
uv run python language_diagnostics/libero_prompt_variant_run.py \
  --prompt-plan-path language_diagnostics/variant_prompt_plan_object_nr1.json \
  --episodes 50 \
  --device cuda \
  --progress-every 10 \
  --output-dir language_diagnostics/outputs/objects/nr1/$LSB_JOBID
