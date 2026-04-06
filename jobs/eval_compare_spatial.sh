#!/bin/sh

# ---------------- LSF directives ----------------
#BSUB -J eval_compare_spatial
#BSUB -q gpul40s
#BSUB -W 12:00
#BSUB -n 12
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -u s234814@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -oo logs/eval_compare_spatial/%J.out
# -------------------------------------------------
. jobs/_env.sh

CKPT_DIR="${1:-/work3/s234814/vla-robotics/checkpoints/sparse_rl/spatial_task_2_seed42_28117746/best}"

export LIBERO_PATH=/work3/s234814/libero
mkdir -p "$LIBERO_PATH"
printf "Y\n/work3/s234814/libero\nY\n" | uv run python -c "import libero.libero; print('Libero configured')"

echo "=============================================="
echo "Checkpoint: $CKPT_DIR"
echo "=============================================="

echo ""
echo "=== 1/2: Custom evaluate.py (same pipeline as training eval) ==="
echo ""

uv run python scripts/evaluate.py \
  --checkpoint-dir "$CKPT_DIR" \
  --checkpoint HuggingFaceVLA/smolvla_libero \
  --simulator libero \
  --suite spatial \
  --num-episodes 100 \
  --max-steps 220 \
  --seed 42 \
  --num-envs 8 \
  --fixed-noise-seed 42

echo ""
echo "=== 2/2: lerobot-eval (independent pipeline) ==="
echo ""

uv run python scripts/convert_checkpoint.py ensure-both -d "$CKPT_DIR"

uv run lerobot-eval \
  --policy.path="$CKPT_DIR" \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.control_mode=relative \
  --eval.batch_size=8 \
  --eval.n_episodes=100 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --env.max_parallel_tasks=1 \
  --seed=42

echo ""
echo "=============================================="
echo "Compare the two success rates above."
echo "If they are close: the pipelines broadly agree."
echo "If they differ materially: there is still a pipeline mismatch to fix."
echo "=============================================="
