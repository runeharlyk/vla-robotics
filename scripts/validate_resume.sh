#!/usr/bin/env bash
# Validation harness for the SRPO training-resume implementation.
#
# Runs three scripts back-to-back, all on a single GPU:
#   1. baseline 2-iteration run written to $BASELINE_DIR
#   2. a 2-iteration run written to $RESUME_DIR, killed with SIGTERM
#      after iteration 1 completes (watching metrics.jsonl)
#   3. a 2-iteration resume launched with --resume-from $RESUME_DIR
#
# After all three complete the script compares the iteration-2 entry
# in $BASELINE_DIR/metrics.jsonl with the resumed iteration-2 entry in
# $RESUME_DIR/metrics.jsonl and prints per-key diffs with relative
# tolerance.
#
# Recommended execution: an interactive GPU session on voltash or a100sh,
# for example:
#   linuxsh -q voltash -W 1:00 -gpu "num=1:mode=exclusive_process"
#   bash scripts/validate_resume.sh
#
# Expected overall wall-clock: 10 - 20 minutes on an L40s with the
# default 2-task / 2-iter config below.

set -euo pipefail

SFT_CHECKPOINT=${SFT_CHECKPOINT:-$HOME/smolvla_libero/spatial/best}
OUT_ROOT=${OUT_ROOT:-$(mktemp -d)}
BASELINE_DIR="$OUT_ROOT/baseline"
RESUME_DIR="$OUT_ROOT/resume"
MODE=${MODE:-sparse_rl}
UPDATE_METHOD=${UPDATE_METHOD:-fpo}
SEED=${SEED:-42}
TASK_IDS=${TASK_IDS:-0}
TRAJS_PER_TASK=${TRAJS_PER_TASK:-4}
NUM_ROLLOUT_ENVS=${NUM_ROLLOUT_ENVS:-4}
MAX_STEPS=${MAX_STEPS:-220}

mkdir -p "$OUT_ROOT"
echo "==> validation output: $OUT_ROOT"
echo "==> SFT checkpoint:    $SFT_CHECKPOINT"

common_args=(
  --sft-checkpoint "$SFT_CHECKPOINT"
  --simulator libero
  --suite spatial
  --libero-suite spatial
  --task-ids "$TASK_IDS"
  --mode "$MODE"
  --update-method "$UPDATE_METHOD"
  --seed "$SEED"
  --trajs-per-task "$TRAJS_PER_TASK"
  --num-rollout-envs "$NUM_ROLLOUT_ENVS"
  --max-steps "$MAX_STEPS"
  --iterations 2
  --eval-every 2
  --eval-episodes 10
  --no-wandb
)

run_dir() {
  local cmd=("$@")
  "${cmd[@]}" &
  echo "$!"
}

echo "==> [1/3] baseline 2-iter run"
uv run python scripts/train_srpo.py \
  --checkpoint-out-dir "$BASELINE_DIR" \
  "${common_args[@]}"

echo "==> [2/3] partial run (will be SIGTERM'd after iter 1)"
pid=$(run_dir uv run python scripts/train_srpo.py \
  --checkpoint-out-dir "$RESUME_DIR" \
  "${common_args[@]}")
echo "  pid=$pid"

# Wait for the iter-1 metrics entry (iteration key == 1) to appear.
metrics="$RESUME_DIR/metrics.jsonl"
timeout_s=1800
deadline=$(( SECONDS + timeout_s ))
iter_key="${MODE}/iteration"
while true; do
  if [[ -f "$metrics" ]] && grep -q "\"$iter_key\": 1" "$metrics"; then
    break
  fi
  if (( SECONDS > deadline )); then
    echo "ERROR: iter 1 did not complete within $timeout_s seconds" >&2
    kill -TERM "$pid" 2>/dev/null || true
    exit 1
  fi
  sleep 5
done

echo "==> iter 1 entry seen; sending SIGTERM to $pid"
kill -TERM "$pid" 2>/dev/null || true
wait "$pid" || true

if [[ ! -f "$RESUME_DIR/latest/state.pt" ]]; then
  echo "ERROR: no state.pt under $RESUME_DIR/latest after SIGTERM" >&2
  exit 1
fi

echo "==> [3/3] resume and run iter 2"
uv run python scripts/train_srpo.py \
  --checkpoint-out-dir "$RESUME_DIR" \
  --resume-from "$RESUME_DIR" \
  "${common_args[@]}"

echo "==> comparing iter-2 entries"
uv run python - <<PY
import json
import math
from pathlib import Path

baseline = Path("$BASELINE_DIR/metrics.jsonl")
resumed = Path("$RESUME_DIR/metrics.jsonl")

def last_iter(path, target_iter):
    for line in reversed(path.read_text().splitlines()):
        d = json.loads(line)
        for k, v in d.items():
            if k.endswith("/iteration") and v == target_iter:
                return d
    raise RuntimeError(f"no entry with iteration == {target_iter} in {path}")

a = last_iter(baseline, 2)
b = last_iter(resumed, 2)

keys = [
    k for k in sorted(set(a) | set(b))
    if any(k.endswith(s) for s in ("_loss", "kl_penalty", "step_kl_penalty", "total_successes",
                                    "rollout_successes", "replay_successes", "mean_ratio",
                                    "max_log_ratio", "raw_kl", "eval/success_rate"))
]
tol = 1e-5
diffs = []
for k in keys:
    va, vb = a.get(k), b.get(k)
    if va is None or vb is None:
        continue
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        denom = max(abs(va), abs(vb), 1.0)
        rel = abs(va - vb) / denom
        ok = rel <= tol
    else:
        rel = math.nan
        ok = va == vb
    diffs.append((k, va, vb, rel, ok))

print(f"{'key':50s} {'baseline':>14s} {'resumed':>14s} {'rel_diff':>12s} ok?")
for k, va, vb, rel, ok in diffs:
    print(f"{k:50s} {va!r:>14s} {vb!r:>14s} {rel:>12.2e} {'OK' if ok else 'FAIL'}")

failed = [d for d in diffs if not d[-1]]
if failed:
    print(f"\nFAIL: {len(failed)} key(s) exceeded tolerance {tol}")
    raise SystemExit(1)
print(f"\nPASS: all {len(diffs)} keys match within {tol}")
PY
