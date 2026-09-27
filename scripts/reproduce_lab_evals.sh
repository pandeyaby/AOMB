#!/usr/bin/env bash
# Reproduce the published labelled lab evals from the captures in lab/published/.
#
#   ./scripts/reproduce_lab_evals.sh            # full protocol: 5 seeds × 120 s per pool (~30 min on Apple MPS)
#   QUICK=1 ./scripts/reproduce_lab_evals.sh    # 1 seed × 30 s per pool — minutes, CPU OK; numbers are noisier
#
# Writes to reports/reproduce/<pool>/ (never overwrites the published reports).
# Compare against docs/lab/in-domain-eval.md, rule-proof-eval.md and value-drift-eval.md.
# The zero-shot eval (docs/lab/ranking-validation.md) also needs the CRISP-500k
# training cache and is not run here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEEDS="0..4"
SECONDS_PER_SEED=120
if [[ "${QUICK:-0}" == "1" ]]; then
  SEEDS="0"
  SECONDS_PER_SEED=30
fi

POOLS=()
for d in lab/published/*/; do
  POOLS+=("${d%/}")
done

for pool in "${POOLS[@]}"; do
  name="$(basename "$pool")"
  out="reports/reproduce/$name"
  echo "=== $name  (seeds $SEEDS, ${SECONDS_PER_SEED}s each) → $out"
  uv run python -m eval.in_domain --capture "$pool" --seeds "$SEEDS" \
    --train-seconds "$SECONDS_PER_SEED" --out-dir "$out"
  # Every lab fault touches checkout; also re-rank the sessions it can reach.
  uv run python -m eval.in_domain --capture "$pool" --out-dir "$out" \
    --subset-marker "op=GET_/api/checkout"
done
echo "Done. Results: reports/reproduce/*/results.md (+ subset-*.json)"
