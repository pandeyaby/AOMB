#!/usr/bin/env bash
# Public ranking card v1 — one-command reproduce.
# Default: baselines + fixture-only model on frozen eval split.
# No CRISP / prepare data download. No overnight / API spend. prepare.py untouched.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run python)
else
  RUN=("$PYTHON")
fi

# If caller passes no args, run the publishable fixture path (model + ε).
if [[ $# -eq 0 ]]; then
  set -- --with-model --check-eps
fi

exec "${RUN[@]}" -m eval.run_public_ranking_card "$@"
