#!/usr/bin/env bash
# Public ranking card v1 — one-command reproduce (baselines by default).
# claim_status=not_published. No overnight / API spend. prepare.py untouched.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run python)
else
  RUN=("$PYTHON")
fi

exec "${RUN[@]}" -m eval.run_public_ranking_card "$@"
