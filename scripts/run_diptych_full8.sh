#!/usr/bin/env bash
# Full-8 DIPTYCH adapter gate (deterministic fixtures; no MPS train).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python3 -m eval.diptych.run_full8 "$@"
