#!/usr/bin/env bash
# Thin wrapper: measured Tale val_bpb JSON card (factual only).
# prepare.py sacred. No invented AUROC / val_bpb. CUDA gate skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}$ROOT"
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  echo "ERROR: python3/python not found" >&2
  exit 127
fi
exec "$PY" -m eval.tale_measured_report "$@"
