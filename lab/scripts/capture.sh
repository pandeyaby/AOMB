#!/usr/bin/env bash
# Snapshot active collector exports into a tagged capture directory.
# Usage: ./capture.sh <capture_id> <window_label> [fault]
#
# Copies captures/_active/{traces,logs}.jsonl →
#   captures/<id>/{window}_traces.jsonl and {window}_logs.jsonl
#
# Caller must stop the otel-collector before snapshot when also clearing or
# rotating _active afterward — otherwise Linux keeps writing to the moved
# inode and later merges see an empty traces.jsonl (data left in _active_prev_*).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CAPTURES="$ROOT/captures"
ACTIVE="$CAPTURES/_active"
ID="${1:?capture_id}"
WINDOW="${2:?window_label}"
FAULT="${3:-}"
DEST="$CAPTURES/$ID"
mkdir -p "$DEST" "$ACTIVE"

ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

python3 -c "
import importlib.util
from pathlib import Path
p = Path(r'$ROOT/scripts/merge_capture_exports.py')
spec = importlib.util.spec_from_file_location('merge_capture_exports', p)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
written = mod.ensure_window_copies(r'$DEST', r'$ACTIVE', r'$WINDOW')
print('Wrote', ', '.join(written) if written else '(no active export files yet)')
"

meta="$DEST/window_${WINDOW}.json"
cat >"$meta" <<EOF
{
  "label": "$WINDOW",
  "captured_at": "$ts",
  "fault": "$FAULT",
  "notes": "Snapshot of collector file exporter under captures/_active"
}
EOF
echo "Captured window=$WINDOW → $DEST"
