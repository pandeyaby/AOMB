#!/usr/bin/env bash
# Snapshot active collector exports into a tagged capture directory.
# Usage: ./capture.sh <capture_id> <window_label> [fault]
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
# Copy whatever the collector has flushed so far
for f in traces.jsonl logs.jsonl; do
  if [[ -f "$ACTIVE/$f" ]]; then
    # Append into window-scoped files, keep raw copies
    cp "$ACTIVE/$f" "$DEST/${WINDOW}_$f" 2>/dev/null || true
  fi
done

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
