#!/usr/bin/env bash
# Full lab capture: normal window → fault injection → incident window → provenance.
# Produces lab/captures/<id>/{provenance.json,traces.jsonl,logs.jsonl}
# plus window-scoped normal_*.jsonl / incident_*.jsonl.
#
# IMPORTANT: never rotate/clear captures/_active while otel-collector still has
# the export files open. On Linux, `mv` keeps the FD writing into _active_prev_*,
# and the final merge used to write an empty traces.jsonl. We always stop the
# collector before archive/clear, then start it again for the next window.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ID="${CAPTURE_ID:-$(date -u +"%Y%m%dT%H%M%SZ")}"
DEST="captures/$ID"
ACTIVE="captures/_active"
FAULT_MODE="${FAULT_MODE:-latency}"
NORMAL_REQUESTS="${NORMAL_REQUESTS:-40}"
INCIDENT_REQUESTS="${INCIDENT_REQUESTS:-40}"

mkdir -p "$DEST" "$ACTIVE"

quiesce_collector() {
  # Close exporter FDs before any mv/rm of _active contents.
  docker compose stop otel-collector >/dev/null 2>&1 || true
  # Brief pause so the process fully exits and flushes.
  sleep 1
}

start_collector() {
  docker compose up -d otel-collector >/dev/null
  sleep 2
}

archive_active() {
  # Caller must have already quiesced the collector.
  # Intentional pre-session archive uses _active_archive_* (NOT _active_prev_*).
  # Merge recovery only scans _active_prev_*, so stale prior-session dumps
  # never pollute the new capture's traces.jsonl.
  if compgen -G "$ACTIVE/*" > /dev/null; then
    STAMP="$(date -u +"%Y%m%dT%H%M%SZ")-$$"
    PREV="captures/_active_archive_${STAMP}"
    mkdir -p "$PREV"
    mv "$ACTIVE"/* "$PREV/" 2>/dev/null || true
    echo "Archived previous _active → $PREV"
  fi
}

clear_active() {
  # After a window snapshot, drop active files so the next window is disjoint.
  # Data already lives under $DEST/<window>_*.jsonl.
  if compgen -G "$ACTIVE/*" > /dev/null; then
    rm -f "$ACTIVE"/* 2>/dev/null || true
  fi
}

# --- boot: stop collector → archive stale _active → bring stack up ---
quiesce_collector
archive_active
mkdir -p "$ACTIVE"
# Session clock starts after archive so --prev-after recovery cannot pull
# intentional _active_archive_* / leftover dirs from before this run.
# (Recovery targets _active_prev_* only — see merge_capture_exports.py.)
SESSION_START_ISO="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
docker compose up -d --build
echo "Waiting for health..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8081/health >/dev/null && curl -sf http://localhost:8080/health >/dev/null; then
    break
  fi
  sleep 2
done

NORMAL_START="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "=== NORMAL window starting $NORMAL_START ==="
./scripts/inject_faults.sh none || true
./scripts/loadgen.sh "$NORMAL_REQUESTS"
sleep 5  # let batch processor flush to file exporter
NORMAL_END="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
quiesce_collector
./scripts/capture.sh "$ID" normal ""
clear_active
start_collector

echo "=== INCIDENT window (fault=$FAULT_MODE) ==="
INCIDENT_START="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
./scripts/inject_faults.sh "$FAULT_MODE"
./scripts/loadgen.sh "$INCIDENT_REQUESTS"
sleep 5
INCIDENT_END="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
quiesce_collector
./scripts/capture.sh "$ID" incident "$FAULT_MODE"

# Restore deps / clear faults (apps still up)
./scripts/inject_faults.sh restore_deps || ./scripts/inject_faults.sh none || true

# Merge window snapshots (+ leftover _active). Also recover any _active_prev_*
# touched during this session (rotate-while-open safety net).
python3 "$ROOT/scripts/merge_capture_exports.py" "$DEST" "$ACTIVE" \
  --captures-root "$ROOT/captures" \
  --prev-after "$SESSION_START_ISO"

# Restart collector so the stack is usable after the session
start_collector || true

cat >"$DEST/provenance.json" <<EOF
{
  "corpus_version": "v1",
  "source_id": "lab-aomb-stack",
  "source_kind": "lab_capture",
  "license": "Apache-2.0",
  "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
  "citation": "AOMB lab stack (lab/docker-compose.yml) — org-level local capture",
  "capture_id": "$ID",
  "captured_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "capture_tool": "lab/scripts/run_capture_session.sh",
  "signals": ["traces", "logs"],
  "windows": [
    {
      "label": "normal",
      "start": "$NORMAL_START",
      "end": "$NORMAL_END",
      "fault": "",
      "notes": "Steady load, FAULT_MODE=none"
    },
    {
      "label": "incident",
      "start": "$INCIDENT_START",
      "end": "$INCIDENT_END",
      "fault": "$FAULT_MODE",
      "notes": "Fault injection active via inject_faults.sh"
    }
  ]
}
EOF

echo ""
echo "Capture complete: $DEST"
echo "Build shards with:"
echo "  uv run python -m corpus.ingest.build_shards --adapter lab_capture --input $DEST --num-train-shards 4 --write-val-shard"
