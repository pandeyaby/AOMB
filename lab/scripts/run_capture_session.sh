#!/usr/bin/env bash
# Full lab capture: normal window → fault injection → incident window → provenance.
# Produces lab/captures/<id>/{provenance.json,traces.jsonl,logs.jsonl}
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
# Rotate previous active dumps so this capture starts clean
if compgen -G "$ACTIVE/*" > /dev/null; then
  STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
  mkdir -p "captures/_active_prev_$STAMP"
  mv "$ACTIVE"/* "captures/_active_prev_$STAMP/" 2>/dev/null || true
fi
# Collector may create these as dirs/files — ensure parent exists
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
sleep 5  # flush
NORMAL_END="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
./scripts/capture.sh "$ID" normal ""

echo "=== INCIDENT window (fault=$FAULT_MODE) ==="
INCIDENT_START="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
./scripts/inject_faults.sh "$FAULT_MODE"
./scripts/loadgen.sh "$INCIDENT_REQUESTS"
sleep 5
INCIDENT_END="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
./scripts/capture.sh "$ID" incident "$FAULT_MODE"

# Restore deps / clear faults
./scripts/inject_faults.sh restore_deps || ./scripts/inject_faults.sh none || true

# Merge window snapshots into adapter inputs
python3 - <<'PY' "$DEST" "$ACTIVE"
import json, os, sys, glob
dest, active = sys.argv[1], sys.argv[2]

def collect(patterns):
    lines = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
            if not text:
                continue
            # File exporter may write one JSON value or NDJSON
            if text.startswith("["):
                try:
                    arr = json.loads(text)
                    for item in arr:
                        lines.append(json.dumps(item))
                    continue
                except json.JSONDecodeError:
                    pass
            for line in text.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)
    return lines

trace_lines = collect([
    os.path.join(dest, "normal_traces.jsonl"),
    os.path.join(dest, "incident_traces.jsonl"),
    os.path.join(active, "traces.jsonl"),
    os.path.join(active, "traces.json"),
])
log_lines = collect([
    os.path.join(dest, "normal_logs.jsonl"),
    os.path.join(dest, "incident_logs.jsonl"),
    os.path.join(active, "logs.jsonl"),
    os.path.join(active, "logs.json"),
])
with open(os.path.join(dest, "traces.jsonl"), "w", encoding="utf-8") as f:
    f.write("\n".join(trace_lines) + ("\n" if trace_lines else ""))
with open(os.path.join(dest, "logs.jsonl"), "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines) + ("\n" if log_lines else ""))
print(f"Merged {len(trace_lines)} trace lines, {len(log_lines)} log lines → {dest}")
PY

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
