# AOMB lab stack

Small **org-level** multi-service app that emits **real** OpenTelemetry traces/logs
into a collector, with scripted fault injection and capture windows.

## Services

| Service | Port | Role |
|---------|------|------|
| frontend | 8080 | Edge proxy + OTel |
| api | 8081 | Business API (Postgres + Redis) + OTel + fault switch |
| postgres | 5433 | Dependency |
| redis | 6379 | Dependency |
| otel-collector | 4317/4318 | OTLP receiver → `captures/_active/*.jsonl` |

## Quick start

```bash
cd lab
docker compose up -d --build
./scripts/run_capture_session.sh
# → lab/captures/<id>/{provenance.json,traces.jsonl,logs.jsonl,
#                      normal_*.jsonl,incident_*.jsonl}

# Convert to AOMB session parquet (prepare.py shape)
cd ..
uv run python -m corpus.ingest.build_shards \
  --adapter lab_capture \
  --input lab/captures/<id> \
  --num-train-shards 4 \
  --write-val-shard
```

## Capture notes (do not rotate `_active` while collector is running)

The collector file exporter keeps long-lived FDs on `captures/_active/*.jsonl`.
On Linux/macOS, `mv`-ing those files into `_active_prev_*` **does not stop
writes** — spans keep landing in the moved inode while a fresh
`_active/traces.jsonl` stays empty. That produced empty final `traces.jsonl`
with the real data left under `_active_prev_*` (observed 20260911T183259Z).

`run_capture_session.sh` therefore:

1. **Stops** `otel-collector` before any archive/clear of `_active`
2. Snapshots each window to `normal_*.jsonl` / `incident_*.jsonl`, then clears `_active`
3. Merges via `scripts/merge_capture_exports.py` (window files + `_active`, with
   optional recovery from `_active_prev_*` touched during the session)

Pre-session leftovers are moved to `_active_archive_*` (not merged). Do not
hand-rotate `_active` while the collector is up.

## Faults

```bash
./scripts/inject_faults.sh latency      # FAULT_LATENCY_MS (default 800)
./scripts/inject_faults.sh errors       # FAULT_ERROR_RATE (default 0.5)
./scripts/inject_faults.sh both
./scripts/inject_faults.sh kill_redis
./scripts/inject_faults.sh kill_postgres
./scripts/inject_faults.sh restore_deps
./scripts/inject_faults.sh none

# "Rule-proof" faults: every request still 200, latency ~normal
./scripts/inject_faults.sh silent_fallback   # new WARN log line
./scripts/inject_faults.sh skip_cache        # checkout skips Redis (span disappears)
./scripts/inject_faults.sh retry_storm       # each DB ping runs 3×
./scripts/inject_faults.sh db_failover       # checkout logs db=replica instead of db=ok
```

Faults are applied by recreating the `api` container with `FAULT_MODE` in its
environment, so every gunicorn worker sees them. (Before 2026-09-25 the compose
file hardcoded `FAULT_MODE: none`, and faults only reached one of two workers.)

Pool several captures for evaluation:

```bash
python3 scripts/pool_captures.py captures/pooled-<id> captures/<a> captures/<b> ...
```

Window labels (`normal` / `incident`) come from **capture metadata** in
`provenance.json`, not from invented per-event anomaly tags.

## Layout

```
lab/
  docker-compose.yml
  otel/otel-collector-config.yaml
  services/api/
  services/frontend/
  scripts/{inject_faults,loadgen,capture,run_capture_session}.sh
  scripts/merge_capture_exports.py
  captures/          # gitignored runtime output
```
