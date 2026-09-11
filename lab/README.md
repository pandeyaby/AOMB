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
# → lab/captures/<id>/{provenance.json,traces.jsonl,logs.jsonl}

# Convert to AOMB session parquet (prepare.py shape)
cd ..
uv run python -m corpus.ingest.build_shards \
  --adapter lab_capture \
  --input lab/captures/<id> \
  --num-train-shards 4 \
  --write-val-shard
```

## Faults

```bash
./scripts/inject_faults.sh latency      # FAULT_LATENCY_MS (default 800)
./scripts/inject_faults.sh errors       # FAULT_ERROR_RATE (default 0.5)
./scripts/inject_faults.sh both
./scripts/inject_faults.sh kill_redis
./scripts/inject_faults.sh kill_postgres
./scripts/inject_faults.sh restore_deps
./scripts/inject_faults.sh none
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
  captures/          # gitignored runtime output
```
