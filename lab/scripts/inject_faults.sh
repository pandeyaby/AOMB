#!/usr/bin/env bash
# Induce faults on the running lab API.
# Modes: none | latency | errors | both | kill_redis | kill_postgres | restore_deps
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-latency}"
LATENCY_MS="${LATENCY_MS:-800}"
ERROR_RATE="${ERROR_RATE:-0.5}"

case "$MODE" in
  none|latency|errors|both)
    echo "Setting FAULT_MODE=$MODE on api (recreate)"
    FAULT_MODE="$MODE" FAULT_LATENCY_MS="$LATENCY_MS" FAULT_ERROR_RATE="$ERROR_RATE" \
      docker compose up -d --no-deps --force-recreate api
    # Also hit runtime endpoint if container already had old env partially
    sleep 2
    curl -sS -X POST "http://localhost:8081/admin/fault" \
      -H 'Content-Type: application/json' \
      -d "{\"mode\":\"$MODE\",\"latency_ms\":$LATENCY_MS,\"error_rate\":$ERROR_RATE}" || true
    echo
    ;;
  kill_redis)
    echo "Stopping redis (dependency kill)"
    docker compose stop redis
    ;;
  kill_postgres)
    echo "Stopping postgres (dependency kill)"
    docker compose stop postgres
    ;;
  restore_deps)
    echo "Restoring redis + postgres; clearing faults"
    docker compose start redis postgres || docker compose up -d redis postgres
    FAULT_MODE=none docker compose up -d --no-deps --force-recreate api
    sleep 2
    curl -sS -X POST "http://localhost:8081/admin/fault" \
      -H 'Content-Type: application/json' \
      -d '{"mode":"none"}' || true
    echo
    ;;
  *)
    echo "Usage: $0 {none|latency|errors|both|kill_redis|kill_postgres|restore_deps}"
    exit 1
    ;;
esac
