#!/usr/bin/env bash
# Generate traffic against the lab frontend/API.
set -euo pipefail
BASE="${BASE_URL:-http://localhost:8080}"
N="${1:-50}"
echo "Loadgen: $N requests against $BASE"
for i in $(seq 1 "$N"); do
  curl -sS -o /dev/null -w "%{http_code}\n" "$BASE/shop/catalog" || true
  curl -sS -o /dev/null -w "%{http_code}\n" "$BASE/shop/checkout" || true
  sleep 0.05
done
echo "Done."
