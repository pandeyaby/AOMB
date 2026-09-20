#!/usr/bin/env bash
# Thin wrapper: quarantine / restore ~/.cache/autoresearch data+tokenizer lanes.
# CRISP ↔ Tale without silent clobber. prepare.py sacred. No AUROC / val_bpb / CUDA.
#
# One-liner:
#   ./scripts/cache_lane.sh quarantine --lane tale
#   ./scripts/cache_lane.sh restore --lane crisp
#   ./scripts/cache_lane.sh status
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED=$'\033[31m'
RST=$'\033[0m'
EXIT_REFUSED_FLAG=1

die_refuse() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit "$EXIT_REFUSED_FLAG"
}

# Invent / publish / CUDA — refuse before Python (mirrors Python REFUSED_METRIC_FLAGS).
for arg in "$@"; do
  key="${arg%%=*}"
  case "$key" in
    --auroc|--lab-auroc|--accuracy|--ranking|--publish|--claim|--invent-metrics|--invent-auroc|--claim-auroc|--val-bpb|--invent-val-bpb|--readme-hero|--publish-readme|--hero-auroc|--cuda|--gpu)
      die_refuse "Refusing invent / publish / CUDA flag: $key (cache_lane moves dirs only)"
      ;;
  esac
done

PYTHON="${PYTHON:-python3}"
exec "$PYTHON" -m corpus.ingest.cache_lane "$@"
