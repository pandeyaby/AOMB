#!/usr/bin/env bash
# Thin wrapper: one factual Tale measured-card line for public-wins.
# Never invents AUROC / val_bpb. prepare.py sacred.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RED=$'\033[31m'
RST=$'\033[0m'
EXIT_REFUSED_FLAG=1
die_refuse() { echo "${RED}ERROR:${RST} $*" >&2; exit "$EXIT_REFUSED_FLAG"; }
for arg in "$@"; do
  key="${arg%%=*}"
  case "$key" in
    --auroc|--lab-auroc|--accuracy|--ranking|--publish|--claim|--invent-metrics|--invent-auroc|--claim-auroc|--val-bpb|--invent-val-bpb|--readme-hero|--publish-readme|--hero-auroc|--cuda|--gpu)
      die_refuse "Refusing invent / publish / CUDA flag: $key"
      ;;
  esac
done
PYTHON="${PYTHON:-python3}"
exec "$PYTHON" -m eval.public_wins_tale_line "$@"
