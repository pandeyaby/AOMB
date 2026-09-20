#!/usr/bin/env bash
# Tale overnight launch — dry-run first; factual measured-card floor only.
#
# Default: --dry-run (print env, exit 0, no agent_loop / no APIs).
# --run: Darwin+MPS + measured card required; starts agent_loop with:
#   AOMB_CORPUS=tale_of_errors
#   AOMB_SOURCE_ID=tale_capped_200k
#   AOMB_BEST_VAL_FROM_CARD=1
#
# Honesty: never invents AUROC / val_bpb. prepare.py sacred.
#   EXIT_REFUSED_FLAG=1  invent / publish / CUDA flags
#   EXIT_PLATFORM=2      missing card or not Darwin+MPS on --run
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

for arg in "$@"; do
  key="${arg%%=*}"
  case "$key" in
    --auroc|--lab-auroc|--accuracy|--ranking|--publish|--claim|--invent-metrics|--invent-auroc|--claim-auroc|--val-bpb|--invent-val-bpb|--readme-hero|--publish-readme|--hero-auroc|--cuda|--gpu)
      die_refuse "Refusing invent / publish / CUDA flag: $key"
      ;;
  esac
done

# Default to --dry-run when no mode args given.
if [[ $# -eq 0 ]]; then
  set -- --dry-run
fi

PYTHON="${PYTHON:-python3}"
exec "$PYTHON" -m eval.tale_overnight_launch "$@"
