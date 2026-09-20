#!/usr/bin/env bash
# Public ranking card v1 — honest fixture runner (one-command reproduce).
# Scores committed local fixtures (length/events + optional session BPB).
# Default: baselines + fixture-only model on frozen eval split + ε check.
# No CRISP / prepare data download. No overnight / API spend. prepare.py untouched.
# Never invents AUROC heroes / README publish / val_bpb.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED=$'\033[31m'
RST=$'\033[0m'

die() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit 1
}

# Loud refusals — invent / publish / README-hero flags (harness computes from fixtures).
for arg in "$@"; do
  case "$arg" in
    --auroc|--lab-auroc|--accuracy|--publish|--claim|--invent-metrics|--invent-auroc|--claim-auroc|--readme-hero|--publish-readme|--hero-auroc|--val-bpb|--invent-val-bpb)
      die "Refusing '$arg'.
  Ranking-card runner scores committed local fixtures only.
  Emits length/events baselines (+ optional session BPB model).
  Never invents AUROC heroes, README marketing numbers, or val_bpb.
  Use --baselines-only, --with-model, --fixture NAME, --session-scores-only, or --help."
      ;;
  esac
done

PYTHON="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run python)
else
  RUN=("$PYTHON")
fi

# If caller passes no args, run the publishable fixture path (model + ε).
if [[ $# -eq 0 ]]; then
  set -- --with-model --check-eps
fi

exec "${RUN[@]}" -m eval.run_public_ranking_card "$@"
