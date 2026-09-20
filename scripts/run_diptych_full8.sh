#!/usr/bin/env bash
# Full-8 DIPTYCH adapter gate + optional probe-pair emit (deterministic fixtures).
#
# Audience: strangers regenerating / gating AOMB → DIPTYCH probe JSON.
# Default: full-8 gate + gate_axis_mutate (no MPS train, no CUDA).
# Optional: --emit-dir DIR regenerates validated probe-pair JSON from fixtures.
#
# Never invents AUROC / val_bpb. Never stub-passes. prepare.py untouched.
# Keep refusals in sync with eval.diptych.emit.REFUSED_METRIC_FLAGS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED=$'\033[31m'
RST=$'\033[0m'

# Must match eval.diptych.emit.EXIT_*
EXIT_REFUSED_FLAG=1
EXIT_PATH_ERROR=2

die_refuse() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit "$EXIT_REFUSED_FLAG"
}

die_path() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit "$EXIT_PATH_ERROR"
}

usage() {
  cat >&2 <<'USAGE'
Usage:
  ./scripts/run_diptych_full8.sh [gate args...]
  ./scripts/run_diptych_full8.sh --emit-dir DIR [--emit-only]
  ./scripts/run_diptych_full8.sh --emit-dir DIR   # emit then full-8 gate

Default: full-8 DIPTYCH gate (manifest / contrast / stubs / gate_axis_mutate).

Honest emit:
  --emit-dir DIR   Regenerate probe-pair JSON from diptych-probes/ fixtures
  --emit-only      Emit only (requires --emit-dir); skip gate

Examples:
  ./scripts/run_diptych_full8.sh
  ./scripts/run_diptych_full8.sh --emit-dir /tmp/diptych-emit
  ./scripts/run_diptych_full8.sh --emit-only --emit-dir /tmp/diptych-emit
  python -m eval.diptych.emit --out /tmp/diptych-emit --dry-run

Honesty: fixture probe JSON only — never AUROC / val_bpb / stub-pass.
CUDA gate stays skipped. prepare.py sacred.
USAGE
}

# ── Loud refusals (same set as eval.diptych.emit.REFUSED_METRIC_FLAGS) ───────
for arg in "$@"; do
  key="${arg%%=*}"
  case "$key" in
    --auroc|--lab-auroc|--accuracy|--ranking|--publish|--claim|--invent-metrics|--invent-auroc|--claim-auroc|--readme-hero|--publish-readme|--hero-auroc|--val-bpb|--invent-val-bpb|--stub-pass|--hardcoded-pass|--force-pass|--fake-green|--cuda)
      die_refuse "Refusing '$key'.
  DIPTYCH path regenerates / gates committed fixture probe-pair JSON only.
  Never invents AUROC / val_bpb / published ranking.
  Never stub-passes or hardcoded-passes operators.
  CUDA gate stays skipped — use fixtures + gate_axis_mutate.
  Use --emit-dir DIR, --emit-only, or --help
  (same refusals as: python -m eval.diptych.emit --help)."
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
done

# --emit-only without --emit-dir → path error (match Python CLI)
emit_only=0
has_emit_dir=0
for arg in "$@"; do
  case "$arg" in
    --emit-only) emit_only=1 ;;
    --emit-dir|--emit-dir=*) has_emit_dir=1 ;;
  esac
done
if [[ "$emit_only" -eq 1 && "$has_emit_dir" -eq 0 ]]; then
  die_path "--emit-only requires --emit-dir DIR.
  Example: ./scripts/run_diptych_full8.sh --emit-only --emit-dir /tmp/diptych-emit
  Or:      python -m eval.diptych.emit --out /tmp/diptych-emit"
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run python)
else
  RUN=("$PYTHON")
fi

exec "${RUN[@]}" -m eval.diptych.run_full8 "$@"
