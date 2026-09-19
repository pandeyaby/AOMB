#!/usr/bin/env bash
# Stranger verify — one-command public gate (no Apple MPS, no API keys).
#
# Prefer scripts/stranger_demo.sh when present (e.g. after #21 merges).
# Until then, this thin vendor runs:
#   1) ./scripts/run_diptych_full8.sh (+ axis_power assert)
#   2) ranking-card baselines-only + ε  (STRANGER_FAST=1 default)
#
# NOT claimed: lab AUROC, production accuracy, MPS train, overnight agent.
# prepare.py is sacred — never touched here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED=$'\033[31m'
GRN=$'\033[32m'
YLW=$'\033[33m'
BOLD=$'\033[1m'
RST=$'\033[0m'

die() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit 1
}

banner() {
  echo
  echo "${BOLD}═══ $* ═══${RST}"
}

# Loud refusals for Mac/agent-only expectations
if [[ "${STRANGER_EXPECT_MPS:-}" == "1" ]] || [[ "${STRANGER_EXPECT_OVERNIGHT:-}" == "1" ]]; then
  die "Stranger verify does not run overnight agent or MPS product train.
  Unset STRANGER_EXPECT_MPS / STRANGER_EXPECT_OVERNIGHT.
  Cite path proves: diptych full-8 + gate_axis_mutate, ranking-card baselines ε (or demo script)."
fi

for arg in "$@"; do
  case "$arg" in
    --overnight|--agent|--mps-train)
      die "Refusing '$arg'. Stranger verify = no overnight, no MPS train, no API keys."
      ;;
  esac
done

# Default cite path is fast (baselines-only). Set STRANGER_FAST=0 for full
# ranking-card --with-model when torch CPU is installed (still not MPS/lab AUROC).
export STRANGER_FAST="${STRANGER_FAST:-1}"

banner "AOMB stranger verify (no MPS / no API keys)"
echo "Repo: $ROOT"
echo "Docs: docs/stranger-verify.md"
echo "STRANGER_FAST=${STRANGER_FAST}"
echo

# Prefer the fuller stranger_demo.sh once it lands on main (#21).
if [[ -x scripts/stranger_demo.sh ]] || [[ -f scripts/stranger_demo.sh ]]; then
  chmod +x scripts/stranger_demo.sh scripts/run_diptych_full8.sh scripts/run_public_ranking_card_v1.sh
  echo "Delegating to scripts/stranger_demo.sh (present on this tree)."
  if [[ "${STRANGER_FAST}" == "1" ]]; then
    exec env STRANGER_FAST=1 ./scripts/stranger_demo.sh
  else
    exec env -u STRANGER_FAST ./scripts/stranger_demo.sh
  fi
fi

# ── Thin vendor path (main before #21) ──────────────────────────────────────
PYTHON="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run python)
  echo "Using: uv run python"
else
  RUN=("$PYTHON")
  echo "Using: $PYTHON"
fi

check_import() {
  local mod="$1"
  "${RUN[@]}" -c "import $mod" 2>/dev/null
}

if ! check_import pyarrow; then
  die "Missing Python deps (need at least pyarrow/numpy).
  Preferred:  curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync
  Fallback:   pip install pyarrow numpy rustbpe tiktoken"
fi

banner "1/2  DIPTYCH full-8 adapter gate (fixtures + gate_axis_mutate)"
chmod +x scripts/run_diptych_full8.sh scripts/run_public_ranking_card_v1.sh
./scripts/run_diptych_full8.sh

"${RUN[@]}" - <<'PY'
import json, sys
from pathlib import Path
m = json.loads(Path("coverage/matrix.json").read_text())
bad = {
    op: c for op, c in m["operators"].items()
    if c.get("aomb") != "green" or c.get("axis_power") is not True
}
if bad:
    print("non-green or missing axis_power:", bad)
    sys.exit(1)
print("all 8 aomb cells green with axis_power=true (gate_axis_mutate)")
PY

banner "2/2  Public ranking card v1 (harness smoke)"
if [[ "${STRANGER_FAST}" == "1" ]]; then
  echo "${YLW}STRANGER_FAST=1 → baselines-only + ε check (CPU, no model train).${RST}"
  ./scripts/run_public_ranking_card_v1.sh --baselines-only --check-eps
  CARD_MODE="baselines-only (length/events ε)"
else
  if ! check_import torch; then
    die "STRANGER_FAST=0 needs torch (CPU wheel OK — no MPS).
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  Or keep default: STRANGER_FAST=1 $0"
  fi
  echo "Full harness smoke: fixture-only short CPU train + ε (no MPS, no CRISP)."
  ./scripts/run_public_ranking_card_v1.sh --with-model --check-eps
  CARD_MODE="baselines + fixture model (published_fixture_card when model beats baselines)"
fi

banner "What this proved"
echo "${GRN}PROVEN${RST} (Linux / CI / Codespaces; no Apple MPS; no API keys):"
echo "  • DIPTYCH full-8 emit path + gate_axis_mutate → all aomb=green, axis_power=true"
echo "  • Public ranking card: $CARD_MODE"
echo "  • Companion grading lives in DIPTYCH — AOMB only emits probes"
echo
echo "${YLW}NOT CLAIMED${RST} / not run by this script:"
echo "  • Overnight agent_loop (needs API keys + usually Mac)"
echo "  • Product MPS train / Uber CRISP val_bpb overnight"
echo "  • Lab ranking AUROC (stays not_published — no invented AUROC)"
echo "  • Production / field accuracy — fixture card is tiny-n synthetic harness smoke only"
echo
echo "Honesty: docs/stranger-verify.md · docs/public-ranking-card-v1.md · README three lanes"
echo "${GRN}Stranger verify PASS${RST}"
