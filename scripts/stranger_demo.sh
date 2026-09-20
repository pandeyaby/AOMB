#!/usr/bin/env bash
# Stranger demo — cross-platform public entry (no Apple MPS, no API keys).
#
# Proven on Linux/CI: DIPTYCH full-8 + gate_axis_mutate, public ranking card
# harness smoke (CPU torch OK for --with-model; baselines-only is faster).
#
# Cite-without-cloning entry (Actions badge / Codespaces): scripts/stranger_verify.sh
#   → docs/stranger-verify.md  (defaults STRANGER_FAST=1; delegates here)
# Clone-first docs: docs/stranger-demo.md
#
# NOT claimed: overnight agent, MPS product train, lab AUROC, production ranking.
# prepare.py is sacred — this script never touches it.
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

# ── Loud refusals for Mac/agent-only expectations ───────────────────────────
if [[ "${STRANGER_EXPECT_MPS:-}" == "1" ]] || [[ "${STRANGER_EXPECT_OVERNIGHT:-}" == "1" ]]; then
  die "This stranger path does not run overnight agent or MPS product train.
  Unset STRANGER_EXPECT_MPS / STRANGER_EXPECT_OVERNIGHT.
  On Apple Silicon with API keys, see README Quickstart (agent_loop / train.py).
  Here we only prove: diptych full-8 gate + public ranking card harness smoke."
fi

for arg in "$@"; do
  case "$arg" in
    --overnight|--agent|--mps-train)
      die "Refusing '$arg'. Stranger demo = no overnight agent, no MPS train, no API keys.
  Run without those flags. Optional: STRANGER_FAST=1 for baselines-only ranking card."
      ;;
  esac
done

banner "AOMB stranger demo (no MPS / no API keys)"
echo "Repo: $ROOT"
echo "Docs: docs/stranger-demo.md"
echo "Story (optional, after gates): docs/anomaly-story.md — val_bpb/surprise IS the signal"
echo

# ── Dependency check ────────────────────────────────────────────────────────
have_uv=0
if command -v uv >/dev/null 2>&1; then
  have_uv=1
fi

PYTHON="${PYTHON:-python3}"
if [[ "$have_uv" -eq 1 ]]; then
  RUN=(uv run python)
  echo "Using: uv run python"
else
  RUN=("$PYTHON")
  echo "Using: $PYTHON (uv not found — pip / system deps must already be present)"
fi

need_torch=1
if [[ "${STRANGER_FAST:-}" == "1" ]] || [[ "${STRANGER_BASELINES_ONLY:-}" == "1" ]]; then
  need_torch=0
fi

check_import() {
  local mod="$1"
  "${RUN[@]}" -c "import $mod" 2>/dev/null
}

if ! check_import pyarrow; then
  die "Missing Python deps (need at least pyarrow/numpy).
  Preferred:  curl -LsSf https://astral.sh/uv/install.sh | sh && uv sync
  Fallback:   pip install pyarrow numpy rustbpe tiktoken
  For full ranking card (--with-model): also
              pip install torch --index-url https://download.pytorch.org/whl/cpu"
fi

if [[ "$need_torch" -eq 1 ]] && ! check_import torch; then
  die "Full ranking-card harness needs torch (CPU wheel is enough — no MPS).
  Install:  uv sync
         or: pip install torch --index-url https://download.pytorch.org/whl/cpu
  Faster subset (no torch):  STRANGER_FAST=1 $0
  That runs length/events baselines + ε check only (still harness smoke, not a model claim)."
fi

# ── 1) DIPTYCH full-8 + gate_axis_mutate ────────────────────────────────────
banner "1/2  DIPTYCH full-8 adapter gate (fixtures + gate_axis_mutate)"
chmod +x scripts/run_diptych_full8.sh
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

# ── 2) Public ranking card harness smoke ────────────────────────────────────
banner "2/2  Public ranking card v1 (harness smoke)"
chmod +x scripts/run_public_ranking_card_v1.sh

if [[ "$need_torch" -eq 0 ]]; then
  echo "${YLW}STRANGER_FAST=1 → baselines-only + ε check (CPU, no model train).${RST}"
  echo "Full harness smoke (fixture model, CPU torch, ~4–10 min): omit STRANGER_FAST."
  ./scripts/run_public_ranking_card_v1.sh --baselines-only --check-eps
  CARD_MODE="baselines-only (length/events ε)"
else
  echo "Full harness smoke: fixture-only short CPU train + ε check (no MPS, no CRISP)."
  ./scripts/run_public_ranking_card_v1.sh --with-model --check-eps
  CARD_MODE="baselines + fixture model (published_fixture_card when model beats baselines)"
fi

# ── Proven vs not claimed ───────────────────────────────────────────────────
banner "What this proved"
echo "${GRN}PROVEN${RST} (Linux / CI / any CPU; no Apple MPS; no Anthropic/OpenAI keys):"
echo "  • DIPTYCH full-8 emit path + gate_axis_mutate → all aomb=green, axis_power=true"
echo "  • Public ranking card: $CARD_MODE"
echo "  • Companion grading lives in DIPTYCH — AOMB only emits probes (no fork of their product)"
echo
echo "${YLW}NOT CLAIMED${RST} / not run by this script:"
echo "  • Overnight agent_loop (needs API keys + usually Mac)"
echo "  • Product MPS train / Uber CRISP val_bpb overnight"
echo "  • Lab ranking AUROC (stays not_published — no invented AUROC)"
echo "  • Production / field accuracy — fixture card is tiny-n synthetic harness smoke only"
echo "  • CRISP val_bpb as ranking accuracy (train fitness only when cited)"
echo
echo "Honesty: docs/stranger-demo.md · docs/stranger-verify.md · docs/public-ranking-card-v1.md · README three lanes"
echo "Cite without cloning: green stranger-verify Actions badge (docs/stranger-verify.md)"
echo "Understand the thesis (~30 min): docs/anomaly-story.md → uv run python demo_anomaly.py"
echo "${GRN}Stranger demo PASS${RST}"
