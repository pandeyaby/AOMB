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
# Honesty refusals (shared with eval.stranger_path / session scorer / demo_anomaly):
#   EXIT_REFUSED_FLAG=1  (--auroc / --publish / --cuda / invent / overnight)
#   EXIT_PATH_ERROR=2    (missing/malformed Tale measured card on --tale-card-line)
# CI-safe: --tale-card-line → factual Tale measured one-liner (no MPS).
# NOT claimed: overnight agent, MPS product train, lab AUROC, production ranking.
# prepare.py is sacred — this script never touches it. CUDA gate stays skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED=$'\033[31m'
GRN=$'\033[32m'
YLW=$'\033[33m'
BOLD=$'\033[1m'
RST=$'\033[0m'

# Must match eval.stranger_path.EXIT_* / REFUSED_METRIC_FLAGS.
EXIT_REFUSED_FLAG=1
EXIT_PATH_ERROR=2  # missing/malformed Tale measured card on --tale-card-line

die() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit 1
}

die_refuse() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit "$EXIT_REFUSED_FLAG"
}

banner() {
  echo
  echo "${BOLD}═══ $* ═══${RST}"
}

usage() {
  cat >&2 <<'USAGE'
Usage:
  ./scripts/stranger_demo.sh
  STRANGER_FAST=1 ./scripts/stranger_demo.sh   # baselines-only ranking card
  ./scripts/stranger_demo.sh --tale-card-line  # factual Tale measured one-liner (CI)

Cite-without-cloning (defaults STRANGER_FAST=1):
  ./scripts/stranger_verify.sh

Honesty: DIPTYCH full-8 + ranking-card harness smoke only.
  Never invents AUROC / val_bpb / published ranking.
  Lab claim_status stays not_published. CUDA gate stays skipped.
  prepare.py sacred. Refused: --auroc / --publish / --cuda / invent flags.
  --tale-card-line: factual Tale measured card line (train fitness only, not AUROC);
    missing/malformed → unavailable (exit 2). Linux CI OK (no MPS).
USAGE
}

# ── Loud refusals for Mac/agent-only expectations ───────────────────────────
if [[ "${STRANGER_EXPECT_MPS:-}" == "1" ]] || [[ "${STRANGER_EXPECT_OVERNIGHT:-}" == "1" ]]; then
  die_refuse "This stranger path does not run overnight agent or MPS product train.
  Unset STRANGER_EXPECT_MPS / STRANGER_EXPECT_OVERNIGHT.
  On Apple Silicon with API keys, see README Quickstart (agent_loop / train.py).
  Here we only prove: diptych full-8 gate + public ranking card harness smoke."
fi

# Keep case arm in sync with eval.stranger_path.REFUSED_METRIC_FLAGS (+ product).
TALE_CARD_LINE=0
for arg in "$@"; do
  key="${arg%%=*}"
  case "$key" in
    --auroc|--lab-auroc|--accuracy|--ranking|--publish|--claim|--invent-metrics|--invent-auroc|--claim-auroc|--val-bpb|--invent-val-bpb|--readme-hero|--publish-readme|--hero-auroc|--cuda|--gpu)
      die_refuse "Refusing '$key'.
  Stranger demo proves DIPTYCH full-8 + ranking-card harness smoke only.
  Never invents AUROC / val_bpb / published ranking accuracy.
  Lab claim_status stays not_published.
  CUDA gate stays skipped — use CPU stranger path (no --cuda / --gpu).
  Run without invent flags. Optional: STRANGER_FAST=1 for baselines-only
  (same refusals as: python -m eval.stranger_path --help)."
      ;;
    --overnight|--agent|--mps-train)
      die_refuse "Refusing '$key'. Stranger demo = no overnight agent, no MPS train, no API keys.
  Run without those flags. Optional: STRANGER_FAST=1 for baselines-only ranking card."
      ;;
    --tale-card-line)
      TALE_CARD_LINE=1
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
done


# ── Optional Tale measured-card one-liner (CI-safe; no MPS / no full demo) ────
if [[ "$TALE_CARD_LINE" -eq 1 ]]; then
  banner "AOMB stranger — Tale measured card line (CI; no MPS)"
  echo "Repo: $ROOT"
  echo "Card: reports/tale-capped/measured_capped_200k.json"
  echo "Honesty: train fitness only — not AUROC / measured_not_published is not a published accuracy claim"
  echo
  PYTHON="${PYTHON:-python3}"
  set +e
  if command -v uv >/dev/null 2>&1; then
    uv run python -m eval.public_wins_tale_line
    rc=$?
  else
    PYTHONPATH="${PYTHONPATH:-$ROOT}" "$PYTHON" -m eval.public_wins_tale_line
    rc=$?
  fi
  set -e
  if [[ "$rc" -eq 0 ]]; then
    banner "Done (tale-card-line)"
    exit 0
  fi
  exit "$rc"
fi

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
echo "  • CUDA (gate stays skipped)"
echo
echo "Honesty: docs/stranger-demo.md · docs/stranger-verify.md · docs/public-ranking-card-v1.md · README three lanes"
echo "Cite without cloning: green stranger-verify Actions badge (docs/stranger-verify.md)"
echo "Understand the thesis (~30 min): docs/anomaly-story.md → uv run python demo_anomaly.py"
echo "${GRN}Stranger demo PASS${RST}"
