#!/usr/bin/env bash
# Product Mac smoke — Darwin + MPS gate → short train fitness (val_bpb only).
#
# Audience: Apple Silicon Mac after stranger CPU verify elsewhere.
# Honest: stranger CPU gates ≠ product MPS train. No AUROC. No CUDA. No API keys.
# prepare.py is sacred — never touched here.
#
# Honesty refusals (shared with eval.product_mac_path / session scorer / stranger):
#   EXIT_REFUSED_FLAG=1  (--auroc / --publish / --cuda / invent flags)
#   EXIT_PLATFORM=2      (not Darwin / MPS unavailable on real path;
#                        also missing/malformed Tale measured card on --tale-card-line)
# CI (Linux OK): --dry-run / --help-only / --tale-card-line — no MPS / no full TIME_BUDGET.
#
# Docs: docs/product-mac-path.md · docs/compute-paths.md · docs/crisp-val-bpb-baseline.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED=$'\033[31m'
GRN=$'\033[32m'
YLW=$'\033[33m'
BOLD=$'\033[1m'
RST=$'\033[0m'

# Must match eval.product_mac_path.EXIT_* / REFUSED_METRIC_FLAGS.
EXIT_REFUSED_FLAG=1
EXIT_PLATFORM=2

die() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit 1
}

die_refuse() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit "$EXIT_REFUSED_FLAG"
}

die_platform() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit "$EXIT_PLATFORM"
}

banner() {
  echo
  echo "${BOLD}═══ $* ═══${RST}"
}

usage() {
  cat >&2 <<'USAGE'
Usage:
  ./scripts/product_mac_smoke.sh                 # Darwin + MPS real path
  ./scripts/product_mac_smoke.sh --dry-run       # CI: honesty + wiring (no MPS)
  ./scripts/product_mac_smoke.sh --help-only     # same as --help (exit 0)
  ./scripts/product_mac_smoke.sh --tale-card-line  # CI: factual Tale measured one-liner

Env (real path only):
  PRODUCT_MAC_SMOKE_SECONDS   wall-clock bound (default 60; 0 = full TIME_BUDGET)
  PRODUCT_MAC_CORPUS          auto|smoke (default auto)

Honesty: factual val_bpb on MPS only. Never invents AUROC / published ranking.
  Lab claim_status stays not_published. CUDA gate stays skipped.
  prepare.py sacred. Refused: --auroc / --publish / --cuda / invent flags (exit 1).
  Platform fail (not Darwin / no MPS): exit 2.
  --tale-card-line: prints factual Tale measured card line (train fitness only, not AUROC);
    missing/malformed card → unavailable (exit 2). Linux CI OK (no MPS).
USAGE
}

MODE="real"  # real | dry-run | help | tale-card-line

# ── Loud refusals (before platform / train) ──────────────────────────────────
if [[ "${PRODUCT_MAC_ALLOW_CUDA:-}" == "1" ]] || [[ "${PRODUCT_MAC_FAKE_CUDA:-}" == "1" ]]; then
  die_refuse "Refusing CUDA / fake-CUDA fallback.
  Product Mac smoke is Darwin + MPS only. No CUDA claim path here.
  CUDA gate stays skipped. Lab claim_status stays not_published.
  See docs/compute-paths.md (CUDA checklist) and docs/product-mac-path.md."
fi

# Keep case arm in sync with eval.product_mac_path.REFUSED_METRIC_FLAGS.
for arg in "$@"; do
  key="${arg%%=*}"
  case "$key" in
    --auroc|--lab-auroc|--accuracy|--ranking|--publish|--claim|--invent-metrics|--invent-auroc|--claim-auroc|--val-bpb|--invent-val-bpb|--readme-hero|--publish-readme|--hero-auroc|--cuda|--gpu)
      die_refuse "Refusing '$key'.
  Product Mac smoke reports factual val_bpb on Darwin + MPS only.
  Never invents AUROC / published ranking accuracy.
  Lab claim_status stays not_published.
  CUDA gate stays skipped — no --cuda / --gpu claim path.
  CI: --dry-run or --help-only (no MPS / no full TIME_BUDGET).
  Same refusals as: python -m eval.product_mac_path --help"
      ;;
    --dry-run)
      MODE="dry-run"
      ;;
    --help-only|-h|--help|help)
      MODE="help"
      ;;
    --tale-card-line)
      MODE="tale-card-line"
      ;;
    *)
      die_refuse "Unknown arg '$arg'.
  Use --dry-run / --help-only / --tale-card-line on CI, or no flags on Darwin + MPS.
  See: ./scripts/product_mac_smoke.sh --help"
      ;;
  esac
done

# ── CI dry-run / help-only (Linux OK — no MPS / no TIME_BUDGET) ──────────────
if [[ "$MODE" == "help" ]]; then
  usage
  exit 0
fi

if [[ "$MODE" == "tale-card-line" ]]; then
  banner "AOMB product Mac smoke — Tale measured card line (CI; no MPS)"
  echo "Repo: $ROOT"
  echo "Card: reports/tale-capped/measured_capped_200k.json"
  echo "Honesty: train fitness only — not AUROC / not a published accuracy claim"
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
  # Propagate exit 1 (invent refuse from module) or exit 2 (unavailable card).
  exit "$rc"
fi

if [[ "$MODE" == "dry-run" ]]; then
  banner "AOMB product Mac smoke — dry-run (CI; no MPS)"
  echo "Repo: $ROOT"
  echo "Docs: docs/product-mac-path.md"
  echo
  echo "${GRN}OK${RST}: dry-run honesty path (no Darwin/MPS/train required)."
  echo "  claim_status=not_published"
  echo "  CUDA gate stays skipped"
  echo "  prepare.py sacred (not edited; not invoked here)"
  echo "  Real path still requires Darwin + MPS → factual val_bpb only"
  echo "  No invent AUROC · no full TIME_BUDGET · no CUDA claim"
  echo
  # Thin module parity (torch-free).
  PYTHON="${PYTHON:-python3}"
  if command -v uv >/dev/null 2>&1; then
    uv run python -m eval.product_mac_path --dry-run
  else
    PYTHONPATH="${PYTHONPATH:-$ROOT}" "$PYTHON" -m eval.product_mac_path --dry-run
  fi
  banner "Done (dry-run)"
  echo "On Apple Silicon: ./scripts/product_mac_smoke.sh (real MPS path)."
  exit 0
fi

banner "AOMB product Mac smoke (MPS train fitness — val_bpb only)"
echo "Repo: $ROOT"
echo "Docs: docs/product-mac-path.md"
echo

# Platform gate — fail before any train attempt
if [[ "$(uname -s)" != "Darwin" ]]; then
  die_platform "Not Darwin (detected: $(uname -s)).
  Product Mac path requires macOS + Apple Silicon MPS.
  On Linux / Codespaces / CI use dry-run / tale-card-line or stranger verify instead:
    ./scripts/product_mac_smoke.sh --dry-run
    ./scripts/product_mac_smoke.sh --tale-card-line
    ./scripts/stranger_verify.sh
  See docs/stranger-verify.md · docs/compute-paths.md · docs/product-mac-path.md"
fi

PYTHON="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run python)
  echo "Using: uv run python"
else
  RUN=("$PYTHON")
  echo "Using: $PYTHON"
fi

# MPS capability gate (no silent CPU/CUDA fallback)
"${RUN[@]}" - <<'PY' || die_platform "MPS check failed — install a Metal-capable PyTorch build on Apple Silicon.
  Stranger CPU gates elsewhere do not substitute for product MPS train.
  CUDA gate stays skipped. See docs/product-mac-path.md"
import sys
import torch

if sys.platform != "darwin":
    raise SystemExit(f"platform={sys.platform} (need darwin)")
if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
    raise SystemExit("torch.backends.mps.is_available() is False")
print("MPS available: yes")
print(f"torch={torch.__version__}")
PY

echo "${GRN}OK${RST}: Darwin + MPS gate passed (no CUDA fallback)."
echo

# ── Corpus readiness ─────────────────────────────────────────────────────────
PRODUCT_MAC_CORPUS="${PRODUCT_MAC_CORPUS:-auto}"
CRISP_SHARD_DIR="${CRISP_SHARD_DIR:-$HOME/.cache/autoresearch/data}"
SMOKE_SECONDS="${PRODUCT_MAC_SMOKE_SECONDS:-60}"

has_shards() {
  # prepare.py expects train shards under the cache data dir
  local n
  n="$(find "$CRISP_SHARD_DIR" -maxdepth 1 -name 'shard_*.parquet' 2>/dev/null | wc -l | tr -d ' ')"
  [[ "${n:-0}" -ge 2 ]]
}

case "$PRODUCT_MAC_CORPUS" in
  auto)
    if has_shards; then
      echo "Corpus: existing shards under $CRISP_SHARD_DIR (prefer CRISP product story)."
    else
      die "No train shards found under $CRISP_SHARD_DIR.
  Prefer Uber CRISP (product train):
    uv run python -m corpus.ingest.fetch_crisp
    uv run python -m corpus.ingest.build_shards --adapter crisp_zenodo \\
      --input ~/.cache/autoresearch/corpus-v1/crisp/extracted \\
      --num-train-shards 8 --write-val-shard
    uv run python prepare.py --num-shards 8
  Or synthetic smoke only (not flagship / not comparable to CRISP):
    PRODUCT_MAC_CORPUS=smoke ./scripts/product_mac_smoke.sh
  See docs/product-mac-path.md · docs/crisp-val-bpb-baseline.md"
    fi
    ;;
  smoke)
    echo "${YLW}NOTE:${RST} PRODUCT_MAC_CORPUS=smoke — synthetic generator path (CI/dev only)."
    echo "       Not interchangeable with CRISP val_bpb baselines."
    if ! has_shards; then
      echo "Generating smoke corpus + prepare (prepare.py sacred — invoked, not edited)..."
      "${RUN[@]}" generate_observability_corpus.py
      "${RUN[@]}" prepare.py --num-shards 20
    fi
    ;;
  *)
    die "Unknown PRODUCT_MAC_CORPUS='$PRODUCT_MAC_CORPUS' (use auto|smoke)."
    ;;
esac

# ── Train smoke → factual val_bpb only ───────────────────────────────────────
banner "Train smoke (factual val_bpb only — no AUROC)"
echo "Bound: PRODUCT_MAC_SMOKE_SECONDS=${SMOKE_SECONDS} (0 = full TIME_BUDGET via train.py)"
echo "Honest: this is train fitness, not lab AUROC, not CUDA."
echo

LOG="$(mktemp -t aomb-product-mac-smoke.XXXXXX)"
cleanup() { rm -f "$LOG"; }
trap cleanup EXIT

set +e
if [[ "${SMOKE_SECONDS}" == "0" ]]; then
  "${RUN[@]}" train.py 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
else
  # Bounded smoke — same pattern as README quickstart; exit 0 on alarm is intentional.
  "${RUN[@]}" -c "
import signal, sys
secs = int('${SMOKE_SECONDS}')
def _alarm(s, f):
    print('\\n[product_mac_smoke] wall-clock bound reached (' + str(secs) + 's) — stopping before full TIME_BUDGET.', flush=True)
    sys.exit(0)
signal.signal(signal.SIGALRM, _alarm)
signal.alarm(secs)
exec(open('train.py').read())
" 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
fi
set -e

if [[ "$rc" -ne 0 ]]; then
  die "train.py exited $rc. Fix MPS / corpus / deps, then retry. No fake CUDA fallback."
fi

# Surface factual val_bpb if the run completed far enough to print it
if grep -E '^val_bpb:' "$LOG" >/dev/null 2>&1; then
  echo
  echo "${GRN}Factual train fitness (from this run):${RST}"
  grep -E '^val_bpb:|^training_seconds:|^total_seconds:' "$LOG" || true
  echo
  echo "Cite CRISP baselines only from docs/crisp-val-bpb-baseline.md — do not invent AUROC."
else
  echo
  echo "${YLW}NOTE:${RST} No val_bpb: line yet (smoke bound may have cut before final eval)."
  echo "      Re-run with PRODUCT_MAC_SMOKE_SECONDS=0 for full TIME_BUDGET → val_bpb."
  echo "      Still not lab AUROC. Still not CUDA."
fi

banner "Done"
echo "Next: overnight agent_loop (API keys) is separate — see README."
echo "Lab AUROC remains not_published until docs/lab/publish-checklist.md is green."
