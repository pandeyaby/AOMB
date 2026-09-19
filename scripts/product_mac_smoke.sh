#!/usr/bin/env bash
# Product Mac smoke — Darwin + MPS gate → short train fitness (val_bpb only).
#
# Audience: Apple Silicon Mac after stranger CPU verify elsewhere.
# Honest: stranger CPU gates ≠ product MPS train. No AUROC. No CUDA. No API keys.
# prepare.py is sacred — never touched here.
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

die() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit 1
}

banner() {
  echo
  echo "${BOLD}═══ $* ═══${RST}"
}

# ── Loud refusals ────────────────────────────────────────────────────────────
if [[ "${PRODUCT_MAC_ALLOW_CUDA:-}" == "1" ]] || [[ "${PRODUCT_MAC_FAKE_CUDA:-}" == "1" ]]; then
  die "Refusing CUDA / fake-CUDA fallback.
  Product Mac smoke is Darwin + MPS only. No CUDA claim path here.
  See docs/compute-paths.md (CUDA checklist) and docs/product-mac-path.md."
fi

for arg in "$@"; do
  case "$arg" in
    --cuda|--gpu|--auroc|--lab-auroc)
      die "Refusing '$arg'. This script reports factual val_bpb on MPS only.
  Lab AUROC stays not_published — docs/lab/publish-checklist.md.
  CUDA = checklist only — docs/compute-paths.md."
      ;;
  esac
done

banner "AOMB product Mac smoke (MPS train fitness — val_bpb only)"
echo "Repo: $ROOT"
echo "Docs: docs/product-mac-path.md"
echo

# Platform gate — fail before any train attempt
if [[ "$(uname -s)" != "Darwin" ]]; then
  die "Not Darwin (detected: $(uname -s)).
  Product Mac path requires macOS + Apple Silicon MPS.
  On Linux / Codespaces / CI use stranger verify instead:
    ./scripts/stranger_verify.sh
  See docs/stranger-verify.md · docs/compute-paths.md"
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
"${RUN[@]}" - <<'PY' || die "MPS check failed — install a Metal-capable PyTorch build on Apple Silicon.
  Stranger CPU gates elsewhere do not substitute for product MPS train.
  See docs/product-mac-path.md"
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
