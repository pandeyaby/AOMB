#!/usr/bin/env bash
# Tale-scale smoke — fixture / capped shard build for Uber Tale of Errors.
#
# Audience: outsider verifying the flagship-scale *train* path wiring without
# pulling multi-GB Zenodo archives into CI / a laptop by accident.
#
# Default: corpus/fixtures/tale_of_errors_sample + --max-spans cap.
# Never invents val_bpb / AUROC. Lab stays claim_status=not_published.
#
# Honesty refusals (mirror eval.stranger_path / product_mac_smoke invent set):
#   EXIT_REFUSED_FLAG=1  (--auroc / --publish / --cuda / invent synonyms)
# Also refuses: --download / --download-all / Zenodo bulk pulls.
# CUDA gate stays skipped. No Zenodo / no MPS required for refusals.
#
# Docs: docs/tale-scale.md · docs/corpus-v1.md
# prepare.py is sacred — optional invoke only (--prepare); never edited here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED=$'\033[31m'
YLW=$'\033[33m'
GRN=$'\033[32m'
BOLD=$'\033[1m'
RST=$'\033[0m'

# Must match stranger / product_mac invent refusals (exit 1).
EXIT_REFUSED_FLAG=1

die() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit 1
}

die_refuse() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit "$EXIT_REFUSED_FLAG"
}

usage() {
  cat >&2 <<'USAGE'
Usage:
  ./scripts/tale_scale_smoke.sh [options]

Default (no network, no multi-GB download):
  Build shards from corpus/fixtures/tale_of_errors_sample with a span cap.

Options:
  --input PATH          Local assembled Jaeger tree (or fixture). Path must exist.
  --max-spans N         Cap spans (default: 100). Like CRISP subset smokes.
  --num-train-shards N  Train shard count (default: 1)
  --data-dir PATH       Shard output dir (default: /tmp/aomb-tale-smoke)
  --prepare             Also run prepare.py --num-shards N (sacred invoke; macOS/Metal)
  --list-only           Proxy to fetch_tale_of_errors --list-only (needs network; no download)
  -h, --help            Show this help

Explicitly refused (exit 1; use fetch module on a big disk, not this smoke):
  --auroc / --publish / --cuda / invent synonyms (same set as stranger / product_mac)
  --download / --download-all / Zenodo bulk pulls
  Invented metrics

Docs: docs/tale-scale.md
Honesty: CRISP/Tale = train lane (val_bpb). No incident labels → no AUROC.
         Lab stays not_published. CUDA gate stays skipped.
USAGE
}

banner() {
  echo
  echo "${BOLD}═══ $* ═══${RST}"
}

INPUT="corpus/fixtures/tale_of_errors_sample"
MAX_SPANS=100
NUM_TRAIN_SHARDS=1
DATA_DIR="/tmp/aomb-tale-smoke"
DO_PREPARE=0
DO_LIST_ONLY=0

# ── Loud refusals (keep in sync with stranger / product_mac invent set) ──────
for arg in "$@"; do
  key="${arg%%=*}"
  case "$key" in
    --auroc|--lab-auroc|--accuracy|--ranking|--publish|--claim|--invent-metrics|--invent-auroc|--claim-auroc|--val-bpb|--invent-val-bpb|--readme-hero|--publish-readme|--hero-auroc|--cuda|--gpu)
      die_refuse "Refusing '$key'.
  Tale-scale is the public-real *train* lane (factual val_bpb when you train).
  Never invents AUROC / published ranking / val_bpb numbers.
  No incident labels on Tale dumps → no AUROC from this path.
  Lab stays claim_status=not_published.
  CUDA gate stays skipped — no --cuda / --gpu claim path.
  See docs/tale-scale.md · docs/public-accuracy-eval.md · docs/lab/publish-checklist.md"
      ;;
    --download|--download-all|--fetch-all|--full-decompress|--decompress-all|--decompress|--assemble-all)
      die_refuse "Refusing '$key' in the smoke wrapper.
  Multi-GB Zenodo pulls are opt-in via:
    uv run python -m corpus.ingest.fetch_tale_of_errors --list-only
    uv run python -m corpus.ingest.fetch_tale_of_errors --download <FILE>
  Full pull needs hundreds of GB free and is refused in CI.
  This smoke uses the fixture only (or a local --input you already assembled).
  See docs/tale-scale.md"
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
done

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      [[ $# -ge 2 ]] || die "--input requires a path"
      INPUT="$2"
      shift 2
      ;;
    --max-spans)
      [[ $# -ge 2 ]] || die "--max-spans requires an integer"
      MAX_SPANS="$2"
      shift 2
      ;;
    --num-train-shards)
      [[ $# -ge 2 ]] || die "--num-train-shards requires an integer"
      NUM_TRAIN_SHARDS="$2"
      shift 2
      ;;
    --data-dir)
      [[ $# -ge 2 ]] || die "--data-dir requires a path"
      DATA_DIR="$2"
      shift 2
      ;;
    --prepare)
      DO_PREPARE=1
      shift
      ;;
    --list-only)
      DO_LIST_ONLY=1
      shift
      ;;
    *)
      die "Unknown argument: $1
$(usage)"
      ;;
  esac
done

PYTHON="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run python)
else
  RUN=("$PYTHON")
fi

banner "AOMB Tale-scale smoke (train lane wiring — not AUROC)"
echo "${YLW}BANNER:${RST} CRISP / Tale = public-real *train* lane."
echo "         No incident labels on Tale → do NOT invent AUROC / ranking accuracy."
echo "         Lab claim_status stays not_published."
echo "         This script never fabricates val_bpb."
echo "Repo: $ROOT"
echo "Docs: docs/tale-scale.md"
echo

# Optional: list Zenodo files (network). Fail loud if unreachable.
if [[ "$DO_LIST_ONLY" -eq 1 ]]; then
  banner "Zenodo list-only (API; no download)"
  echo "Needs network. Does not pull multi-GB archives."
  set +e
  "${RUN[@]}" -m corpus.ingest.fetch_tale_of_errors --list-only
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    die "fetch_tale_of_errors --list-only failed (exit $rc).
  Check network / Zenodo reachability, or skip --list-only and use the fixture smoke.
  Docs: docs/tale-scale.md"
  fi
  echo
  echo "${GRN}OK${RST}: listed Zenodo files (no bulk download)."
  echo "Next: selective --download on a big disk, or fixture smoke without --list-only."
  exit 0
fi

# Resolve input path (relative to repo root when not absolute)
if [[ "$INPUT" != /* ]]; then
  INPUT_ABS="$ROOT/$INPUT"
else
  INPUT_ABS="$INPUT"
fi

if [[ ! -e "$INPUT_ABS" ]]; then
  die "Input path not found: $INPUT_ABS
  Pass a real assembled Jaeger JSON tree, or use the fixture:
    ./scripts/tale_scale_smoke.sh
    ./scripts/tale_scale_smoke.sh --input corpus/fixtures/tale_of_errors_sample
  Do not expect this script to download Zenodo for you.
  Scale path: docs/tale-scale.md"
fi

# If user pointed at the default cache dir but never assembled — fail loud
DEFAULT_CACHE="${HOME}/.cache/autoresearch/corpus-v1/tale_of_errors"
if [[ "$INPUT_ABS" == "$DEFAULT_CACHE" || "$INPUT_ABS" == "$DEFAULT_CACHE/"* ]]; then
  if [[ ! -d "$INPUT_ABS" ]] || [[ -z "$(find "$INPUT_ABS" -name '*.json' 2>/dev/null | head -1)" ]]; then
    die "Tale cache path has no Jaeger JSON yet: $INPUT_ABS
  Fetch + assemble first (hundreds of GB free), then re-run with --input.
  Or use the in-repo fixture (default). See docs/tale-scale.md"
  fi
fi

banner "Build shards (capped)"
echo "Adapter: tale_of_errors"
echo "Input:   $INPUT_ABS"
echo "max-spans: $MAX_SPANS"
echo "train shards: $NUM_TRAIN_SHARDS"
echo "data-dir: $DATA_DIR"
echo

mkdir -p "$DATA_DIR"

"${RUN[@]}" -m corpus.ingest.build_shards \
  --adapter tale_of_errors \
  --input "$INPUT_ABS" \
  --max-spans "$MAX_SPANS" \
  --num-train-shards "$NUM_TRAIN_SHARDS" \
  --write-val-shard \
  --data-dir "$DATA_DIR"

echo
echo "${GRN}OK${RST}: shard build finished under $DATA_DIR"
echo "     Factual next step (optional, needs time/disk): prepare.py → train.py → read val_bpb."
echo "     Still not AUROC. Still not a published ranking claim."

if [[ "$DO_PREPARE" -eq 1 ]]; then
  banner "prepare.py (sacred invoke — not edited)"
  if [[ "$(uname -s)" != "Darwin" ]]; then
    die "prepare.py requires macOS + Metal (this host: $(uname -s)).
  Shard smoke above is enough on Linux / CI.
  On a Mac: docs/product-mac-path.md · docs/tale-scale.md"
  fi
  # prepare.py reads from the default cache data dir unless configured otherwise.
  if [[ "$DATA_DIR" != "${HOME}/.cache/autoresearch/data" && "$DATA_DIR" != "$HOME/.cache/autoresearch/data" ]]; then
    echo "${YLW}NOTE:${RST} shards are under $DATA_DIR."
    echo "      prepare.py expects ~/.cache/autoresearch/data by default."
    echo "      Copy/symlink shards there, or re-run smoke with:"
    echo "        --data-dir \$HOME/.cache/autoresearch/data"
    die "Refusing ambiguous --prepare with non-default --data-dir (no silent wrong train)."
  fi
  "${RUN[@]}" prepare.py --num-shards "$NUM_TRAIN_SHARDS"
  echo "${GRN}OK${RST}: prepare.py finished. Short train fitness:"
  echo "  uv run python train.py"
  echo "  # or 60s bound — see docs/tale-scale.md"
  echo "Cite only the val_bpb your run prints — do not invent Tale baselines or AUROC."
fi

banner "Done"
echo "Docs: docs/tale-scale.md · provenance CC BY 4.0 (Zenodo 13947828 + 13952897)"
echo "Lab AUROC remains not_published."
