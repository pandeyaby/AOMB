#!/usr/bin/env bash
# Tale capped val_bpb baseline — disk-safe public-real subset path.
#
# Audience: Mac / big-disk outsider who will NOT full-decompress Tale
# (hundreds of GB per archive). Maintainer Mac ~315 Gi free → capped only.
#
# Requires:
#   --max-spans N   (explicit; no silent uncapped shard)
#   --input PATH    (existing local Jaeger JSON tree)
# OR:
#   --fetch-key KEY (single Zenodo key only; documents selective fetch)
#
# Honesty refusals (mirror eval.stranger_path / product_mac_smoke invent set):
#   EXIT_REFUSED_FLAG=1  (--auroc / --publish / --cuda / invent synonyms)
# Also refuses: --download-all / full decompress / multi-key bulk pulls.
# Never invents val_bpb. Lab stays claim_status=not_published.
# prepare.py is sacred — optional invoke only (--prepare); never edited here.
# CUDA gate stays skipped. No Zenodo bulk / no MPS required for refusals.
#
# Docs: docs/tale-val-bpb-baseline.md · docs/tale-scale.md · docs/corpus-v1.md
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

# Free-disk floors (GiB). Full Tale decompress is hundreds of GB — out of scope.
MIN_FREE_SHARD_GIB=8          # local --input → shard / prepare room
MIN_FREE_FETCH_GIB=5          # one selective Zenodo piece
MIN_FREE_DECOMPRESS_GIB=350   # refuse any decompress-intent path under this

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
  ./scripts/tale_capped_baseline.sh --input PATH --max-spans N [options]
  ./scripts/tale_capped_baseline.sh --fetch-key KEY [--out DIR]

Capped Tale public-real train lane (factual val_bpb when you train on Mac).
Full archive decompress is OUT OF SCOPE (~315 Gi free ≠ 300–500 GB/archive).

Required (shard path):
  --input PATH          Existing local assembled Jaeger JSON tree (must exist)
  --max-spans N         Explicit span cap (required — no silent uncapped build)

Fetch path (single key only; no shard until you have a local tree):
  --fetch-key KEY       Download one Zenodo file key via fetch_tale_of_errors
  --out DIR             Download dir (default: ~/.cache/autoresearch/corpus-v1/tale_of_errors)

Optional (shard path):
  --num-train-shards N  Train shard count (default: 8)
  --data-dir PATH       Shard output dir (default: ~/.cache/autoresearch/data)
  --prepare             Also run prepare.py --num-shards N (sacred; macOS/Metal)
  --min-free-gib N      Override free-disk floor for the chosen path
  -h, --help            Show this help

Explicitly refused (exit 1):
  --auroc / --publish / --cuda / invent synonyms (same set as stranger / product_mac)
  --download-all / multi-key bulk / full corpus / decompress pulls
  Invented metrics / fixture-as-baseline without a real tree

Docs: docs/tale-val-bpb-baseline.md
Honesty: train lane only. No incident labels → no AUROC. Lab stays not_published.
         Do not invent Tale val_bpb — row stays pending until a measured Mac run.
         CUDA gate stays skipped.
USAGE
}

banner() {
  echo
  echo "${BOLD}═══ $* ═══${RST}"
}

free_gib() {
  # Portable-ish: df -Pk reports 1K-blocks on Linux/macOS when available.
  local target="${1:-.}"
  local avail_k
  avail_k="$(df -Pk "$target" 2>/dev/null | awk 'NR==2 {print $4}')"
  if [[ -z "${avail_k:-}" || ! "$avail_k" =~ ^[0-9]+$ ]]; then
    echo ""
    return 0
  fi
  echo $((avail_k / 1024 / 1024))
}

require_free_gib() {
  local need="$1"
  local target="${2:-.}"
  local label="${3:-chosen path}"
  local have
  have="$(free_gib "$target")"
  if [[ -z "$have" ]]; then
    die "Could not read free disk for $target (df failed).
  Refusing to continue silently for: $label
  Docs: docs/tale-val-bpb-baseline.md"
  fi
  echo "Free disk at $target: ${have} Gi (floor for $label: ${need} Gi)"
  if [[ "$have" -lt "$need" ]]; then
    die "Free disk looks unsafe for $label (${have} Gi < ${need} Gi).
  Full Tale decompress needs ~${MIN_FREE_DECOMPRESS_GIB}+ Gi free per archive — OUT OF SCOPE.
  Capped subset only: existing local --input + --max-spans, or one --fetch-key.
  Maintainer Mac ~315 Gi free is not enough for full decompress.
  See docs/tale-val-bpb-baseline.md"
  fi
}

INPUT=""
MAX_SPANS=""
NUM_TRAIN_SHARDS=8
DATA_DIR="${HOME}/.cache/autoresearch/data"
DO_PREPARE=0
FETCH_KEY=""
FETCH_OUT="${HOME}/.cache/autoresearch/corpus-v1/tale_of_errors"
MIN_FREE_OVERRIDE=""

# ── Loud refusals (keep in sync with stranger / product_mac invent set) ──────
for arg in "$@"; do
  key="${arg%%=*}"
  case "$key" in
    --auroc|--lab-auroc|--accuracy|--ranking|--publish|--claim|--invent-metrics|--invent-auroc|--claim-auroc|--val-bpb|--invent-val-bpb|--readme-hero|--publish-readme|--hero-auroc|--cuda|--gpu)
      die_refuse "Refusing '$key'.
  Tale capped baseline is the public-real *train* lane (factual val_bpb when you train).
  Never invents AUROC / published ranking / val_bpb numbers.
  No incident labels on Tale dumps → no AUROC from this path.
  Lab stays claim_status=not_published.
  CUDA gate stays skipped — no --cuda / --gpu claim path.
  See docs/tale-val-bpb-baseline.md · docs/public-accuracy-eval.md · docs/lab/publish-checklist.md"
      ;;
    --download-all|--fetch-all|--download|--decompress|--assemble-all|--full-decompress|--decompress-all)
      die_refuse "Refusing '$key' in the capped baseline wrapper.
  Full / multi-GB / decompress-all paths are OUT OF SCOPE (~315 Gi free ≠ hundreds of GB/archive).
  Allowed:
    --fetch-key <ONE_KEY>     # selective single Zenodo piece
    --input <EXISTING_TREE> --max-spans N
  Or use the module intentionally on a machine with hundreds of GB free:
    uv run python -m corpus.ingest.fetch_tale_of_errors --list-only
  See docs/tale-val-bpb-baseline.md · docs/tale-scale.md"
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
    --fetch-key)
      [[ $# -ge 2 ]] || die "--fetch-key requires a Zenodo file key"
      FETCH_KEY="$2"
      shift 2
      ;;
    --out)
      [[ $# -ge 2 ]] || die "--out requires a path"
      FETCH_OUT="$2"
      shift 2
      ;;
    --min-free-gib)
      [[ $# -ge 2 ]] || die "--min-free-gib requires an integer"
      MIN_FREE_OVERRIDE="$2"
      shift 2
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

banner "AOMB Tale capped baseline (train lane — not AUROC)"
echo "${YLW}BANNER:${RST} Capped public-real Tale subset only."
echo "         Full decompress OUT OF SCOPE (300–500 GB/archive; ~315 Gi free typical)."
echo "         No incident labels → do NOT invent AUROC / ranking accuracy."
echo "         Lab claim_status stays not_published."
echo "         This script never fabricates val_bpb (doc row stays pending until measured)."
echo "Repo: $ROOT"
echo "Docs: docs/tale-val-bpb-baseline.md"
echo

# ── Single-key fetch path ────────────────────────────────────────────────────
if [[ -n "$FETCH_KEY" ]]; then
  if [[ -n "$INPUT" || -n "$MAX_SPANS" ]]; then
    die "Use either --fetch-key OR (--input + --max-spans), not both in one invocation.
  1) ./scripts/tale_capped_baseline.sh --fetch-key trace1_aa
  2) After you have a *local assembled Jaeger tree* that fits on disk:
       ./scripts/tale_capped_baseline.sh --input /path/to/tree --max-spans N
  Docs: docs/tale-val-bpb-baseline.md"
  fi
  # Refuse multi-key smuggling via commas/spaces already split — one key only.
  if [[ "$FETCH_KEY" == *" "* || "$FETCH_KEY" == *","* ]]; then
    die "Refusing multi-key fetch ('$FETCH_KEY'). Pass exactly one --fetch-key.
  Full pulls / --download-all are out of scope for this capped baseline."
  fi
  floor="${MIN_FREE_OVERRIDE:-$MIN_FREE_FETCH_GIB}"
  mkdir -p "$FETCH_OUT"
  require_free_gib "$floor" "$FETCH_OUT" "single Zenodo --fetch-key"

  have="$(free_gib "$FETCH_OUT")"
  if [[ -n "$have" && "$have" -lt "$MIN_FREE_DECOMPRESS_GIB" ]]; then
    echo "${YLW}NOTE:${RST} ${have} Gi free < ${MIN_FREE_DECOMPRESS_GIB} Gi."
    echo "      Fetching one compressed piece is allowed; full assemble+decompress is NOT."
    echo "      Do not cat/zstd the full archive on this disk. Capped local --input only."
  fi

  banner "Selective fetch (one Zenodo key)"
  echo "Key: $FETCH_KEY"
  echo "Out: $FETCH_OUT"
  echo
  "${RUN[@]}" -m corpus.ingest.fetch_tale_of_errors \
    --download "$FETCH_KEY" \
    --out "$FETCH_OUT"
  echo
  echo "${GRN}OK${RST}: single key fetched (or already complete)."
  echo "     A split piece is not a Jaeger tree. Full part decompress remains OUT OF SCOPE"
  echo "     when free disk is under ~${MIN_FREE_DECOMPRESS_GIB} Gi."
  echo "Next (only with an existing local assembled tree that fits):"
  echo "  ./scripts/tale_capped_baseline.sh --input /path/to/jaeger/tree --max-spans N"
  echo "Docs: docs/tale-val-bpb-baseline.md"
  echo "val_bpb row stays pending / not yet measured until a Mac TIME_BUDGET run."
  exit 0
fi

# ── Shard path: require --input + --max-spans ────────────────────────────────
[[ -n "$INPUT" ]] || die "Missing --input PATH (existing local Jaeger tree).
  Or document a single-key fetch: --fetch-key KEY
$(usage)"
[[ -n "$MAX_SPANS" ]] || die "Missing required --max-spans N (explicit cap; no silent uncapped build).
  Example: --max-spans 200000
  Docs: docs/tale-val-bpb-baseline.md"

if [[ ! "$MAX_SPANS" =~ ^[1-9][0-9]*$ ]]; then
  die "--max-spans must be a positive integer (got: $MAX_SPANS)"
fi

if [[ "$INPUT" != /* ]]; then
  INPUT_ABS="$ROOT/$INPUT"
else
  INPUT_ABS="$INPUT"
fi

if [[ ! -e "$INPUT_ABS" ]]; then
  die "Input path not found: $INPUT_ABS
  Pass an existing local assembled Jaeger JSON tree.
  This wrapper does not download Zenodo for the shard path (use --fetch-key for one piece).
  Fixture wiring smoke (not a baseline number): ./scripts/tale_scale_smoke.sh
  Docs: docs/tale-val-bpb-baseline.md"
fi

# Refuse treating the in-repo fixture as a "public-real baseline" silently
FIXTURE_ABS="$ROOT/corpus/fixtures/tale_of_errors_sample"
if [[ "$INPUT_ABS" == "$FIXTURE_ABS" || "$INPUT_ABS" == "$FIXTURE_ABS/"* ]]; then
  die "Refusing in-repo fixture as capped *baseline* input: $INPUT_ABS
  Fixture = adapter wiring only → ./scripts/tale_scale_smoke.sh
  This script is for a local public-real (or locally extracted) Jaeger tree + --max-spans.
  Docs: docs/tale-val-bpb-baseline.md · docs/tale-scale.md"
fi

if [[ ! -d "$INPUT_ABS" ]] || [[ -z "$(find "$INPUT_ABS" -name '*.json' 2>/dev/null | head -1)" ]]; then
  die "No Jaeger JSON under --input: $INPUT_ABS
  Point at a directory tree of .json traces, or fetch one key (not a ready tree):
    ./scripts/tale_capped_baseline.sh --fetch-key trace1_aa
  Docs: docs/tale-val-bpb-baseline.md"
fi

floor="${MIN_FREE_OVERRIDE:-$MIN_FREE_SHARD_GIB}"
mkdir -p "$DATA_DIR"
require_free_gib "$floor" "$DATA_DIR" "capped shard / prepare"

banner "Build shards (capped public-real)"
echo "Adapter: tale_of_errors"
echo "Input:   $INPUT_ABS"
echo "max-spans: $MAX_SPANS"
echo "train shards: $NUM_TRAIN_SHARDS"
echo "data-dir: $DATA_DIR"
echo

"${RUN[@]}" -m corpus.ingest.build_shards \
  --adapter tale_of_errors \
  --input "$INPUT_ABS" \
  --max-spans "$MAX_SPANS" \
  --num-train-shards "$NUM_TRAIN_SHARDS" \
  --write-val-shard \
  --data-dir "$DATA_DIR"

echo
echo "${GRN}OK${RST}: capped shard build finished under $DATA_DIR"
echo "     Factual next step (Mac + Metal): prepare.py → train.py → read val_bpb."
echo "     Still not AUROC. Still not a published ranking claim."
echo "     Do not invent a number in docs/tale-val-bpb-baseline.md — fill only after measure."

if [[ "$DO_PREPARE" -eq 1 ]]; then
  banner "prepare.py (sacred invoke — not edited)"
  if [[ "$(uname -s)" != "Darwin" ]]; then
    die "prepare.py requires macOS + Metal (this host: $(uname -s)).
  Capped shard build above is enough on Linux / CI.
  On a Mac: docs/product-mac-path.md · docs/tale-val-bpb-baseline.md"
  fi
  if [[ "$DATA_DIR" != "${HOME}/.cache/autoresearch/data" && "$DATA_DIR" != "$HOME/.cache/autoresearch/data" ]]; then
    echo "${YLW}NOTE:${RST} shards are under $DATA_DIR."
    echo "      prepare.py expects ~/.cache/autoresearch/data by default."
    die "Refusing ambiguous --prepare with non-default --data-dir (no silent wrong train)."
  fi
  "${RUN[@]}" prepare.py --num-shards "$NUM_TRAIN_SHARDS"
  echo "${GRN}OK${RST}: prepare.py finished. TIME_BUDGET train fitness:"
  echo "  uv run python train.py"
  echo "Cite only the val_bpb your run prints — update the pending row; never invent AUROC."
fi

banner "Done"
echo "Docs: docs/tale-val-bpb-baseline.md · provenance CC BY 4.0 (Zenodo 13947828 + 13952897)"
echo "Measured Tale val_bpb: pending until a Mac capped run is attached."
echo "Lab AUROC remains not_published. HOLD merge on invented metrics."
