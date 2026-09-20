#!/usr/bin/env bash
# Stream-capped Tale extract → shards → optional prepare / train / score.
#
# Mac MPS product path: extract (or local tree) → build_shards --max-spans N
# → prepare.py (sacred) → short train / score.
# CI / Linux: --fixture or synthetic --extract-input + --score-dry-run.
#
# Never invents val_bpb / AUROC. No full decompress. No uncapped shards.
# Lab claim_status stays not_published. prepare.py is sacred — invoke only.
#
# Prefer: uv run python -m corpus.ingest.tale_capped_pipeline --help
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED=$'\033[31m'
YLW=$'\033[33m'
GRN=$'\033[32m'
BOLD=$'\033[1m'
RST=$'\033[0m'

die() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit 1
}

usage() {
  cat >&2 <<'USAGE'
Usage:
  ./scripts/tale_capped_train_score.sh --fixture --max-spans N [--score-dry-run]
  ./scripts/tale_capped_train_score.sh --extract-input PATH --max-spans N [opts]
  ./scripts/tale_capped_train_score.sh --jaeger-tree PATH --max-spans N [opts]

E2E train/score wiring on a *stream-capped* Tale subset (Mac MPS) or fixture CI.

Required:
  --max-spans N              Span cap (extract + shards; no silent uncapped)

Input (exactly one):
  --fixture                  In-repo corpus/fixtures/tale_of_errors_sample
  --extract-input PATH       .tar.zst or trace*_ pieces → tale_stream_extract
  --jaeger-tree PATH         Existing capped Jaeger JSON tree

Optional:
  --data-dir PATH            Shard dir (default: /tmp/aomb-tale-capped-pipeline)
  --extract-out PATH         Extract out (default: <data-dir>/extracted)
  --extract-max-files N      Extra extract file cap
  --num-train-shards N       Default: 1
  --prepare                  Invoke sacred prepare.py (Darwin + Metal only)
  --train [--train-seconds N]  Bounded train.py smoke (Darwin + MPS)
  --score-dry-run            eval.score_cli --dry-run (CI-safe)
  --score [--score-out PATH] Score path (pair with --score-dry-run on CI)
  -h, --help                 Show this help

Refused: --auroc / ranking / invent metrics / --full-decompress / --uncapped

Equivalent:
  uv run python -m corpus.ingest.tale_capped_pipeline --help
USAGE
}

banner() {
  echo
  echo "${BOLD}═══ $* ═══${RST}"
}

# ── Loud refusals ────────────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    -h|--help|help)
      usage
      exit 0
      ;;
    --auroc|--lab-auroc|--accuracy|--ranking|--publish|--claim|--val-bpb|--invent-metrics|--invent-val-bpb)
      die "Refusing '$arg'.
  This wrapper is the *train/score* lane on a stream-capped Tale subset.
  Tale dumps have no AOMB incident labels → no AUROC.
  Never invents val_bpb. Lab claim_status stays not_published."
      ;;
    --full-decompress|--decompress-all|--uncapped|--download-all)
      die "Refusing '$arg'.
  Full Tale decompress is OUT OF SCOPE (300–500 GB/archive).
  Pass --max-spans N (and optional --extract-max-files).
  Stream extract: uv run python -m corpus.ingest.tale_stream_extract --help"
      ;;
  esac
done

[[ $# -gt 0 ]] || { usage; exit 2; }

banner "AOMB Tale capped train/score (stream-capped — not AUROC)"
echo "${YLW}BANNER:${RST} Stream-capped extract → shards → optional prepare/train/score."
echo "         Full decompress OUT OF SCOPE. No invented val_bpb / AUROC."
echo "         Lab claim_status stays not_published. prepare.py is sacred."
echo "Repo: $ROOT"
echo

PYTHON="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run python)
else
  RUN=("$PYTHON")
fi

"${RUN[@]}" -m corpus.ingest.tale_capped_pipeline "$@"
rc=$?
if [[ "$rc" -ne 0 ]]; then
  die "tale_capped_pipeline exited $rc"
fi

echo
echo "${GRN}OK${RST}: pipeline finished (claim_status=not_published)."
echo "     Mac next (if shards + prepare done): cite only printed val_bpb — never invent AUROC."
exit 0
