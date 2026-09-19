#!/usr/bin/env bash
# BYO session scorer — one-command polish over score_session / eval.score_cli.
#
# Audience: someone with their own OTel / telemetry dump who wants session
# surprise (BPB) without inventing AUROC.
#
# Prints session BPB / dry-run metadata only. Never AUROC. Never a published
# ranking claim. Lab stays claim_status=not_published.
#
# Docs: docs/byo-and-scorer.md · docs/public-accuracy-eval.md
# prepare.py is sacred — never touched here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED=$'\033[31m'
YLW=$'\033[33m'
BOLD=$'\033[1m'
RST=$'\033[0m'

die() {
  echo "${RED}ERROR:${RST} $*" >&2
  exit 1
}

usage() {
  cat >&2 <<'USAGE'
Usage:
  ./scripts/byo_score.sh <dump-path> [--dry-run | --train-seconds N | --checkpoint PATH] [score_session args...]

Required:
  <dump-path>   Your OTLP JSONL / Jaeger / parquet dump (or a public fixture)

Modes (pick one; default = --dry-run if none given):
  --dry-run           Load sessions; print ids / lengths (no torch)
  --train-seconds N   Short train-then-score (needs prepared tokenizer + shards)
  --checkpoint PATH   Score from a saved aomb_session_scorer_v1 checkpoint

Examples:
  ./scripts/byo_score.sh corpus/fixtures/lab_sample
  ./scripts/byo_score.sh /path/to/my_dump --dry-run
  ./scripts/byo_score.sh /path/to/my_dump --train-seconds 30 --out /tmp/byo-score.json
  ./scripts/byo_score.sh /path/to/my_dump --checkpoint /tmp/aomb-scorer.pt --json

Docs: docs/byo-and-scorer.md
Honesty: session BPB ≠ published ranking / AUROC. Lab stays not_published.
USAGE
}

banner() {
  echo
  echo "${BOLD}═══ $* ═══${RST}"
}

# ── Loud refusals (never invent ranking / AUROC) ─────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --auroc|--lab-auroc|--accuracy|--ranking|--publish|--claim)
      die "Refusing '$arg'.
  This wrapper prints session BPB / surprise only — never AUROC, never a
  published ranking claim. Lab stays claim_status=not_published.
  See docs/byo-and-scorer.md · docs/public-accuracy-eval.md · docs/lab/publish-checklist.md"
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
done

if [[ $# -lt 1 ]]; then
  echo "${RED}ERROR:${RST} missing dump path." >&2
  echo >&2
  usage
  exit 1
fi

DUMP="$1"
shift

if [[ -z "$DUMP" || "$DUMP" == -* ]]; then
  echo "${RED}ERROR:${RST} first argument must be a dump path (got: ${DUMP:-empty})." >&2
  echo >&2
  usage
  exit 1
fi

if [[ ! -e "$DUMP" ]]; then
  die "Dump path not found: $DUMP
  Pass a real OTLP JSONL / Jaeger / parquet directory (or file).
  Fixtures: corpus/fixtures/lab_sample · corpus/fixtures/crisp_sample
  See docs/byo-and-scorer.md"
fi

# Detect scoring mode among remaining args
MODE=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) MODE="dry-run" ;;
    --train-seconds) MODE="train" ;;
    --checkpoint) MODE="checkpoint" ;;
  esac
done

# Default: cheap dry-run so a stranger can smoke without torch / prepared shards
EXTRA=()
if [[ -z "$MODE" ]]; then
  EXTRA=(--dry-run)
  MODE="dry-run"
fi

banner "AOMB BYO session scorer (BPB / surprise only — not published ranking)"
echo "${YLW}BANNER:${RST} Scoring ≠ published ranking claim. Session BPB is diagnostic surprise."
echo "         Do NOT invent AUROC / accuracy from this output."
echo "         Lab claim_status stays not_published."
echo "Repo: $ROOT"
echo "Dump: $DUMP"
echo "Mode: $MODE"
echo "Docs: docs/byo-and-scorer.md"
echo

PYTHON="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run python)
  echo "Using: uv run python"
else
  RUN=("$PYTHON")
  echo "Using: $PYTHON"
fi

# Thin wrap → score_session → eval.score_cli (prepare.evaluate_bpb untouched)
exec "${RUN[@]}" -m score_session --input "$DUMP" "${EXTRA[@]}" "$@"
