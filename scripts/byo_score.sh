#!/usr/bin/env bash
# BYO session scorer — one-command polish over score_session / eval.score_cli.
#
# Audience: someone with their own OTel / telemetry dump who wants session
# surprise (BPB) without inventing AUROC.
#
# Prints session BPB / dry-run metadata only. Never AUROC. Never a published
# ranking claim. Lab stays claim_status=not_published.
#
# Hardened to share refusals + exit codes with eval.score_cli (post-#46):
#   EXIT_REFUSED_FLAG=1  (--auroc / --publish / invent flags)
#   EXIT_PATH_ERROR=2    (missing dump / bad first arg)
# Dry-run and real score (--checkpoint / --train-seconds) share the same gate.
#
# prepare.py is sacred — never touched here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RED=$'\033[31m'
YLW=$'\033[33m'
BOLD=$'\033[1m'
RST=$'\033[0m'

# Must match eval.score_cli.EXIT_* and REFUSED_METRIC_FLAGS.
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

Honesty: session BPB ≠ published ranking / AUROC. Lab stays not_published.
Help: python -m score_session --help
USAGE
}

banner() {
  echo >&2
  echo "${BOLD}═══ $* ═══${RST}" >&2
}

# ── Loud refusals (same set as eval.score_cli.REFUSED_METRIC_FLAGS) ──────────
# Applies before mode detection so --dry-run and real score paths match.
for arg in "$@"; do
  key="${arg%%=*}"
  case "$key" in
    --auroc|--lab-auroc|--accuracy|--ranking|--publish|--claim|--invent-metrics|--invent-auroc|--claim-auroc)
      die_refuse "Refusing '$key'.
  This wrapper prints session BPB / surprise only — never AUROC, never a
  published ranking claim. Lab stays claim_status=not_published.
  Use --dry-run, --checkpoint PATH, or --train-seconds N
  (same refusals as: python -m score_session --help)."
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
  exit "$EXIT_PATH_ERROR"
fi

DUMP="$1"
shift

if [[ -z "$DUMP" || "$DUMP" == -* ]]; then
  echo "${RED}ERROR:${RST} first argument must be a dump path (got: ${DUMP:-empty})." >&2
  echo >&2
  usage
  exit "$EXIT_PATH_ERROR"
fi

if [[ ! -e "$DUMP" ]]; then
  die_path "Dump path not found: $DUMP
  Pass a real OTLP JSONL / Jaeger / parquet directory (or file).
  Fixtures: corpus/fixtures/lab_sample · corpus/fixtures/crisp_sample
  Same exit code as: python -m score_session --input <dump> --dry-run"
fi

# Detect scoring mode among remaining args
MODE=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) MODE="dry-run" ;;
    --train-seconds|--train-seconds=*) MODE="train" ;;
    --checkpoint|--checkpoint=*) MODE="checkpoint" ;;
  esac
done

# Default: cheap dry-run so a stranger can smoke without torch / prepared shards
EXTRA=()
if [[ -z "$MODE" ]]; then
  EXTRA=(--dry-run)
  MODE="dry-run"
fi

banner "AOMB BYO session scorer (BPB / surprise only — not published ranking)"
echo "${YLW}BANNER:${RST} Scoring ≠ published ranking claim. Session BPB is diagnostic surprise." >&2
echo "         Do NOT invent AUROC / accuracy from this output." >&2
echo "         Lab claim_status stays not_published." >&2
echo "Repo: $ROOT" >&2
echo "Dump: $DUMP" >&2
echo "Mode: $MODE" >&2
echo >&2

PYTHON="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
  RUN=(uv run python)
  echo "Using: uv run python" >&2
else
  RUN=("$PYTHON")
  echo "Using: $PYTHON" >&2
fi

# Thin wrap → score_session → eval.score_cli (prepare.evaluate_bpb untouched).
# exec preserves score_cli exit codes (2 = path error, 1 = refused flag).
exec "${RUN[@]}" -m score_session --input "$DUMP" "${EXTRA[@]}" "$@"
