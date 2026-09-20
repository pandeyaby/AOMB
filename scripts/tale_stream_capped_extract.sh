#!/usr/bin/env bash
# Thin wrapper: streaming capped Tale of Errors extract (no full decompress).
#
# Full zstd -d of Tale archives = 300–500 GB — OUT OF SCOPE on ~315 Gi free.
# This only stream-writes enough Jaeger JSON to hit caps, then stops.
#
# Does NOT invent val_bpb / AUROC. prepare.py is sacred — not invoked here.
# Prefer: uv run python -m corpus.ingest.tale_stream_extract --help
#
# Docs: docs/tale-val-bpb-baseline.md · docs/tale-scale.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

usage() {
  cat >&2 <<'USAGE'
Usage:
  ./scripts/tale_stream_capped_extract.sh \
    --input DIR_OR_TAR_ZST --out DIR \
    [--prefix trace1_] [--concat-out PATH] \
    (--max-spans N | --max-files N | --max-bytes N) [...]

Stream-decompress Tale of Errors .tar.zst (or cat sorted trace*_ pieces) and
write a capped Jaeger JSON tree under --out/traces/ for tale_of_errors /
build_shards. Never materializes the full 300–500 GB tree.

Refused: --auroc / ranking / --full-decompress / uncapped extract / invented metrics.

Equivalent:
  uv run python -m corpus.ingest.tale_stream_extract --help
USAGE
}

# Loud refusals before forwarding.
for arg in "$@"; do
  case "$arg" in
    -h|--help|help)
      usage
      exit 0
      ;;
    --auroc|--lab-auroc|--accuracy|--ranking|--publish|--claim|--val-bpb)
      die "Refusing '$arg'.
  Streaming extract is disk-safe prep for the *train* lane only.
  Tale dumps have no AOMB incident labels → no AUROC.
  This wrapper never invents val_bpb. See docs/tale-val-bpb-baseline.md"
      ;;
    --full-decompress|--decompress-all|--uncapped|--download-all)
      die "Refusing '$arg'.
  Full Tale decompress is OUT OF SCOPE (~315 Gi free ≠ 300–500 GB/archive).
  Pass --max-spans / --max-files / --max-bytes instead."
      ;;
  esac
done

[[ $# -gt 0 ]] || { usage; exit 2; }

if command -v uv >/dev/null 2>&1; then
  exec uv run python -m corpus.ingest.tale_stream_extract "$@"
else
  exec python3 -m corpus.ingest.tale_stream_extract "$@"
fi
