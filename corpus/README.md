# AOMB corpus tree

Flagship reference corpus = **real production telemetry** (see [`docs/corpus-v1.md`](../docs/corpus-v1.md)).

```
corpus/
  README.md
  fixtures/
    crisp_sample/             # tiny Jaeger JSON for unit tests (not product data)
    tale_of_errors_sample/    # tiny Jaeger JSON, tale_of_errors provenance (smoke only)
    lab_sample/               # tiny lab capture for unit tests
    public_ranking_card_v1/   # synthetic labeled pack for public ranking card v1
  ingest/
    jaeger.py                 # Jaeger API JSON → SpanRecord
    session_format.py         # prepare.py text contract
    otlp_to_sessions.py       # spans/logs → session docs
    build_shards.py           # shard_NNNNN.parquet + provenance
    fetch_crisp.py            # Zenodo CRISP-main.zip (~2.33 GB, opt-in)
    fetch_tale_of_errors.py   # Zenodo part1+part2 list/selective download (not CI)
    fetch_aiops_challenge.py  # eval-only cite+fetch (no redistribute)
    rejected_sources.py       # explicit denylist (demo/testbed/synthetic)
    adapters/
      base.py
      crisp_zenodo.py         # v1 bootstrap public-real
      tale_of_errors.py       # flagship-scale path (do not mix sanitization with CRISP)
      lab_capture.py          # lab JSONL + provenance windows
      byo.py                  # user OTLP JSONL / Jaeger / parquet sessions
```

Public accuracy ranking protocol (claim not published): [`docs/public-accuracy-eval.md`](../docs/public-accuracy-eval.md), harness [`eval/`](../eval/).
**Public ranking card v1 (fixture / harness smoke):** [`docs/public-ranking-card-v1.md`](../docs/public-ranking-card-v1.md) — synthetic pack (72 sessions, n_eval=36 balanced); distinct from private lab pool (`docs/lab/`). Reproduce: `./scripts/run_public_ranking_card_v1.sh`. Not production AUROC.
BYO ingest + session scorer: [`docs/byo-and-scorer.md`](../docs/byo-and-scorer.md).
Tale-scale (flagship public-real train path): [`docs/tale-scale.md`](../docs/tale-scale.md) · `./scripts/tale_scale_smoke.sh`.

## Quick commands

```bash
# v1 bootstrap — Uber CRISP (CC BY 4.0)
uv run python -m corpus.ingest.fetch_crisp              # instructions / local zip
uv run python -m corpus.ingest.fetch_crisp --download   # ~2.33 GB, not CI
uv run python -m corpus.ingest.build_shards \
  --adapter crisp_zenodo \
  --input ~/.cache/autoresearch/corpus-v1/crisp/extracted \
  --num-train-shards 8 --write-val-shard

# Flagship scale — Uber Tale of Errors (CC BY 4.0; do not mix sanitization with CRISP)
# One-pager: docs/tale-scale.md · fixture smoke (no multi-GB): ./scripts/tale_scale_smoke.sh
uv run python -m corpus.ingest.fetch_tale_of_errors --list-only
uv run python -m corpus.ingest.fetch_tale_of_errors --download trace1_aa   # one piece; not CI
# After assembling Jaeger tree (cat split parts + zstd; 300–500 GB/archive):
uv run python -m corpus.ingest.build_shards \
  --adapter tale_of_errors \
  --input /path/to/assembled/jaeger/tree \
  --max-spans N --num-train-shards 8 --write-val-shard
# Smoke fixture (no Zenodo):
./scripts/tale_scale_smoke.sh
# or:
uv run python -m corpus.ingest.build_shards \
  --adapter tale_of_errors \
  --input corpus/fixtures/tale_of_errors_sample \
  --max-spans 100 --num-train-shards 1 --write-val-shard

# Lab
uv run python -m corpus.ingest.build_shards \
  --adapter lab_capture --input lab/captures/<id> \
  --num-train-shards 4 --write-val-shard

# Eval-only (non-commercial; do not redistribute)
uv run python -m corpus.ingest.fetch_aiops_challenge

# Public accuracy ranking eval (protocol scaffolding — claim not published)
uv run python -m eval.run_eval \
  --capture corpus/fixtures/lab_sample \
  --scores-from length \
  --out-dir /tmp/aomb-eval-smoke
# See docs/public-accuracy-eval.md

# Public ranking card v1 — fixture baselines (claim not published; not lab pool)
./scripts/run_public_ranking_card_v1.sh
# See docs/public-ranking-card-v1.md

# BYO dump → shards (OTLP JSONL / Jaeger / parquet); see docs/byo-and-scorer.md
uv run python -m corpus.ingest.build_shards \
  --adapter byo --input corpus/fixtures/lab_sample \
  --num-train-shards 2 --write-val-shard --data-dir /tmp/aomb-byo

# Session scorer dry-run (not a public accuracy claim)
./scripts/byo_score.sh corpus/fixtures/lab_sample
# equivalent:
uv run python -m score_session --input corpus/fixtures/lab_sample --dry-run
```

Synthetic `generate_observability_corpus.py` = **smoke/CI only**.
