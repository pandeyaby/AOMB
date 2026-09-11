# AOMB corpus tree

Flagship reference corpus = **real telemetry** (see [`docs/corpus-v1.md`](../docs/corpus-v1.md)).

```
corpus/
  README.md                 # this file
  fixtures/                 # tiny sample OTLP for unit tests (not product data)
  ingest/
    session_format.py       # event / session text helpers (prepare.py contract)
    otlp_to_sessions.py     # OTLP spans+logs → session documents
    build_shards.py         # write shard_NNNNN.parquet + provenance
    fetch_otel_demo.py      # download public-real HF dataset
    adapters/
      base.py               # SourceAdapter interface
      otel_demo_hf.py       # public-real adapter
      lab_capture.py        # lab JSONL adapter
```

## Quick commands

```bash
# Public-real
uv run python -m corpus.ingest.fetch_otel_demo --signals traces,logs
uv run python -m corpus.ingest.build_shards --adapter otel_demo_hf \
  --input ~/.cache/autoresearch/corpus-v1/public --num-train-shards 8 --write-val-shard

# Lab (after docker capture)
uv run python -m corpus.ingest.build_shards --adapter lab_capture \
  --input lab/captures/<id> --num-train-shards 4 --write-val-shard
```

Synthetic generator (`generate_observability_corpus.py`) is **smoke/CI only**.
