# AOMB Reference Corpus v1 — Provenance & Schema

**Product decision (locked):** the flagship corpus is **real telemetry only**.
Synthetic data from `generate_observability_corpus.py` is retained for **smoke / CI only** — not the product story.

v1 is dual-source:

| Source | What it is | Role |
|--------|------------|------|
| **(A) Public-real** | Licensed real OTel traces/logs from a public dataset | Reproducible baseline anyone can fetch |
| **(B) Lab-captured** | OTel from an org-level local multi-service stack with induced faults | Controlled normal vs incident windows |

---

## Downstream contract (sacred)

`prepare.py` / `evaluate_bpb` / the BPE tokenizer must keep working unchanged.

Ingest writes the **same parquet shape** that the synthetic generator already produces:

- Path: `~/.cache/autoresearch/data/shard_NNNNN.parquet`
- Schema: single column `text` (`string`)
- Each row = one **session document** (multi-line string of correlated events)
- Pinned validation shard index: `6542` (matches `prepare.py`)

Event line shape (compatible with existing demos / visualization heuristics):

```text
[ts=YYYY-MM-DDTHH:MM:SS.sssZ] [src=OTel] trace_id=... span_id=... parent=... op=... svc=... duration_ms=... status=ok|error ...
[ts=...] [src=OTelLog] level=INFO svc=... msg=... trace_id=... ...
```

Optional provenance prefix on the first line of a session (ignored by training as ordinary text; useful for audit):

```text
# aomb_meta source=lab window=incident capture_id=2026-09-11T18:00:00Z fault=api_latency
```

---

## Provenance record schema

Every ingested batch SHOULD write a JSON provenance sidecar next to the raw capture
(or under `~/.cache/autoresearch/corpus-v1/provenance/`).

```json
{
  "corpus_version": "v1",
  "source_id": "otel-demo-hf | lab-aomb-stack",
  "source_kind": "public_real | lab_capture",
  "license": "Apache-2.0",
  "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
  "citation": "see Sources below",
  "captured_at": "ISO-8601 UTC",
  "capture_tool": "corpus/ingest/fetch_otel_demo.py | lab/scripts/capture.sh",
  "window_label": "normal | incident | mixed",
  "windows": [
    {
      "label": "normal",
      "start": "ISO-8601",
      "end": "ISO-8601",
      "notes": "steady load, no faults"
    },
    {
      "label": "incident",
      "start": "ISO-8601",
      "end": "ISO-8601",
      "fault": "api_latency | api_errors | kill_redis | kill_postgres",
      "notes": "fault injection active"
    }
  ],
  "signals": ["traces", "logs"],
  "raw_path": "path to OTLP JSON / parquet dump",
  "session_count": 0,
  "shard_indices": [0],
  "converter": "corpus/ingest/otlp_to_sessions.py",
  "notes": ""
}
```

**Labels** attach via **capture window metadata**, not by inventing per-event anomaly flags.
A session whose spans fall primarily inside an `incident` window inherits `window=incident`.

---

## Source A — Public real: OpenTelemetry Demo telemetry

| Field | Value |
|-------|--------|
| **Dataset** | [`smithclay/otel-demo-telemetry`](https://huggingface.co/datasets/smithclay/otel-demo-telemetry) |
| **Contents** | Real OTLP traces, logs, and metrics from the [OpenTelemetry Demo](https://github.com/open-telemetry/opentelemetry-demo) (Astronomy Shop), captured to Parquet via duckdb-otlp |
| **License** | **Apache-2.0** (dataset card) |
| **Upstream demo license** | Apache-2.0 |
| **Why this dataset** | Real multi-service OTel (not synthetic generators); parquet on Hugging Face; commercially permissive license |
| **Citation** | Smithclay / OpenTelemetry Demo community capture — Hugging Face dataset `smithclay/otel-demo-telemetry`; upstream https://github.com/open-telemetry/opentelemetry-demo |

### How to reproduce (public-real)

```bash
# Download traces (+ optional logs) into ~/.cache/autoresearch/corpus-v1/public/
uv run python -m corpus.ingest.fetch_otel_demo --signals traces,logs

# Convert OTLP parquet → AOMB session shards (text column)
uv run python -m corpus.ingest.build_shards \
  --adapter otel_demo_hf \
  --input ~/.cache/autoresearch/corpus-v1/public \
  --num-train-shards 8 \
  --write-val-shard
```

If Hugging Face download is blocked or must be manual:

1. Download `otlp_traces/**/*.parquet` (and optionally `otlp_logs/**`) from the dataset page.
2. Place them under `~/.cache/autoresearch/corpus-v1/public/`.
3. Run `build_shards` as above.

The adapter interface (`corpus/ingest/adapters/base.py`) is the extension point for additional public sources.

---

## Source B — Lab-captured OTel

| Field | Value |
|-------|--------|
| **Stack** | `lab/docker-compose.yml` — frontend, API, Postgres, Redis, OpenTelemetry Collector |
| **Export** | OTLP/HTTP → collector → JSONL files under `lab/captures/` |
| **Faults** | `lab/scripts/inject_faults.sh` — latency, HTTP errors, kill Redis/Postgres |
| **Capture** | `lab/scripts/run_capture_session.sh` tags **normal** then **incident** windows |

### How to reproduce (lab)

```bash
cd lab
docker compose up -d --build
./scripts/run_capture_session.sh   # normal window → faults → incident window → export
docker compose down

# Convert capture JSONL → session shards
uv run python -m corpus.ingest.build_shards \
  --adapter lab_capture \
  --input lab/captures/<capture_id> \
  --num-train-shards 4 \
  --write-val-shard
```

See [`lab/README.md`](../lab/README.md) for ports, fault modes, and file layout.

---

## Building the combined v1 corpus

Recommended order:

1. Ingest public-real into train shards `0 .. N-1`.
2. Ingest lab captures into subsequent train shards.
3. Hold out a fixed val shard at index `6542` (mix of public + lab normal/incident, or lab-only pinned set).
4. Run `uv run python prepare.py --num-shards <N>` to train the tokenizer on the real shards.
5. Smoke-test with `uv run train.py` / `demo_anomaly.py`.

**Do not** mix synthetic shards into the flagship v1 story. Use synthetic only when you need a fast CI path without Docker/HF.

---

## Honesty constraints

- Do **not** invent fake “real” telemetry.
- If a public fetch is not wired yet, leave a clear TODO on the adapter and document the exact source (as above).
- Window labels come from capture metadata or dataset time bounds you document — not from heuristic “looks anomalous” guessing for the product corpus.

---

## Hypothesis (verified by design)

> Existing parquet session format can be filled from OTLP spans/logs with a converter; labels attach via capture window metadata.

Implemented by `corpus/ingest/otlp_to_sessions.py` + provenance sidecars. Spans group by `trace_id` into sessions; logs correlate by `trace_id` when present, else by time proximity within a window.
