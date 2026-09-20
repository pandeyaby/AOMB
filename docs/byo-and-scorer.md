# Bring your own (BYO) telemetry + shippable session scorer

Scaffolding for ingesting **user-provided** dumps and scoring sessions with
per-session surprise / bits-per-byte (BPB).

> **Scoring ≠ public accuracy claim.**  
> Session BPB is a diagnostic surprise score. A **public accuracy claim**
> requires labeled ranking metrics and the pass/fail checklist in
> [`public-accuracy-eval.md`](public-accuracy-eval.md).  
> Do **not** invent AUROC / accuracy numbers from scorer output.  
> CRISP overnight `val_bpb=0.4309` remains a README fact only (unlabeled).

`prepare.py` / `evaluate_bpb` stay sacred — this path never modifies them.

---

## Expected dump shapes

### 1. OTLP JSONL directory

```
my_dump/
  provenance.json          # optional but recommended (windows, license, capture_id)
  traces.jsonl             # one span object per line (or OTLP resourceSpans docs)
  logs.jsonl               # optional
```

Flat span JSON (same as lab export):

```json
{
  "trace_id": "...",
  "span_id": "...",
  "parent_span_id": "",
  "name": "GET /api",
  "service_name": "api",
  "start_time": "2026-09-11T17:01:00.100Z",
  "duration_ms": 42,
  "status_code": "ok",
  "attributes": {"http.method": "GET", "http.status_code": 200}
}
```

Also accepted: collector file-exporter JSONL with `resourceSpans` / `resourceLogs`.

Without `provenance.json`, windows default to `unknown` (fine for training shards;
**not** enough for the public accuracy ranking claim).

### 2. Jaeger JSON

Directory of Jaeger HTTP API JSON files (CRISP-style), or a single `.json` file:

```json
{
  "data": [
    {
      "traceID": "...",
      "spans": [ ... ],
      "processes": { "p1": { "serviceName": "svc" } }
    }
  ]
}
```

Session key = `traceID` (multi-service spans share one AOMB document).

### 3. Parquet sessions

Already in AOMB session format — column **`text`** (string), same contract as
`prepare.py` shards (`shard_NNNNN.parquet`). Rows pass through; no fake events
are invented.

---

## Build shards from a BYO dump

```bash
# Auto-detect OTLP JSONL / Jaeger / parquet
uv run python -m corpus.ingest.build_shards \
  --adapter byo \
  --input /path/to/your/dump \
  --num-train-shards 8 \
  --write-val-shard

# Then (unchanged sacred path)
uv run python prepare.py --num-shards 8
```

Provenance is written under `~/.cache/autoresearch/corpus-v1/provenance/` with
`adapter=byo`, detected format, input path, and source metadata. The adapter
**does not invent telemetry** — it only reformats what you provide into the
session line format (`corpus/ingest/session_format.py`).

Fixtures for smoke tests:

| Fixture | Format |
|---------|--------|
| `corpus/fixtures/lab_sample` | OTLP-ish JSONL (+ provenance windows) |
| `corpus/fixtures/crisp_sample` | Jaeger JSON |

```bash
uv run python -m corpus.ingest.build_shards \
  --adapter byo --input corpus/fixtures/lab_sample \
  --num-train-shards 2 --write-val-shard --data-dir /tmp/aomb-byo-lab

uv run python -m corpus.ingest.build_shards \
  --adapter byo --input corpus/fixtures/crisp_sample \
  --num-train-shards 1 --write-val-shard --data-dir /tmp/aomb-byo-crisp
```

---

## Score sessions (one-command polish)

Thin wrapper (fails loudly without a dump path; defaults to `--dry-run`):

```bash
./scripts/byo_score.sh corpus/fixtures/lab_sample
./scripts/byo_score.sh /path/to/your/dump --dry-run
./scripts/byo_score.sh /path/to/your/dump --train-seconds 30 --out /tmp/byo-score.json
./scripts/byo_score.sh /path/to/your/dump --checkpoint /tmp/aomb-scorer.pt --json
```

Loud banner: **scoring ≠ published ranking claim**; lab stays `not_published`.
Refuses `--auroc` / ranking flags. Same honesty as below.

Direct CLI (equivalent):

```bash
# Dry-run — load + print session ids (no torch / no train)
uv run python -m eval.score_cli --input corpus/fixtures/lab_sample --dry-run
# equivalent:
uv run python -m score_session --input corpus/fixtures/lab_sample --dry-run

# Short train-then-score (requires prepared tokenizer + train shards)
uv run python -m score_session \
  --input corpus/fixtures/lab_sample \
  --train-seconds 30 \
  --save-checkpoint /tmp/aomb-scorer.pt \
  --out /tmp/aomb-score-report.json

# Score from a saved checkpoint
uv run python -m score_session \
  --input /path/to/dump \
  --checkpoint /tmp/aomb-scorer.pt \
  --json
```

Output: one line per session with `session_id`, `label` (if known), `n_chars`,
and `bpb` (bits-per-byte surprise). Optional `--out` writes a JSON report with
`claim_status=not_published`. Never AUROC.

Implementation notes:

- Reuses `eval.score.session_bpb_texts` (same CE→BPB spirit as `demo_anomaly`)
- Loads `prepare.Tokenizer` / train symbols the same way as `eval.run_eval`
- **Never** calls `prepare.evaluate_bpb`
- Checkpoint format: `aomb_session_scorer_v1` (`model_state_dict` + meta)

---

## What this is / is not

| Artifact | Role |
|----------|------|
| BYO ingest → shards | Train on your real dump |
| Session scorer BPB | Per-session surprise diagnostics |
| `evaluate_bpb` / CRISP `0.4309` | Shard-level / overnight breeding facts — **not** accuracy |
| Public accuracy claim | Only after [`public-accuracy-eval.md`](public-accuracy-eval.md) checklist |

Synthetic `generate_observability_corpus.py` remains **smoke/CI only**.

**Before BYO:** get the thesis in ~30 min — [`anomaly-story.md`](anomaly-story.md) (`demo_anomaly.py`). Session BPB here is the same surprise family.
