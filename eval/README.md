# Public accuracy eval harness

Scaffolding for the protocol in [`docs/public-accuracy-eval.md`](../docs/public-accuracy-eval.md).

**Public ranking card v1 (fixture):** frozen card protocol [`docs/public-ranking-card-v1.md`](../docs/public-ranking-card-v1.md) — `python -m eval.run_public_ranking_card` / `./scripts/run_public_ranking_card_v1.sh`. Fixture **eval-split** baselines + optional fixture-only model (`--train-corpus fixture-train`). Distinct from private lab pool (`docs/lab/`).

**Claim status:** fixture card may be `published_fixture_card` (harness smoke) when model mean AUROC beats length+events on the frozen synthetic eval split — **not** production AUROC. Still never cite lab-pool AUROC or CRISP `val_bpb` as this card.

## What this does

1. Load lab_capture sessions + `provenance.json` window labels (`normal` vs `incident`/`cascade`)
2. Score sessions (length baseline, event-count baseline, precomputed scores, or optional train-then-score session BPB)
3. Emit **JSON + markdown** with AUROC, PR-AUC, precision@k, random ranking baseline, seed, corpus id/hash, `train.py` / `prepare.py` SHAs
4. Multi-seed runner → mean±std aggregate

`prepare.evaluate_bpb` is **not** called or modified. Session BPB scoring (model path) uses the same CE→bits/byte idea as `demo_anomaly.session_bpb`.

## Window ↔ event timestamp alignment

Session labels come from **capture metadata**, not invented per-event flags:

1. `provenance.json` declares windows with `label` + `start` / `end` (RFC3339 / ISO-8601 UTC, as written by `lab/scripts/run_capture_session.sh`).
2. Each session’s midpoint event time is matched with `TimeWindow.contains` (`corpus/ingest/adapters/base.py`).
3. Event times are parsed from OTel JSONL via `corpus/ingest/timestamps.py`:
   - ProtoJSON **string** `startTimeUnixNano` / `timeUnixNano` (fixed64 → decimal string)
   - int/float unix **ns / µs / ms / s** (magnitude heuristic)
   - RFC3339 strings (`…Z` or offset)

**Requirement:** every scorable span/log timestamp must fall inside a normal or incident/cascade window. If timestamps are missing, mis-parsed, or outside all windows, sessions stay `label=unknown` and ranking fails with:

`need both normal and incident/cascade labeled sessions… Got label_counts={'unknown': N}`.

Practical checks:

- Prefer collector exports under `lab/captures/<id>/{traces,logs}.jsonl` (OTLP `resourceSpans` / `resourceLogs`).
- Window `start`/`end` must cover the same clock as OTel (UTC). Second-precision shell times are fine if load + flush complete before `end`.
- Do **not** invent wall-clock `now()` for missing event times — that drifts sessions out of windows.

Fixture `corpus/fixtures/lab_sample` uses RFC3339 `start_time` fields and is the smoke corpus. Real captures use ProtoJSON nanos; both paths must label correctly.

## Quick smoke (no torch / no train)

```bash
# From repo root
python -m eval.run_eval \
  --capture corpus/fixtures/lab_sample \
  --scores-from length \
  --out-dir /tmp/aomb-eval-seed0

python -m eval.run_multiseed \
  --seeds 0..2 \
  --capture corpus/fixtures/lab_sample \
  --scores-from length \
  --out-dir /tmp/aomb-eval-multi
```

Length / event baselines verify the harness wiring on fixtures. They are **not** a public accuracy claim.

## Model path

**Public ranking card (fixture-only):** trains on frozen train-split **normal** session texts via an in-memory dataloader — no CRISP / prepare shards:

```bash
python -m eval.run_eval \
  --capture corpus/fixtures/public_ranking_card_v1 \
  --scores-from model \
  --train-corpus fixture-train \
  --session-split eval \
  --train-seconds 45 \
  --seed 0 \
  --out-dir /tmp/aomb-fixture-model
```

**Lab / local prepare path** (not for the public card claim) still uses prepared tokenizer + train shards:

```bash
python -m eval.run_eval \
  --capture lab/captures/<id> \
  --scores-from model \
  --train-seconds 300 \
  --seed 0 \
  --out-dir reports/public-accuracy/seed-0
```

Full card runs: 3–5 seeds via `run_multiseed` / `run_public_ranking_card`. See protocol checklist before any public language beyond the fixture card.

## Session scorer (shippable, claim not published)

Per-session surprise/BPB without ranking metrics — for BYO dumps and diagnostics:

```bash
./scripts/byo_score.sh corpus/fixtures/lab_sample
uv run python -m eval.score_cli --input corpus/fixtures/lab_sample --dry-run
uv run python -m score_session --input path/to/dump --train-seconds 30 --out /tmp/score.json
```

See [`docs/byo-and-scorer.md`](../docs/byo-and-scorer.md). Scorer output is **not** a public accuracy claim.
