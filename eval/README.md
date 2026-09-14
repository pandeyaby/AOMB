# Public accuracy eval harness

Scaffolding for the protocol in [`docs/public-accuracy-eval.md`](../docs/public-accuracy-eval.md).

**Claim status:** not published. Reports always say so. Do not invent AUROC numbers in docs.

## What this does

1. Load lab_capture sessions + `provenance.json` window labels (`normal` vs `incident`/`cascade`)
2. Score sessions (length baseline, event-count baseline, precomputed scores, or optional train-then-score session BPB)
3. Emit **JSON + markdown** with AUROC, PR-AUC, precision@k, random ranking baseline, seed, corpus id/hash, `train.py` / `prepare.py` SHAs
4. Multi-seed runner → mean±std aggregate

`prepare.evaluate_bpb` is **not** called or modified. Session BPB scoring (model path) uses the same CE→bits/byte idea as `demo_anomaly.session_bpb`.

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

## Model path (optional)

Requires prepared tokenizer + train shards and torch (same prerequisites as `demo_anomaly.py`):

```bash
python -m eval.run_eval \
  --capture lab/captures/<id> \
  --scores-from model \
  --train-seconds 300 \
  --seed 0 \
  --out-dir reports/public-accuracy/seed-0
```

Full claim runs: 3–5 seeds via `run_multiseed` with the same budget. See protocol checklist before any public language.

## Session scorer (shippable, claim not published)

Per-session surprise/BPB without ranking metrics — for BYO dumps and diagnostics:

```bash
uv run python -m eval.score_cli --input corpus/fixtures/lab_sample --dry-run
uv run python -m score_session --input path/to/dump --train-seconds 30 --out /tmp/score.json
```

See [`docs/byo-and-scorer.md`](../docs/byo-and-scorer.md). Scorer output is **not** a public accuracy claim.
