# Rule-proof faults, in-domain eval (2026-09-25): raw output

Raw output behind [`docs/lab/rule-proof-eval.md`](../../../docs/lab/rule-proof-eval.md).

```bash
python3 lab/scripts/pool_captures.py lab/captures/pooled-20260925-ruleproof \
  lab/captures/20260925v2-{silent_fallback,skip_cache,retry_storm,db_failover}
uv run python -m eval.in_domain --capture lab/captures/pooled-20260925-ruleproof \
  --seeds 0..4 --train-seconds 120 --out-dir reports/public-accuracy/lab-ruleproof-in-domain-20260925
uv run python -m eval.in_domain --capture lab/captures/pooled-20260925-ruleproof \
  --out-dir reports/public-accuracy/lab-ruleproof-in-domain-20260925 --subset-marker "op=GET_/api/checkout"
```

| Path | Contents |
|------|----------|
| `results.json` / `results.md` | Whole-window AUROC for every baseline and model variant, per seed and per fault |
| `subset-op_GET_api_checkout.json` | The same scores re-ranked on checkout sessions only |
| `sessions.json` | Every eval session's id, label, capture, and model scores per seed |
| `heatmap-seed0.html` | Per-token surprise for the most/least surprising and missed sessions |
