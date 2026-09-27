# Value-drift faults, in-domain eval (2026-09-26): raw output

Raw output behind [`docs/lab/value-drift-eval.md`](../../../docs/lab/value-drift-eval.md).

```bash
uv run python -m eval.in_domain --capture lab/published/pooled-20260926-valuedrift \
  --seeds 0..4 --train-seconds 120 --out-dir reports/public-accuracy/lab-valuedrift-in-domain-20260926
uv run python -m eval.in_domain --capture lab/published/pooled-20260926-valuedrift \
  --out-dir reports/public-accuracy/lab-valuedrift-in-domain-20260926 --subset-marker "op=GET_/api/checkout"
```

| Path | Contents |
|------|----------|
| `results.json` / `results.md` | Whole-window AUROC for every baseline and model variant, per seed and per fault |
| `subset-op_GET_api_checkout.json` | The same scores re-ranked on checkout sessions only |
| `sessions.json` | Every eval session's id, label, capture, and model scores per seed |
| `heatmap-seed0.html` | Per-token surprise (IDs, timestamps and counters faded: not scored) |
