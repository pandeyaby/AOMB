# Published lab captures

The exact captures behind AOMB's labelled lab evals. They're real OpenTelemetry traces and logs from the Docker stack in [`lab/`](../) (frontend → API → Postgres + Redis), recorded during normal load and during injected faults. Labels come from the capture windows in each `provenance.json`.

| Capture | Faults | Used by |
|---------|--------|---------|
| `pooled-20260918` | latency, errors, latency+errors, Redis killed (5 runs) | [zero-shot](../../docs/lab/ranking-validation.md) · [in-domain](../../docs/lab/in-domain-eval.md) |
| `pooled-20260925-ruleproof` | silent_fallback, skip_cache, retry_storm, db_failover | [rule-proof faults](../../docs/lab/rule-proof-eval.md) |
| `pooled-20260926-valuedrift` | region_new, currency_swap, pricing_flip | [value drift](../../docs/lab/value-drift-eval.md) |

Each folder is byte-identical to what was evaluated. `tests/test_published_captures.py` checks the content hash against the reports.

Note that `pooled-20260918` was captured before a lab bug was fixed, so its faults were active on only about half of API requests. See [`ranking-validation.md`](../../docs/lab/ranking-validation.md#limitations).

```bash
./scripts/reproduce_lab_evals.sh            # full protocol (5 seeds × 120 s per pool)
QUICK=1 ./scripts/reproduce_lab_evals.sh    # 1 seed × 30 s per pool, minutes on CPU
```

The only network details inside are Docker-internal addresses (`172.25.x`, Docker Desktop's `192.168.65.1` gateway) and `localhost`. License: Apache-2.0, the same as the capture provenance.
