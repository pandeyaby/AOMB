# Rule-proof faults: where a telemetry language model helps, and where it doesn't

> **claim_status=`published`** (2026-09-25). The result is mixed, published as measured.
> The model is the **only** method here that catches a pure value change (`db=ok` → `db=replica`).
> It **cannot** see a missing call, and averaged over a whole fault window it loses to a hand-built heuristic.

The [in-domain eval](in-domain-eval.md) showed that on error and latency faults, a 5-line rule beats the model. Those faults are what rules are built for. This eval asks the follow-up question: **on faults a rule can't see, does the model earn its keep?**

## The faults

Four new modes in [`lab/services/api/app.py`](../../lab/services/api/app.py). Every request still returns 200 and nothing is logged at ERROR level:

| Fault | What changes | Latency |
|-------|--------------|---------|
| `silent_fallback` | A new WARN log: `pricing_fallback source=static_table ...` | unchanged |
| `skip_cache` | Checkout stops calling Redis, so the `cache.incr` span disappears | unchanged (slightly faster) |
| `retry_storm` | Every DB ping runs 3 times, so there are extra `db.ping` spans | checkout ~5 → ~11 ms |
| `db_failover` | The checkout log says `db=replica` instead of `db=ok` | unchanged |

**Lab bug found and fixed along the way.** Faults used to be switched on only in one of the API's two gunicorn workers, so about half of requests. `docker-compose.yml` hardcoded `FAULT_MODE: none`, and the runtime switch reached one worker. The compose file now reads the fault from the environment. These captures (`20260925v2-*`) have the fault on every request: 40/40 fallbacks, 0/40 cache calls, 240/240 pings, 40/40 `db=replica`, and zero errors. Earlier captures had the bug; see the limitation note in [`ranking-validation.md`](ranking-validation.md#limitations).

## Setup

Same as the [in-domain eval](in-domain-eval.md). Pooled capture: `lab/captures/pooled-20260925-ruleproof` (built with [`lab/scripts/pool_captures.py`](../../lab/scripts/pool_captures.py)). A temporal split trains on the earlier half of each capture's normals (324 sessions); eval is 328 later normals plus 644 incidents. The model trains for 120 s × 5 seeds, and IDs and timestamps are masked from scoring.

Two baselines, both built only from training normals:
- **Rule:** error lines, then per-operation duration z-score. This is what an SRE alerts on.
- **Novelty (heuristic):** an unseen operation, an unseen log template (every `=value` masked, Drain-style), or an unseen trace shape (the multiset of spans). This is the strongest detector a sharp SRE could hand-build.

## Results

### Whole fault window (AUROC)

| Fault | Rule | Novelty | Rule + novelty | Model (mean BPB) | Model (max event) |
|-------|------|---------|----------------|------------------|-------------------|
| `retry_storm` | **0.996** | 0.752 | **1.000** | 0.667 | 0.716 |
| `silent_fallback` | 0.479 | **0.627** | 0.593 | 0.607 | 0.540 |
| `db_failover` | 0.687 | 0.503 | **0.693** | 0.578 | 0.542 |
| `skip_cache` | 0.255 | **0.627** | 0.458 | 0.476 | 0.495 |
| **Pooled** | 0.608 | 0.627 | **0.680** | 0.583 ± 0.008 | 0.574 ± 0.008 |

Every fault here touches **checkout requests only**, and checkout sessions are about a quarter of each window. Catalog requests and frontend-only traces look normal whatever the fault. So across the whole window, even a perfect checkout detector scores only about **0.63**, and all these numbers are compressed toward 0.5.

### Checkout sessions only (where the fault can be seen)

Here the same scores are re-ranked on checkout sessions only (244 sessions, 160 incidents). The subset is chosen **by endpoint, not by label**, so it's fair to every method (`eval.in_domain --subset-marker "op=GET_/api/checkout"`).

| Fault | Rule | Novelty | Length | Model (mean BPB) | Model (max event) |
|-------|------|---------|--------|------------------|-------------------|
| `silent_fallback` (new log line) | 0.455 | **1.000** | 1.000* | **1.000** | **1.000** |
| `db_failover` (value changed) | 0.662 | 0.500 | 1.000* | 0.679 ± 0.028 | **1.000 ± 0.000** |
| `retry_storm` (extra calls) | **1.000** | **1.000** | 1.000* | 0.834 ± 0.032 | 0.470 ± 0.187 |
| `skip_cache` (missing call) | 0.217 | **1.000** | 0.000* | 0.257 ± 0.053 | 0.372 ± 0.106 |

\* Checkout sessions in this lab are nearly identical, so *any* change in text length separates the classes perfectly. For example, `replica` is 5 characters longer than `ok`. That's an artifact of a tiny, uniform lab and wouldn't survive real traffic, so treat the length column as a warning about the lab, not as a competitor.

## What this shows

1. **Value drift is the model's unique win.** When only a value changes (`db=replica`) and the log template is the same, rules and template novelty are blind (0.66 / 0.50). The model's most-surprising-event score separates it perfectly (1.000 across 5 seeds). This is the first result where the model catches something the hand-built detectors miss.
2. **New log lines: the model ties the novelty detector** (both 1.000 on checkout sessions). That's useful but not unique; template mining already does it.
3. **A missing call is invisible to surprise.** When the cache call disappears, the session becomes *more* predictable, so the score inverts (0.26–0.37). Next-token surprise measures what happened that shouldn't have; it can't flag what should have happened and didn't. Trace-shape novelty catches it perfectly.
4. **Retry storms go to the rule,** because extra calls add latency. The model's mean score sees them (0.83), but its max-event score doesn't: the extra `db.ping` lines are each individually normal.
5. **The scoring method matters as much as the model.** Max-event scoring wins on value drift and loses on retry storms; mean scoring does the reverse. No single score fits every fault type.
6. **Across whole windows the model still loses** (0.58 vs 0.68 for rule + novelty). Adding the model to rule + novelty by rank-sum helps on value drift (checkout: 0.66 → 0.93) but costs on the missing-call fault, so it's no better overall (0.63). A smarter combination was *not* tuned on these results; that would be overfitting to 4 faults.

## What it means

The honest positioning, supported by this data, is that **a telemetry language model complements rules and template or shape novelty; it doesn't replace them.** Its unique value is catching changes *within* a known template: values, parameters, content. That's the one thing log-template and trace-shape tools throw away by design.

What to build on this:
- **Score per event, not per session.** The model's clean wins are all at event level. A session or window average dilutes a single anomalous line among dozens of normal ones.
- **Pair it with a trace-shape check** for missing or extra calls, which surprise can't express.
- **Test value drift at scale.** One synthetic `db=replica` switch in a uniform lab is a promising signal, not proof. Next: several value-drift faults of varied length (to kill the length artifact), on traffic with realistic variety.

## Limitations

- A single small lab, one capture per fault (161 incident sessions each), and scripted faults.
- Checkout-only faults compress whole-window AUROC. The checkout subset is the fairer view but has only 40 incident sessions per fault.
- The length baseline's perfect scores show this lab is too uniform. Real traffic varies far more in session length and content.
- The capture isn't in the repo yet (it's local under `lab/captures/`, gitignored); per-session scores and content hashes are.

Reports: [`reports/public-accuracy/lab-ruleproof-in-domain-20260925/`](../../reports/public-accuracy/lab-ruleproof-in-domain-20260925/). These include `results.json`, `subset-op_GET_api_checkout.json`, per-session scores, and a per-token heatmap.
