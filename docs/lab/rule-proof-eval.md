# Rule-proof faults: one model against hand-built detectors

> **claim_status=`published`** (2026-09-25; **corrected 2026-09-26**, see [Corrections](#corrections)).
> On checkout sessions, the model is the **best single detector** across four faults that evade error/latency rules (AUROC 0.944 ± 0.012). It isn't the best on any *individual* fault: for each one, a purpose-built check matches or beats it.

The [in-domain eval](in-domain-eval.md) showed that on error and latency faults, a 5-line rule beats the model. Those faults are what rules are built for. This eval asks the follow-up question: **on faults a rule can't see, does the model earn its keep?**

## The faults

Four modes in [`lab/services/api/app.py`](../../lab/services/api/app.py). Every request still returns 200 and nothing is logged at ERROR level:

| Fault | What changes | Latency |
|-------|--------------|---------|
| `silent_fallback` | A new WARN log: `pricing_fallback source=static_table ...` | unchanged |
| `skip_cache` | Checkout stops calling Redis, so the `cache.incr` span disappears | unchanged (slightly faster) |
| `retry_storm` | Every DB ping runs 3 times, so there are extra `db.ping` spans | checkout ~5 → ~11 ms |
| `db_failover` | The checkout log says `db=replica` instead of `db=ok` | unchanged |

**Lab bug found and fixed along the way.** Faults used to reach only one of the API's two gunicorn workers, because `docker-compose.yml` hardcoded `FAULT_MODE: none`. These captures (`20260925v2-*`) have the fault on every request: 40/40 fallbacks, 0/40 cache calls, 240/240 pings, 40/40 `db=replica`, and zero errors.

## Setup

Pooled capture: [`lab/published/pooled-20260925-ruleproof/`](../../lab/published/pooled-20260925-ruleproof/). A temporal split trains on the earlier half of each capture's normals (324 sessions); eval is 328 later normals plus 644 incidents. The model trains for 120 s × 5 seeds with an uncropped 512-token context. IDs, timestamps and monotonic counters are masked from scoring.

Baselines, all fitted on training normals only:
- **Rule:** error lines, then per-operation duration z-score. This is what an SRE alerts on.
- **Template/shape novelty:** an unseen operation, an unseen log template (every `=value` masked, Drain-style), or an unseen trace shape (the multiset of spans).
- **Rule + novelty:** both of the above combined.
- **Value novelty:** a categorical field value never seen in training.

## Results: checkout sessions (244 sessions, 160 incidents)

Every fault touches **checkout requests only**, so this re-ranks the checkout sessions (chosen by endpoint, not by label; `eval.in_domain --subset-marker "op=GET_/api/checkout"`).

| Fault | Rule | Template/shape novelty | Value novelty | **Model (masked mean)** | Model (max event) |
|-------|------|------------------------|---------------|------------------------|-------------------|
| `silent_fallback` (new log line) | 0.455 | **1.000** | **1.000** | **1.000 ± 0.000** | **1.000 ± 0.000** |
| `db_failover` (value changed) | 0.662 | 0.500 | **1.000** | 0.985 ± 0.013 | **1.000 ± 0.000** |
| `retry_storm` (extra calls) | **1.000** | **1.000** | 0.500 | 0.955 ± 0.035 | 0.958 ± 0.026 |
| `skip_cache` (missing call) | 0.217 | **1.000** | 0.500 | 0.855 ± 0.033 | 0.795 ± 0.064 |
| **All four pooled** | 0.601 | 0.875 | 0.750 | **0.944 ± 0.012** | 0.943 ± 0.019 |

Rule + novelty combined reaches 0.890 on the pooled set. Session length scores 0.750, and 1.000 on three faults, because checkout sessions in this lab are nearly identical, so any change in text length separates them. That's a sign the lab is too uniform, not a competitor. The model's clean result on `skip_cache`, where length scores 0.000, shows it isn't just measuring length.

### Whole fault window

Across whole windows, where only about a quarter of sessions are checkouts, the model (0.706 ± 0.015) edges out rule + novelty (0.680). Every method is compressed toward 0.5 by the untouched sessions.

## What this shows

1. **Breadth without prior knowledge.** Four different kinds of change (a new log line, a changed value, extra calls, a missing call) each need a different hand-built detector. The model catches all four, from 0.86 to 1.00, with no one telling it what to look for, and it's the best single method on the pooled set.
2. **No individual win.** For every fault there's a purpose-built check that matches or beats the model. Template/shape novelty plus value novelty together would cover all four. If you know what's coming, write the rule.
3. **It does see missing calls.** When the cache call disappears, the log line after the database ping arrives where the model expects a `cache.incr` span, and that transition is surprising (0.86). The earlier claim that surprise *can't* see a missing call was an artifact; see Corrections.

## Corrections

The 2026-09-25 version of this page made three claims that don't survive:
- *"The model is the only method that catches a pure value change."* **Wrong.** A simple value-novelty check (never-seen `db=replica`) catches it perfectly, and that baseline wasn't in the original comparison.
- *"A missing call is invisible to surprise."* **Wrong.** Training cropped sessions at 256 tokens, and the checkout log line starts around token 430, so the model had never seen the part of the session where the change shows. The earlier `db_failover` and `silent_fallback` "wins" were artifacts of the same bug.
- The earlier numbers also scored a monotonic Redis counter (`hits=`) that always looks new in later windows. It's now masked.

## Limitations

- A single small lab, with one capture per fault (40 incident checkouts each) and scripted faults.
- Checkout sessions are nearly uniform, which is why session length is so informative here. Real traffic varies far more.
- `./scripts/reproduce_lab_evals.sh` reruns this eval from the published capture.

Reports: [`reports/public-accuracy/lab-ruleproof-in-domain-20260925/`](../../reports/public-accuracy/lab-ruleproof-in-domain-20260925/).
