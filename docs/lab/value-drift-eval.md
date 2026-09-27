# Value drift: one model against three hand-built value checks

> **claim_status=`published`** (2026-09-26).
> The model catches a **never-seen value** (0.99), only partly catches a **wrong pairing** of familiar values (0.65), and **misses a frequency shift** (0.55). Each simple value check beats it on the fault it was built for, and the model is *not* the best single detector here.

## Why this eval exists

The [rule-proof eval](rule-proof-eval.md) had one value-drift fault (`db_failover`), and a simple "never-seen value" check caught it perfectly. This eval asks whether the model does better on subtler drift *inside* a known log line, where:
- lengths can't give it away (every drifted value has the same length as the normal ones)
- simple value checks are included as baselines

## The faults

The lab's checkout log carries three fields with realistic normal variation. Region is `us-east` or `eu-west` (50/50), currency **matches the region** (`USD` / `EUR`), and pricing is `v1` 90% of the time, `v2` 10%.

| Fault | What changes | Same length? |
|-------|--------------|--------------|
| `region_new` | `region=ap-east`, a value never seen in normal traffic | yes |
| `currency_swap` | Currency mismatched with region (`us-east` + `EUR`, `eu-west` + `USD`). **Every individual value is familiar; only the pairing is wrong** | yes |
| `pricing_flip` | `pricing=v2` on 90% of requests. A familiar value at the wrong frequency | yes |

Captures: 120 requests per window, faults verified on 120/120 checkouts, zero errors. Published at [`lab/published/pooled-20260926-valuedrift/`](../../lab/published/pooled-20260926-valuedrift/).

## Baselines

Everything from the [rule-proof eval](rule-proof-eval.md), plus three value checks fitted on training normals. They look only at categorical fields; numeric and high-cardinality fields (counters, durations, IDs) are excluded automatically.
- **Value novelty:** a field value never seen in training.
- **Value rarity:** per-line naive-Bayes surprisal, the sum of −log₂ p(value | field) over a line's fields, taking the worst line.
- **Value-pair novelty:** two field values seen together on one line for the first time. You'd only write this check if you'd anticipated a pairing fault.

## Results: checkout sessions (543 sessions, 360 incidents; 5 seeds)

| Fault | Rule | Value novelty | Value rarity | Value pair | Model (masked mean) | Model (max event) |
|-------|------|---------------|--------------|------------|---------------------|-------------------|
| `region_new` | 0.685 | **1.000** | **1.000** | **1.000** | 0.942 ± 0.033 | 0.991 ± 0.011 |
| `currency_swap` | 0.474 | 0.500 | 0.493 | **1.000** | 0.648 ± 0.080 | 0.557 ± 0.067 |
| `pricing_flip` | 0.528 | 0.500 | **0.886** | 0.500 | 0.551 ± 0.051 | 0.519 ± 0.032 |
| **All three pooled** | 0.565 | 0.667 | 0.792 | **0.833** | 0.719 ± 0.045 | 0.695 ± 0.024 |

For `pricing_flip`, about **0.89 is the most any per-session method can score**: 90% of incident checkouts are `v2`, against 12% of normal ones. The rarity check is at that bound. Session length is at chance (0.500) on every fault, so there's no length artifact here.

Across whole fault windows, only about a quarter of sessions are checkouts, and every method is near chance (model 0.564–0.569; best baseline, value rarity, 0.586).

## What it shows

1. **A never-seen value is easy, for everyone.** The model catches `ap-east` (0.99), and so does the simplest value check.
2. **The model learned the region→currency dependency, but session scores drown it.** Per-token surprise on normal traffic is about 0.7 bits for `currency`, because currency follows from region; a swapped currency costs about 6.6 bits. That's a strong signal at the token, but averaged into a whole session it yields only 0.65 AUROC. The pair check, which you'd only write if you anticipated this fault, gets 1.000.
3. **A frequency shift is invisible per session.** A single `v2` checkout looks the same in a normal window and in a fault window. Only its rate changes, and the model's session score doesn't measure rates. The rarity check does better (0.89) because it's explicitly scoring how rare each value is.

## What to build next

- **Score per field, not per session.** The model's token-level signal on `currency_swap` is strong; the session-level aggregation loses it. Scoring the most surprising *field value* (not the average of a line or session) is the obvious next experiment.
- **Score windows, not just sessions, for frequency drift.** A shift in the *rate* of a familiar value needs a window-level statistic: compare the model's expected distribution with what's observed.

## Correction note

An interim version of these numbers, shared in conversation before publication, showed the model as the best single detector here (0.857). That run scored a monotonic Redis counter (`hits=`), whose values in later fault windows are always new. With the counter masked, the result is the one above.

## Reproduce

```bash
QUICK=1 ./scripts/reproduce_lab_evals.sh    # minutes; or without QUICK for the full protocol
```

Reports: [`reports/public-accuracy/lab-valuedrift-in-domain-20260926/`](../../reports/public-accuracy/lab-valuedrift-in-domain-20260926/).
