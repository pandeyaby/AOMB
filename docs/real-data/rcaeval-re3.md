# RCAEval RE3: faults designed by someone else

> **claim_status=`published`** (2026-09-27).
> On code-level faults from an external benchmark, the model is **roughly level with a trace-shape novelty check**. It's ahead on Train Ticket and behind on Online Boutique for the traces the fault touches. It needs no rules, but it doesn't beat a simple structural check.

## Why this eval matters

Every baseline in the lab evals was written by someone who knew what the faults were. [RCAEval](https://github.com/phamquiluan/RCAEval) (Pham et al., WWW'25 companion; [arXiv 2412.17015](https://arxiv.org/abs/2412.17015); MIT) was built by other researchers for root-cause analysis. Its RE3 suite has 90 **code-level** fault cases across three microservice benchmark systems. Nobody on this project chose these faults, so this is the closest thing here to a held-out test of the "breadth" claim.

## Data and protocol

| Item | Value |
|------|-------|
| Cases | RE3, Online Boutique (30) and Train Ticket (30). **Sock Shop's RE3 cases ship without traces**, so they're excluded |
| Per case | ~15 min of normal traffic, a code-level fault injected into one service at `inject_time`, then ~15 min more |
| Session | One trace, spans sorted by start time, in AOMB's span format (IDs, timestamps masked from scoring) |
| Split (per case, temporal) | Earlier half of pre-fault traces → training candidates; later half → eval normals; post-fault traces → eval incidents. 60 of each per case (seed 0) → 1,800 train, 1,800 normal, 1,800 incident traces per system |
| Model | One per system, trained on that system's 1,800 training traces (≤24-line chunks, none cropped); 120 s per seed. **Online Boutique 5 seeds; Train Ticket 3 seeds** (its traces run to 1,106 spans, making each seed ~25 min) |
| Subset | Traces that **touch the faulted service** (chosen by service, not label, for both classes). Most post-fault traces that don't touch it can't show the fault |

Rebuild (data not vendored):

```bash
# download re3* case folders (logs.parquet, traces.parquet, inject_time.txt) and cases.parquet
# from https://huggingface.co/datasets/phamquiluan/RCAEval into ~/.cache/aomb-datasets/rcaeval
uv run python -m corpus.ingest.rcaeval --input ~/.cache/aomb-datasets/rcaeval --suite re3 \
  --out-dir ~/.cache/aomb-datasets/rcaeval/sessions
# expected sessions_sha256: online-boutique 5ddeb31c…45cc7c · train-ticket a7956fd1…e2ae783
uv run python -m eval.in_domain --sessions ~/.cache/aomb-datasets/rcaeval/sessions/re3_<system>.jsonl \
  --seeds 0..4 --train-seconds 120 --train-chunk-lines 24 --out-dir <out>
```

## Results (AUROC)

### Traces that touch the faulted service

| Method | Online Boutique (297 traces, 142 incidents) | Train Ticket (1,585 traces, 973 incidents) |
|--------|----------------------------------------------|---------------------------------------------|
| Error/latency rule | 0.440 | 0.425 |
| Value checks (novelty / rarity / pair) | 0.500 / 0.201 / 0.500 | 0.500 / 0.500 / 0.500 |
| Trace-shape novelty | 0.947 | 0.828 |
| **Rule + shape novelty** | **0.952** | 0.882 |
| Model, masked session mean | 0.897 ± 0.035 | 0.788 ± 0.014 |
| Model, max per-event | 0.902 ± 0.061 | 0.875 ± 0.005 |
| Model, top-10% tokens | 0.928 ± 0.027 | 0.868 ± 0.005 |
| Model, per-field | 0.808 ± 0.054 | **0.887 ± 0.001** |

### All eval traces (AUROC / PR-AUC)

| Method | Online Boutique | Train Ticket |
|--------|-----------------|--------------|
| Trace-shape novelty | 0.542 / 0.542 | 0.731 / 0.733 |
| Rule + shape novelty | 0.507 / 0.564 | 0.712 / 0.791 |
| Model, max per-event | **0.636 ± 0.011** / **0.641 ± 0.041** | 0.669 ± 0.012 / 0.735 ± 0.017 |
| Model, top-10% tokens | 0.587 ± 0.007 / 0.561 ± 0.009 | **0.749 ± 0.014** / **0.806 ± 0.010** |
| Model, per-field | 0.586 ± 0.006 / 0.566 ± 0.010 | 0.715 ± 0.009 / 0.772 ± 0.006 |

## What it shows

1. **Level with a structural check, not better.** On the traces a fault can reach, trace-shape novelty (an unseen set of spans) beats the best model variant on Online Boutique (0.952 vs 0.928) and ties it on Train Ticket (0.882 vs 0.887).
2. **These faults are mostly structural.** Faulty traces are *shorter* (length scores 0.16–0.35, below chance): code-level faults cut calls short. That's the "something didn't happen" pattern, where a shape check is the natural tool and next-token surprise has to work harder.
3. **Across all traces the model does a little better**, but everything is weak there, because most post-fault traces never touch the faulted service.
4. **Error/latency rules and value checks are useless here** (at or below chance). These faults change structure, not error codes, latency or field values.

## Selection caveat

The best model variant differs by dataset and view (top-10% on Online Boutique, per-field on Train Ticket's subset, max-event on Online Boutique's whole set). Picking the best variant per column after seeing results flatters the model. With one score fixed in advance, the per-field score that won on the lab, the model **loses clearly on Online Boutique** (0.808 vs 0.952) and ties on Train Ticket (0.887 vs 0.882). Until a single scoring rule is chosen and held fixed on a new dataset, read "best variant" numbers as optimistic.

Reports: [`reports/public-accuracy/rcaeval-re3-online-boutique-20260927/`](../../reports/public-accuracy/rcaeval-re3-online-boutique-20260927/) · [`…-train-ticket-20260927/`](../../reports/public-accuracy/rcaeval-re3-train-ticket-20260927/).
