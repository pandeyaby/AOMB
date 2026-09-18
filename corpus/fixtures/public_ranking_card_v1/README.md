# Public ranking card v1 — fixture pack

**Card id:** `public_ranking_card_v1`  
**claim_status:** see [`reports/public-ranking-card-v1/CARD.md`](../../../reports/public-ranking-card-v1/CARD.md)  
**License:** Apache-2.0 (synthetic content authored for this repo)

## What this is

A small, **synthetic** labeled session pack for the frozen public ranking card protocol
([`docs/public-ranking-card-v1.md`](../../../docs/public-ranking-card-v1.md)).

| Class | Count | Window labels |
|-------|------:|---------------|
| Normal | 8 | `normal` |
| Incident / anomaly | 5 | `incident` |
| Cascade | 3 | `cascade` |
| **Scorable total** | **16** | binary: normal=0, incident∪cascade=1 |

Frozen split (`split.json`): **10 train / 6 eval**. Eval always contains both classes.
LM training uses **train-split normals only** (fixture texts; no CRISP).

Lengths intentionally **overlap** across classes so length / event-count baselines are
non-trivial on the eval split.

## Provenance

| Field | Value |
|-------|-------|
| `source_kind` | `synthetic_lab_fixture` |
| `source_id` | `aomb-public-ranking-card-v1` |
| `capture_id` | `public_ranking_card_v1` |
| Customer data | **None** — fully synthetic / stylized |
| Private lab pool | **Not included** — do not confuse with `docs/lab/` |
| CRISP / Tale of Errors | **Not used** for this card’s train or eval |

## What this is not

- **Not** private lab-pool AUROC (including 0.766)
- **Not** a CRISP `val_bpb` number
- **Not** production / customer telemetry
- **Not** a production support / SLO metric

## Layout

```
public_ranking_card_v1/
  README.md           # this file
  provenance.json     # windows + labels + card metadata
  split.json          # frozen train/eval session ids
  traces.jsonl        # synthetic spans (flat JSONL)
  logs.jsonl          # synthetic logs
```

## Reproduce

```bash
./scripts/run_public_ranking_card_v1.sh --with-model --check-eps
```

Reports land under `reports/public-ranking-card-v1/`.
