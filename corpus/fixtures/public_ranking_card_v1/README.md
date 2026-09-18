# Public ranking card v1 — fixture pack

**Card id:** `public_ranking_card_v1`  
**claim_status:** `published_fixture_card` (harness smoke — see CARD.md)  
**License:** Apache-2.0 (synthetic content authored for this repo)

## Limitations (loud)

- **Synthetic** stylized sessions — not customer / production telemetry.
- Eval set size is **n=24** held-out labeled sessions (48 total with frozen train/eval split).
- **High / perfect AUROC on this pack is toy separation / harness smoke**, not general public accuracy and not production AUROC.
- **Not** private lab-pool AUROC (incl. 0.766). **Not** CRISP `val_bpb`. **Not** a support/SLO metric.

## What this is

A **synthetic** labeled session pack for the frozen public ranking card protocol
([`docs/public-ranking-card-v1.md`](../../../docs/public-ranking-card-v1.md)).

| Class | Count | Window labels |
|-------|------:|---------------|
| Normal | 24 | `normal` |
| Incident / anomaly | 16 | `incident` |
| Cascade | 8 | `cascade` |
| **Scorable total** | **48** | binary: normal=0, incident∪cascade=1 |

Frozen split (`split.json`): **24 train / 24 eval**. Eval always contains both classes.
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

Reports: [`reports/public-ranking-card-v1/CARD.md`](../../../reports/public-ranking-card-v1/CARD.md).
