# Public ranking card v1 — fixture pack

**Card id:** `public_ranking_card_v1`  
**claim_status:** `not_published`  
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

Sessions are trace-grouped spans + logs under two (plus cascade) provenance windows.
Lengths intentionally **overlap** across classes so length / event-count baselines are
non-trivial (not perfect AUROC by construction).

## Provenance

| Field | Value |
|-------|-------|
| `source_kind` | `synthetic_lab_fixture` |
| `source_id` | `aomb-public-ranking-card-v1` |
| `capture_id` | `public_ranking_card_v1` |
| Customer data | **None** — fully synthetic / stylized |
| Derived from | Pattern of `corpus/fixtures/lab_sample` (flat JSONL + `provenance.json`) |
| Private lab pool | **Not included** — do not confuse with `docs/lab/` private captures |
| CRISP / Tale of Errors | **Not used** |

Timestamps are RFC3339 UTC in a fixed 2026-09-11 demo clock. Fault names
(`api_latency`, `kill_redis`) mirror lab inject scripts for realism only.

## What this is not

- **Not** a public accuracy claim (checklist + Abhinav greenlight still required)
- **Not** the private lab pooled validation path (lab-pool AUROC must never appear on this card)
- **Not** a CRISP `val_bpb` number
- **Not** production / customer telemetry

## Layout

```
public_ranking_card_v1/
  README.md           # this file
  provenance.json     # windows + labels + card metadata
  traces.jsonl        # synthetic spans (flat JSONL)
  logs.jsonl          # synthetic logs
```

## Reproduce baselines

```bash
./scripts/run_public_ranking_card_v1.sh
# or
python -m eval.run_public_ranking_card --baselines-only
```

Reports land under `reports/public-ranking-card-v1/`.
