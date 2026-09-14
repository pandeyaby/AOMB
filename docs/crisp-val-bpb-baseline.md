# CRISP val_bpb baseline — Reference corpus v1

Recorded `val_bpb` on a **CRISP subset** for README / internal factual documentation.

> **Not a public accuracy claim, marketing number, or product benchmark.**
> Factual documentation only. Claim language stays gated until the checklist in [`docs/public-accuracy-eval.md`](public-accuracy-eval.md) passes ([PR #6](https://github.com/pandeyaby/AOMB/pull/6)).

> **Do not 1:1 compare** CRISP numbers to the synthetic / smoke-era best of **0.3682**.
> Different data, tokenizer, and scale. See README: *Empirical Results — Synthetic / smoke-era (legacy)*.

---

## Overnight CRISP breeding (current CRISP best)

| Field | Value |
|-------|--------|
| **val_bpb** | **0.4309** |
| Commit / experiment | `73b1645` / exp 20 |
| Hardware | Mac Apple Silicon (MPS) |
| Experiments | 20 (overnight `agent_loop`) |
| Improve chain | 0.4554 → 0.4525 → 0.4396 → 0.4309 |

This is the current CRISP-lane overnight best. Synthetic **0.3682** remains separate / non-CRISP.

---

## Pre-overnight floor (TIME_BUDGET single run)

Prior factual baseline before overnight breeding:

| Field | Value |
|-------|--------|
| **val_bpb** | **0.458756** |
| Date | 2026-09-14 |
| Hardware | MacBook Pro Apple Silicon (MPS) |
| `training_seconds` | 300.1 (`TIME_BUDGET=300` from `prepare.py`) |
| `total_seconds` | 401.4 (includes eval) |
| `num_steps` | 603 |
| `total_tokens_M` | 19.8 |
| `num_params_M` | 8.5 |
| depth | 4 |
| `window_pattern` | SSL |
| `vocab_size` | 5206 |
| Overnight `agent_loop` | **No** |
| API keys used | **No** |

---

## Corpus provenance

| Field | Value |
|-------|--------|
| Source | Uber CRISP — Zenodo [13956078](https://doi.org/10.5281/zenodo.13956078), **CC BY 4.0** |
| Subset | `CRISP-main/data/bottom-up-trace` with `--max-spans 200000` |
| Sessions | 2185 (1967 train / 218 val) |
| Spans | 200000 |
| Jaeger JSON files | 2185 |
| Shards | 20 train shards + val `shard_06542` |
| Provenance file id | `crisp_zenodo_20260914T050751Z.json` |
| Windows | **normal only** (CRISP dump has no incident labels) |

**Citation (required under CC BY):** Zhang et al., CRISP, USENIX ATC'22 — https://doi.org/10.5281/zenodo.13956078

```bibtex
@inproceedings{zhang2022crisp,
  title={{CRISP}: Critical path analysis of {Large-Scale} microservice architectures},
  author={Zhang, Zhizhou and Ramanathan, Murali Krishna and Raj, Prithvi
          and Parwal, Abhishek and Sherwood, Timothy and Chabbi, Milind},
  booktitle={2022 USENIX Annual Technical Conference (USENIX ATC 22)},
  pages={655--672},
  year={2022}
}
```

---

## Reproduce the pre-overnight floor

```bash
# 1. Fetch CRISP (~2.33 GB) — not for CI
uv run python -m corpus.ingest.fetch_crisp --download

# 2. Build shards from bottom-up-trace, capped at 200k spans, 20 train shards
uv run python -m corpus.ingest.build_shards \
  --adapter crisp_zenodo \
  --input ~/.cache/autoresearch/corpus-v1/crisp/extracted \
  --max-spans 200000 \
  --num-train-shards 20 \
  --write-val-shard
# Expected subset path: CRISP-main/data/bottom-up-trace

# 3. Tokenizer + data prep (sacred pipeline — do not change objective)
uv run python prepare.py --num-shards 20

# 4. Train (default TIME_BUDGET=300)
uv run python train.py
```

`prepare.py` / `train.py` training objective are unchanged for this baseline documentation.
