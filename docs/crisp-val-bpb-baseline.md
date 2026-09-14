# CRISP val_bpb baseline — Reference corpus v1

Recorded `val_bpb` on a **CRISP subset** for README / internal factual documentation.

> **Not a public accuracy claim, marketing number, or product benchmark.**
> Honest single-run baseline on a capped CRISP extract — nothing more.
> For the frozen public accuracy ranking protocol (claim not published until checklist passes), see [`docs/public-accuracy-eval.md`](public-accuracy-eval.md).

> **Do not 1:1 compare** this number to the synthetic / smoke-era best of **0.3682**.
> Different data, tokenizer, and scale. See README: *Empirical Results — Synthetic / smoke-era (legacy)*.

---

## Result (exact)

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

## Reproduce

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
