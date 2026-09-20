# Tale val_bpb baseline — capped public-real subset (honest)

Recorded `val_bpb` on a **capped Tale of Errors subset** for README / internal factual documentation — **when a measured Mac run is attached**.

> **Not a public accuracy claim, marketing number, or product benchmark.**
> Factual documentation only (`claim_status=not_published`). Claim language stays gated until the checklist in [`docs/public-accuracy-eval.md`](public-accuracy-eval.md) passes.

> **Loud honesty**
>
> - **Train lane only.** CRISP / Tale = public-real Jaeger → shards → `prepare.py` → factual **`val_bpb`**.
> - Tale dumps have **no AOMB incident / normal window labels** → **no AUROC** from this path.
> - Lab ranking evidence stays under [`docs/lab/`](lab/) — default **`claim_status=not_published`**. Do not invent AUROC.
> - **No CUDA invent.** Product overnight / MPS train: [`compute-paths.md`](compute-paths.md) · [`product-mac-path.md`](product-mac-path.md).
> - `prepare.py` is **sacred** — invoke it; never edit it for this path.
> - Do **not** 1:1 compare Tale numbers to CRISP ([`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md)) or synthetic / smoke-era **0.3682**. Different source, tokenizer, and scale.
> - **Full Tale decompress is out of scope** for this baseline (hundreds of GB per archive). Maintainer Mac disk ~**315 Gi free** → **capped subset only** (one/few Zenodo pieces + explicit `--max-spans`).

Bootstrap (smaller) public-real path: Uber **CRISP** — [`corpus-v1.md`](corpus-v1.md) · [`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md).  
Full Tale-scale outsider path: [`tale-scale.md`](tale-scale.md). Fixture wiring smoke (no Zenodo): `./scripts/tale_scale_smoke.sh`.

---

## Measured row (Mac capped 200k)

**Source of truth:** [`reports/tale-capped/measured_capped_200k.json`](../reports/tale-capped/measured_capped_200k.json) — emitted from the real Mac train log via `eval.tale_measured_report` (parse-only `val_bpb`; never hardcoded).

| Field | Value |
|-------|--------|
| **val_bpb** | **1.379520** (parsed from train log footer) |
| Spans / sessions | `--max-spans 200000` · **158** sessions (143 train + 15 val) |
| Hardware | `mac_mps` (Apple Silicon / Metal) |
| Config | Single-run `TIME_BUDGET=300` train |
| vocab_size | 2115 |
| Subset | Capped public-real Tale |
| Train tip | `7083fc1` (main tip at emit: `c9673fb`) |
| Overnight `agent_loop` | **No** |
| API keys used | **No** |
| claim_status | **`measured_not_published`** — factual training metric only; **not** AUROC / public ranking; **not** 1:1 vs CRISP |

Honesty unchanged: train lane only; Tale has no incident labels → no AUROC; do not invent; do not blend with CRISP baselines.

---

## Disk reality (why capped)

| Constraint | Honest bound |
|------------|--------------|
| Full Tale compressed | ~35 GB + ~37 GB (two Zenodo parts) |
| Full decompress | **300–500 GB per archive** after `zstd` |
| Maintainer Mac free (approx.) | ~**315 Gi** → full decompress **unsafe / out of scope** |
| This baseline | Selective fetch (one/few keys) **or** existing local Jaeger tree + **`--max-spans`** |

Do **not** run `--download-all` or full `cat trace*_ *` → `zstd -d` on a disk that cannot hold hundreds of GB free. Prefer CRISP if you only need a recorded public-real `val_bpb` today.

---

## Protocol — capped public-real train fitness

Thin wrapper: `./scripts/tale_capped_baseline.sh` (refuses `--auroc` / `--download-all`; requires `--max-spans` + existing `--input`, or documents a **single** Zenodo `--fetch-key`).

### 1. Selective fetch (optional; one key)

```bash
# List keys (network; no download)
uv run python -m corpus.ingest.fetch_tale_of_errors --list-only

# One piece only (example). Resume-safe. Not full corpus.
uv run python -m corpus.ingest.fetch_tale_of_errors \
  --download trace1_aa \
  --out ~/.cache/autoresearch/corpus-v1/tale_of_errors

# Same via wrapper (still one key; no --download-all):
./scripts/tale_capped_baseline.sh --fetch-key trace1_aa
```

A single split piece is **not** a ready Jaeger tree. Full part reassembly + decompress remains **out of scope** under the disk cap above. Use an **already-local** assembled / extracted Jaeger JSON tree that fits, or stop after fetch documentation.

### 2. Shard with explicit `--max-spans`

```bash
./scripts/tale_capped_baseline.sh \
  --input /path/to/local/jaeger/tree \
  --max-spans 200000 \
  --num-train-shards 8

# equivalent:
uv run python -m corpus.ingest.build_shards \
  --adapter tale_of_errors \
  --input /path/to/local/jaeger/tree \
  --max-spans 200000 \
  --num-train-shards 8 \
  --write-val-shard
```

Pinned val shard index remains **6542**. Schema: one `text` column per session (`traceID`) — [`corpus-v1.md`](corpus-v1.md).

### 3. Prepare (sacred)

```bash
uv run python prepare.py --num-shards 8
```

`prepare.py` expects macOS + Metal (same gate as the product Mac path). On Linux / CI, stop after the capped shard step.

### 4. `TIME_BUDGET` train — record factual `val_bpb` only

```bash
uv run python train.py
```

Record whatever `val_bpb:` the run prints into the measured row above. **Do not** turn it into AUROC / ranking accuracy. **Do not** blend with CRISP-500k **0.407753**, overnight CRISP 200k **0.4309**, or synthetic **0.3682**.

---

## Corpus provenance (when filled)

| Field | Value |
|-------|--------|
| Source | *The Tale of Errors in Microservices* — Zenodo [13947828](https://doi.org/10.5281/zenodo.13947828) + [13952897](https://doi.org/10.5281/zenodo.13952897), **CC BY 4.0** |
| Paper | https://doi.org/10.1145/3700436 |
| Subset | Local capped tree + `--max-spans` (record N when measured) |
| Adapter | `tale_of_errors` (`source_id=uber-tale-of-errors`) |
| Windows | **normal only** (Tale dump has no AOMB incident labels) |
| Sanitization | **Do not mix** mapping with CRISP (Zenodo [13956078](https://doi.org/10.5281/zenodo.13956078)) |
| claim_status | Factual training metric only — **not** a public accuracy claim |

**Attribution (required under CC BY):** Lee, Zhang, Parwal, Chabbi — *The Tale of Errors in Microservices*, SIGMETRICS 2025 — https://doi.org/10.1145/3700436; artifacts https://doi.org/10.5281/zenodo.13947828 and https://doi.org/10.5281/zenodo.13952897 (CC BY 4.0).

---

## What this is / is not

| Artifact | Role |
|----------|------|
| This doc + `tale_capped_baseline.sh` | Protocol for a **capped** Tale train-fitness `val_bpb` on a disk-safe subset |
| Measured `val_bpb` row | **1.379520** on capped 200k — see `reports/tale-capped/measured_capped_200k.json` (not AUROC; not CRISP) |
| Full Tale decompress / `--download-all` | **Out of scope** here (~315 Gi free ≠ hundreds of GB per archive) |
| CRISP `val_bpb` baselines | Separate bootstrap lane — [`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md) |
| Lab AUROC / ranking | **`not_published`** until [`lab/publish-checklist.md`](lab/publish-checklist.md) |
| Fixture `tale_scale_smoke.sh` | Adapter wiring only — not a public-real baseline number |

---

## Related

- Tale-scale outsider path: [`tale-scale.md`](tale-scale.md) · smoke: `./scripts/tale_scale_smoke.sh`
- Corpus map: [`corpus-v1.md`](corpus-v1.md) · tree: [`corpus/README.md`](../corpus/README.md)
- CRISP train fitness: [`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md)
- Public wins / stranger path: [`public-wins.md`](public-wins.md)
- Accuracy claim gate (not this path): [`public-accuracy-eval.md`](public-accuracy-eval.md)
