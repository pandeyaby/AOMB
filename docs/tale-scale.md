# Tale-scale corpus path — public train lane (honest)

**Audience:** outsiders with disk + network who want the **flagship-scale** Uber Tale of Errors train path — not a ranking claim.

> **Loud honesty**
>
> - **CRISP** and **Tale of Errors** are the **train lane** (public-real Jaeger → shards → `prepare.py` → factual **`val_bpb`**).
> - Tale traces are **sanitized production** dumps. They do **not** ship AOMB incident / normal window labels → **no AUROC** from this path.
> - Lab ranking evidence stays under [`docs/lab/`](lab/) — default **`claim_status=not_published`**. Do not invent AUROC.
> - Do **not** mix Tale sanitization mapping with CRISP (Zenodo note / [13956078](https://doi.org/10.5281/zenodo.13956078)).
> - `prepare.py` is **sacred** — invoke it; never edit it for this path.
> - **No CUDA claim.** Product overnight / MPS train honesty: [`compute-paths.md`](compute-paths.md) · [`product-mac-path.md`](product-mac-path.md).

Bootstrap (smaller) public-real path: Uber **CRISP** — [`corpus-v1.md`](corpus-v1.md) · factual CRISP `val_bpb`: [`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md).  
One-command fixture smoke (no Zenodo): `./scripts/tale_scale_smoke.sh`.

---

## Provenance (CC BY 4.0)

| Field | Value |
|-------|--------|
| **Name** | *The Tale of Errors in Microservices* (SIGMETRICS 2025) — Lee, Zhang, Parwal, Chabbi |
| **Part 1** | [doi:10.5281/zenodo.13947828](https://doi.org/10.5281/zenodo.13947828) |
| **Part 2** | [doi:10.5281/zenodo.13952897](https://doi.org/10.5281/zenodo.13952897) |
| **Paper** | https://doi.org/10.1145/3700436 |
| **License** | **CC BY 4.0** — attribution required |
| **Scale** | ~1.4M sanitized production Jaeger traces |
| **Disk** | ~35 GB + ~37 GB compressed; **300–500 GB decompressed per archive** after `zstd` |
| **CI / this PR** | Full download **must not** run in CI. Fixture sample only lives in-repo. |

**Attribution (required under CC BY):** Lee, Zhang, Parwal, Chabbi — *The Tale of Errors in Microservices*, SIGMETRICS 2025 — artifacts https://doi.org/10.5281/zenodo.13947828 and https://doi.org/10.5281/zenodo.13952897 (CC BY 4.0).

Adapter: `corpus/ingest/adapters/tale_of_errors.py` (`source_id=uber-tale-of-errors`).  
Fetch helper: `python -m corpus.ingest.fetch_tale_of_errors` (list / selective download with resume; refuses full pull in CI).

---

## One-command smoke (no multi-GB download)

Uses the tiny in-repo fixture only — proves adapter → shards wiring. **Does not** invent `val_bpb` / AUROC.

```bash
./scripts/tale_scale_smoke.sh
# equivalent bounded shard build:
uv run python -m corpus.ingest.build_shards \
  --adapter tale_of_errors \
  --input corpus/fixtures/tale_of_errors_sample \
  --max-spans 100 \
  --num-train-shards 1 --write-val-shard \
  --data-dir /tmp/aomb-tale-smoke
```

---

## Scale path — fetch → assemble → shard → prepare → train fitness

Do this only on a machine with **hundreds of GB free**. Not for CI / Codespaces default disks.

### 1. Fetch (selective; resume-safe)

```bash
# Instructions only (exit 2) — no download
uv run python -m corpus.ingest.fetch_tale_of_errors

# List Zenodo files via API (needs network; no bulk download)
uv run python -m corpus.ingest.fetch_tale_of_errors --list-only

# Pull one split piece (example). Repeat for all trace1_* / trace2_* keys you need.
uv run python -m corpus.ingest.fetch_tale_of_errors \
  --download trace1_aa \
  --out ~/.cache/autoresearch/corpus-v1/tale_of_errors
```

`--download-all` pulls **~70+ GB compressed** and is **refused in CI**. Prefer selective `--download`.

### 2. Assemble + decompress

```bash
cd ~/.cache/autoresearch/corpus-v1/tale_of_errors
cat trace1_* > trace1-sanitized.tar.zst
cat trace2_* > trace2-sanitized.tar.zst
zstd -d trace1-sanitized.tar.zst   # needs 300–500 GB free per archive
zstd -d trace2-sanitized.tar.zst
tar -xf trace1-sanitized.tar
tar -xf trace2-sanitized.tar
# Point --input at the assembled Jaeger JSON tree (directory of .json traces).
```

### 3. Shard (cap spans for a first pass)

```bash
# Bound the first pass like CRISP subsets — factual train lane only
uv run python -m corpus.ingest.build_shards \
  --adapter tale_of_errors \
  --input /path/to/assembled/jaeger/tree \
  --max-spans 200000 \
  --num-train-shards 8 \
  --write-val-shard
```

Pinned val shard index remains **6542**. Schema stays one `text` column per session (`traceID`) — see [`corpus-v1.md`](corpus-v1.md).

### 4. Prepare (sacred)

```bash
uv run python prepare.py --num-shards 8
```

`prepare.py` currently expects macOS + Metal (same gate as the product Mac path). On Linux / CI, stop after the capped shard smoke above.

### 5. Short train fitness (`val_bpb` only)

```bash
# Bounded wall-clock smoke (same spirit as README quickstart) — reports factual val_bpb if it reaches eval
uv run python -c "
import signal, sys
signal.signal(signal.SIGALRM, lambda s,f: sys.exit(0))
signal.alarm(60)
exec(open('train.py').read())
"

# Or full TIME_BUDGET:
uv run python train.py
```

Record whatever `val_bpb:` the run prints as a **factual training metric**. Do **not** turn it into AUROC / ranking accuracy. There is **no published Tale `val_bpb` baseline** in this repo yet — do not invent one. CRISP numbers in [`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md) are a **different** source / scale; do not 1:1 compare.

---

## What this path is / is not

| Artifact | Role |
|----------|------|
| Tale fixture smoke | Adapter + shard wiring (no Zenodo) |
| Tale scale shards + `prepare.py` + `train.py` | Public-real **train fitness** (`val_bpb`) when you have the disk |
| CRISP `val_bpb` baselines | Separate bootstrap lane — not interchangeable with Tale |
| Lab AUROC / ranking | **`not_published`** until [`lab/publish-checklist.md`](lab/publish-checklist.md) |
| Public ranking card | Synthetic harness smoke — unrelated to Tale |

---

## Related

- Corpus map: [`corpus-v1.md`](corpus-v1.md) · tree: [`corpus/README.md`](../corpus/README.md)
- CRISP train fitness: [`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md)
- BYO dumps (your telemetry): [`byo-and-scorer.md`](byo-and-scorer.md) · `./scripts/byo_score.sh`
- Public wins / stranger path: [`public-wins.md`](public-wins.md)
- Accuracy claim gate (not this path): [`public-accuracy-eval.md`](public-accuracy-eval.md)
