# AOMB Reference Corpus v1 — Provenance & Schema

**Product decision (locked):** the flagship corpus is **real production telemetry only**.
Synthetic / demo / testbed sources are not the product story.

---

## Locked source map

| Role | Source | License | Action in v1 |
|------|--------|---------|--------------|
| **Bootstrap (implement)** | **Uber CRISP** — ~100k prod Jaeger traces, Zenodo `13956078`, `CRISP-main.zip` ~2.33 GB | **CC BY 4.0** | Fetch + ingest wired |
| **Flagship scale (later)** | **Uber Tale of Errors** — ~1.4M sanitized prod Jaeger; DOIs `10.5281/zenodo.13947828` + `13952897`; 300–500 GB decompressed | **CC BY 4.0** | Document + optional adapter; **no CI download** |
| **Eval-only** | **AIOps Challenge 2020** — labeled faults | **Non-commercial** | Cite + fetch locally; **do not redistribute** |
| **Lab (required)** | Org-level local stack with induced faults + OTel export | Apache-2.0 (our code) | `lab/` docker-compose |
| **Smoke / CI only** | `generate_observability_corpus.py` | — | Demoted; not flagship |

### Rejected as flagship (synthetic / testbed)

Do **not** present these as the AOMB reference corpus:

- OpenTelemetry Demo / `smithclay/otel-demo-telemetry`
- `opentelemetry-tracegen`
- Sock Shop + Chaos Mesh Zenodo testbeds
- DeathStarBench packs

See `corpus/ingest/rejected_sources.py`.

---

## Downstream contract (sacred)

`prepare.py` / `evaluate_bpb` / the BPE tokenizer must keep working unchanged.

Ingest writes:

- Path: `~/.cache/autoresearch/data/shard_NNNNN.parquet`
- Schema: single column `text` (`string`)
- Each row = one **session** (multi-line correlated events)
- Pinned validation shard index: `6542`

**Session key for Jaeger public dumps = `traceID`** (multi-service spans share one document).

Event line shape:

```text
[ts=YYYY-MM-DDTHH:MM:SS.sssZ] [src=OTel] trace_id=... span_id=... parent=... op=... svc=... duration_ms=... status=ok|error ...
```

Optional meta line:

```text
# aomb_meta source=uber-crisp-zenodo-13956078 window=normal capture_id=...
```

---

## Source A — Uber CRISP (v1 bootstrap)

| Field | Value |
|-------|--------|
| **Name** | CRISP: Critical Path Analysis of Large-Scale Microservice Architectures (Artifact) |
| **DOI** | [10.5281/zenodo.13956078](https://doi.org/10.5281/zenodo.13956078) |
| **File** | `CRISP-main.zip` (~2.33 GB) |
| **md5** | `efc646e625270685734e8988fc5ef8ec` |
| **Contents** | ~100k sanitized **production** Jaeger traces (multi-service) |
| **License** | **CC BY 4.0** |
| **Format** | Jaeger HTTP API JSON (directory of `.json` traces) |
| **Session** | One AOMB document per `traceID` |
| **Repo** | https://github.com/uber-research/CRISP |

**Attribution / citation** (required under CC BY):

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

**Notes from Zenodo:** unrelated tags removed; start times randomly shifted (relative timings preserved); sanitization mapping is **inconsistent** with Tale of Errors — do not mix.

### Reproduce

```bash
# Prints manual steps; use --download only when you intend to pull ~2.33 GB
uv run python -m corpus.ingest.fetch_crisp
uv run python -m corpus.ingest.fetch_crisp --download   # optional, not CI

uv run python -m corpus.ingest.build_shards \
  --adapter crisp_zenodo \
  --input ~/.cache/autoresearch/corpus-v1/crisp/extracted \
  --num-train-shards 8 \
  --write-val-shard

uv run python prepare.py --num-shards 8
```

---

## Source B — Uber Tale of Errors (flagship scale, optional later)

| Field | Value |
|-------|--------|
| **Part 1** | [10.5281/zenodo.13947828](https://doi.org/10.5281/zenodo.13947828) |
| **Part 2** | [10.5281/zenodo.13952897](https://doi.org/10.5281/zenodo.13952897) |
| **Scale** | ~1.4M sanitized production Jaeger traces |
| **Disk** | Split `.tar.zst` pieces; **300–500 GB decompressed per archive** |
| **License** | **CC BY 4.0** |
| **CI** | Full download **not required** and must not be part of CI |

Adapter `tale_of_errors` accepts a *local* assembled Jaeger JSON tree (same parser as CRISP) for experiments. Document reassembly from Zenodo part1+part2 before use. Cite the SIGMETRICS 2025 paper / Zenodo records.

---

## Eval-only — AIOps Challenge 2020

| Field | Value |
|-------|--------|
| **Repo** | https://github.com/NetManAIOps/AIOps-Challenge-2020-Data |
| **Signals** | Labeled faults + metrics + call-chain traces |
| **License** | **Non-commercial** (research / classroom); do not redistribute via AOMB |
| **Fetch help** | `uv run python -m corpus.ingest.fetch_aiops_challenge` |

Use for **evaluation** (labeled fault windows), not as the training flagship story.

---

## Lab-captured OTel (required dual source)

| Field | Value |
|-------|--------|
| **Stack** | `lab/docker-compose.yml` — frontend, API, Postgres, Redis, OTel Collector |
| **Faults** | `lab/scripts/inject_faults.sh` |
| **Capture** | `lab/scripts/run_capture_session.sh` → `normal` then `incident` windows |

```bash
cd lab && docker compose up -d --build && ./scripts/run_capture_session.sh
uv run python -m corpus.ingest.build_shards \
  --adapter lab_capture --input lab/captures/<id> \
  --num-train-shards 4 --write-val-shard
```

Window labels come from **capture metadata** (`provenance.json`), not invented per-event flags.

---

## Provenance sidecar schema

Written under `~/.cache/autoresearch/corpus-v1/provenance/`:

```json
{
  "corpus_version": "v1",
  "source_id": "uber-crisp-zenodo-13956078 | lab-aomb-stack | uber-tale-of-errors",
  "source_kind": "public_real | lab_capture",
  "license": "CC-BY-4.0 | Apache-2.0",
  "citation": "...",
  "captured_at": "ISO-8601",
  "windows": [{"label": "normal|incident", "start": "...", "end": "...", "fault": "..."}],
  "session_count": 0,
  "shard_indices": [0, 6542]
}
```

---

## Building combined v1

1. Ingest **CRISP** into train shards `0..N-1` (+ pinned val `6542`).
2. Ingest **lab** captures into subsequent shards (normal + incident).
3. Optionally hold out AIOps-labeled periods for eval only.
4. Scale later with Tale of Errors (local disks only).
5. `prepare.py --num-shards <N>` then smoke `train.py` / `demo_anomaly.py`.

**Do not** mix synthetic shards or rejected demo/testbed dumps into the flagship story.
