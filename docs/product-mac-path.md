# Product Mac path — stranger CPU → real MPS train fitness

**Audience:** Apple Silicon Mac user who already proved the public stranger gates elsewhere (Actions / Codespaces / Linux CPU).

**Honest bridge:** green stranger verify ≠ product train. Stranger paths are **CPU** harness smoke (full-8 + ranking-card baselines). Product overnight / `TIME_BUDGET` breed is **Apple Silicon MPS** and reports factual **`val_bpb` only** — not AUROC, not CUDA.

`prepare.py` is **sacred**. No API keys required for the smoke train path below (keys are overnight `agent_loop` only).

---

## What this path proves / does not prove

| Proves | Does **not** prove |
|--------|---------------------|
| Darwin + MPS available; shards ready; a local `train.py` run prints **`val_bpb`** | Lab AUROC (`not_published` — [`lab/publish-checklist.md`](lab/publish-checklist.md)) |
| Train fitness on CRISP (preferred) or documented smoke corpus | CUDA claim ([`compute-paths.md`](compute-paths.md) checklist only) |
| Same honesty as [`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md) | Production / field ranking accuracy |

---

## One-command smoke (Mac only)

```bash
./scripts/product_mac_smoke.sh
```

Fails **loudly** unless:

- `uname` is Darwin
- PyTorch reports MPS available
- No CUDA-fallback / “pretend GPU” path is requested

Optional env:

| Var | Default | Meaning |
|-----|---------|---------|
| `PRODUCT_MAC_SMOKE_SECONDS` | `60` | Wall-clock bound for the train smoke (SIGALRM); full `TIME_BUDGET` = unset/`0` then run `uv run python train.py` yourself |
| `PRODUCT_MAC_CORPUS` | `auto` | `auto` = use CRISP shards if present, else refuse with fetch instructions; `smoke` = synthetic generator path (CI/dev only — **not** the flagship CRISP story) |

No Anthropic/OpenAI keys. No invented metrics — script only surfaces the `val_bpb:` line from `train.py` when training completes.

---

## Manual steps (same honesty)

### 1. Prefer Uber CRISP (product train story)

```bash
uv sync
uv run python -m corpus.ingest.fetch_crisp                 # or --download (~2.33 GB)
uv run python -m corpus.ingest.build_shards \
  --adapter crisp_zenodo \
  --input ~/.cache/autoresearch/corpus-v1/crisp/extracted \
  --num-train-shards 8 --write-val-shard
uv run python prepare.py --num-shards 8                   # sacred — do not edit
uv run python train.py                                    # full TIME_BUDGET (300s) → prints val_bpb
```

Factual tables: [`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md). Do **not** blend with synthetic smoke-era numbers.

### 2. Synthetic smoke corpus (dev only)

```bash
uv run python generate_observability_corpus.py            # SMOKE / CI ONLY
uv run python prepare.py --num-shards 20
PRODUCT_MAC_CORPUS=smoke ./scripts/product_mac_smoke.sh
```

Not interchangeable with CRISP `val_bpb`.

### 3. Overnight breed (separate — needs API keys)

```bash
caffeinate -i uv run python agent_loop.py >> logs/agent_loop.log 2>&1 &
```

Out of scope for stranger paths and for the no-keys smoke script. See README overnight section.

---

## Explicit non-claims

- Stranger CPU green ≠ MPS product fitness.
- This path does **not** publish lab AUROC or flip [`lab/publish-checklist.md`](lab/publish-checklist.md).
- No CUDA numbers; no wall-clock “speedup” claims vs stranger CPU.
- Smoke-bounded runs are **not** the CRISP-500k / overnight baselines — cite those only from [`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md).

Related: [`anomaly-story.md`](anomaly-story.md) (thesis first) · [`compute-paths.md`](compute-paths.md) · [`public-wins.md`](public-wins.md) · [`stranger-verify.md`](stranger-verify.md) · [`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md) · README [Requirements / overnight](../README.md#requirements).
