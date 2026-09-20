# Anomaly story — `val_bpb` *is* the signal

**Audience:** someone who can clone and wants the product idea in under ~30 minutes.  
**Hardware:** CPU is enough. MPS is nicer, not required. **No API keys. No CUDA claim.**

`prepare.py` is **sacred** — this path never edits it.

---

## One sentence

Train so the model is less surprised by *normal* telemetry → score a session → **higher bits-per-byte (surprise) ≈ more anomalous.**

That surprise number is the same family as train-lane **`val_bpb`**. No separate anomaly head. No labels at train time. No invented AUROC.

---

## Three steps (clone → see the gap)

```bash
git clone https://github.com/pandeyaby/AOMB.git && cd AOMB
uv sync   # needs torch; CPU wheel OK: pip install torch --index-url https://download.pytorch.org/whl/cpu

# 1) Smoke corpus + sacred prepare (dev path — not Uber CRISP)
uv run python generate_observability_corpus.py
uv run python prepare.py --num-shards 20

# 2) Short train + score held-out sessions
uv run python demo_anomaly.py
# tighter wall:  uv run python demo_anomaly.py --seconds 45 --per-class 12
```

**What to look for in stdout:** `anomalous` / `cascade` **mean_bpb** above **normal**. The gap *is* the detector.

---

## How the pieces map

| Step | What you do | What the number means |
|------|-------------|------------------------|
| Train fitness | Model learns next-token on normal-ish streams | Lower corpus **`val_bpb`** = better grasp of *normal* |
| Score a session | Same CE → bits-per-byte on one session | Session **BPB / surprise** |
| Read the gap | Compare classes or sessions | Higher surprise ≈ more off-manifold |

Same objective. Fixture ranking-card AUROC / lab AUROC are **other lanes** — not this story.

---

## Honesty (loud, short)

| Surface | Status |
|---------|--------|
| This demo gap | Qualitative smoke on synthetic generator — **understand the thesis** |
| Public ranking card | **`published_fixture_card` / harness smoke** only — tiny-n; not production |
| Lab AUROC | Stays **`not_published`** — **no invented AUROC** |
| CRISP / Tale `val_bpb` | Train fitness only when cited — **not** ranking accuracy |
| CUDA | **No claim** |

Public gates (DIPTYCH emit + card ε) without the story: [`stranger-verify.md`](stranger-verify.md) · [`stranger-60s.md`](stranger-60s.md).

---

## After you get the story (optional)

Not required for the 30-minute bar:

1. **Product Mac MPS** — real train fitness on Apple Silicon: [`product-mac-path.md`](product-mac-path.md) · `./scripts/product_mac_smoke.sh`
2. **BYO scorer** — your dump → session BPB only: [`byo-and-scorer.md`](byo-and-scorer.md) · `./scripts/byo_score.sh /path/to/dump`
3. **Uber CRISP / Tale train lane** — factual `val_bpb` tables: [`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md) · [`tale-val-bpb-baseline.md`](tale-val-bpb-baseline.md) (Tale measured row may still be **pending**)

---

## Related

- Script: [`demo_anomaly.py`](../demo_anomaly.py)
- Stranger gates (no story required): [`stranger-demo.md`](stranger-demo.md) · `./scripts/stranger_demo.sh`
- Outsider index: [`public-wins.md`](public-wins.md)
- Idea / metric table: README → *The Idea* · *val_bpb — The Only Metric That Matters*
