# Autonomous Observability Model Breeder (AOMB)

![Traces → small next-token ILM → anomaly via surprise; Apple Silicon research loop](docs/assets/aomb-readme-hero.png)

> *"Frontier AI research used to require meat computers. Now it runs overnight on your MacBook."*

AOMB is the first open-source **Infrastructure Language Model (ILM)** — the same autoregressive architecture as GPT (RoPE, GQA, focal loss), trained on enterprise observability telemetry instead of the internet.

A domain-specific fork of [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch) —
adapted for Apple Silicon by [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) —
with a Claude agent that reads `program.md`, edits `train.py`, runs 5-minute experiments, commits improvements, and loops —
all while you sleep. You wake up to a git log of experiments and a better model.

## Write-up

[I Let an AI Improve Itself Overnight. Here's What I Woke Up To](https://medium.com/@pandeyaby/i-let-an-ai-improve-itself-overnight-heres-what-i-woke-up-to-6db1905fc212) — Abhinav's overnight autonomous research-loop story for AOMB.

---

## The Idea

Monitoring tools today are rule-based and threshold-driven. Someone decided `latency_ms > 500` means alert.
The on-call engineer gets paged 847 times a week. 803 are noise.

AOMB takes the language model approach: **train a model to predict what comes next in your telemetry stream.**
A model that predicts well has learned what *normal* looks like. Anomalies are sequences the model finds surprising.
No rules. No labels. No thresholds. Just next-token prediction — and the anomaly detection is a free consequence.

The training objective (`val_bpb` — validation bits-per-byte) *is* the anomaly detection capability.
Lower val_bpb = model understands your infrastructure's language = better anomaly detector.

An ILM trained on your own telemetry has an anomaly detector no vendor can replicate — because the model learned the statistical fingerprint of that specific environment.

---

## Quickstart

```bash
# Requirements: macOS + Apple Silicon, Python 3.10+, uv
# Optional for real corpus: Docker (lab), ~2.3GB disk+net for Uber CRISP fetch

# 1. Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 2a. FLAGSHIP — real reference corpus v1 (Uber CRISP + lab). See section below.
#     Details: docs/corpus-v1.md

# 2b. SMOKE / CI ONLY — synthetic generator (not the product corpus)
uv run python generate_observability_corpus.py

# 3. Train BPE tokenizer on telemetry vocabulary (~4 sec)
uv run python prepare.py --num-shards 20

# 4. Verify the training loop works (60-second smoke test)
uv run python -c "
import signal, sys
signal.signal(signal.SIGALRM, lambda s,f: sys.exit(0))
signal.alarm(60)
exec(open('train.py').read())
" 2>&1 | tail -5

# 5. Anomaly story — surprise/BPB on normal vs anomalous vs cascade (~1 min)
#    Look for: anomalous and cascade mean_bpb >> normal.
#    That gap is the detector: next-token surprise *is* anomaly detection.
uv run python demo_anomaly.py

# 6. Run the autonomous agent loop overnight
caffeinate -i uv run python agent_loop.py >> logs/agent_loop.log 2>&1 &
echo "Agent running. Check logs/agent_loop.log. Sleep well."

# 7. Morning report
uv run python morning_report.py --plot
```

---

## Reference corpus v1 (real)

**Flagship training data is real production telemetry** — not synthetic demos/testbeds.

| Role | Source |
|------|--------|
| **(A) Public-real bootstrap** | **Uber CRISP** — ~100k production Jaeger traces, [Zenodo 13956078](https://doi.org/10.5281/zenodo.13956078), `CRISP-main.zip` ~2.33 GB, **CC BY 4.0** (cite Zhang et al., ATC'22). Session = `traceID`. |
| **(B) Lab-captured** | Org-level stack (`lab/`) with OTel export + induced faults; windows `normal` / `incident` |
| **Flagship scale (later)** | Uber Tale of Errors — DOIs [13947828](https://doi.org/10.5281/zenodo.13947828) + [13952897](https://doi.org/10.5281/zenodo.13952897), ~1.4M traces, CC BY 4.0, **300–500GB decompressed** — document only; not CI |
| **Eval-only** | AIOps Challenge 2020 — labeled faults, **non-commercial**; cite+fetch, do not redistribute |
| **Rejected as flagship** | OTel Demo / `otel-demo-telemetry`, tracegen, Sock Shop+Chaos Mesh testbeds, DeathStarBench |

Full provenance: **[`docs/corpus-v1.md`](docs/corpus-v1.md)**. Ingest: [`corpus/README.md`](corpus/README.md). Lab: [`lab/README.md`](lab/README.md).

### Public accuracy eval (protocol)

AOMB’s **public accuracy claim** (when published) is a **ranking** claim: session-level next-token surprise / BPB ranks incident/cascade windows above normal — measured with AUROC, PR-AUC, and precision@k over multi-seed runs.

That claim is **not published** until the pass/fail checklist in **[`docs/public-accuracy-eval.md`](docs/public-accuracy-eval.md)** passes. Until labeled ranking metrics exist, ship **protocol/checklist + harness only** — no inflated accuracy language. Harness: [`eval/`](eval/).

Keep factual `val_bpb` lanes separate (neither is public accuracy):

| Lane | val_bpb | Role |
|------|---------|------|
| Synthetic / smoke-era | **0.3682** | Legacy generator overnight best — separate README table |
| CRISP factual baseline | **0.458756** | README subset run only — [`docs/crisp-val-bpb-baseline.md`](docs/crisp-val-bpb-baseline.md) |

Do **not** 1:1 compare or blend those two numbers. CRISP alone is insufficient for the ranking claim (no incident labels); prefer lab captures with provenance windows.

`prepare.py` stays sacred — ingest writes the same parquet shape (`text` column, pinned val shard `6542`).

### Build from Uber CRISP (public-real)

```bash
uv run python -m corpus.ingest.fetch_crisp                 # instructions / use local zip
# uv run python -m corpus.ingest.fetch_crisp --download    # ~2.33 GB — not for CI

uv run python -m corpus.ingest.build_shards \
  --adapter crisp_zenodo \
  --input ~/.cache/autoresearch/corpus-v1/crisp/extracted \
  --num-train-shards 8 \
  --write-val-shard

uv run python prepare.py --num-shards 8
```

### Run lab capture (Docker)

```bash
cd lab
docker compose up -d --build
./scripts/run_capture_session.sh    # normal → fault → incident → export
cd ..

uv run python -m corpus.ingest.build_shards \
  --adapter lab_capture \
  --input lab/captures/<capture_id> \
  --num-train-shards 4 \
  --write-val-shard
```

Fault modes: `latency`, `errors`, `both`, `kill_redis`, `kill_postgres` (`lab/scripts/inject_faults.sh`).

### Synthetic = smoke / CI only

`generate_observability_corpus.py` remains for fast loops without Docker or Zenodo.
It is **not** the reference corpus product story.

### Bring your own (BYO) telemetry

Point a new adapter at your Jaeger/OTLP dump (see `corpus/ingest/adapters/base.py`),
or convert to the session line format and write `shard_*.parquet` yourself.
Same downstream path: `prepare.py` → `train.py` / `evaluate_bpb`.

---

## How It Works

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT LOOP                           │
│                                                         │
│  program.md + train.py + git log                        │
│         │                                               │
│         ▼                                               │
│   claude --print ──► proposed train.py                  │
│         │                                               │
│         ▼                                               │
│   uv run train.py  (5 minutes, MPS)                     │
│         │                                               │
│         ▼                                               │
│   val_bpb improved? ──Yes──► git commit ──► loop        │
│         │ No                                            │
│         ▼                                               │
│   restore backup ──────────────────────────► loop       │
└─────────────────────────────────────────────────────────┘
```

**Three files. That's the whole system.**

| File | Who touches it | What it is |
|------|---------------|------------|
| `prepare.py` | Nobody | Data pipeline, tokenizer, `evaluate_bpb` — sacred, never modified |
| `train.py` | The Claude agent | Model architecture + training loop — the only thing that changes |
| `program.md` | You (rarely) | Research constitution — domain context, experiment queue, constraints |

The agent loop (`agent_loop.py`) drives the cycle automatically. You set it and forget it.

---

## The Training Data (legacy synthetic smoke path)

> Prefer **[Reference corpus v1 (real)](#reference-corpus-v1-real)** above.
> The shards below describe what `generate_observability_corpus.py` produces for smoke/CI.

21 parquet shards (~22 MB) in `~/.cache/autoresearch/data/`, generated by `generate_observability_corpus.py` (**smoke only**).
Each row is a coherent session of 8–60 correlated events across telemetry sources:

```
[ts=2026-03-08T18:00:00Z] [src=APMTracer] [svc=payment-gateway] latency_ms=420 error=timeout trace_id=a3f9 http_status=500 drift_score=0.87
[ts=2026-03-08T18:00:01Z] [src=NetIntel] path=internet→aws-us-east-1 latency_ms=3200 packet_loss=0.123 bgp_changes=3
[ts=2026-03-08T18:00:01Z] [src=LogStream] level=CRITICAL svc=auth msg=circuit_breaker_open latency_ms=28500 pagerduty=triggered
[ts=2026-03-08T18:00:02Z] [src=OTel] trace_id=a3f9 op=db.query svc=inventory-db duration_ms=390 status=ok
[ts=2026-03-08T18:00:02Z] [src=WANController] site=branch-07 link=mpls→inet bw_util=0.982 policy=VIOLATED
[ts=2026-03-08T18:00:03Z] [src=APMTracer-BizTxn] svc=checkout health=STALL response_time_ms=28900 error_pct=0.812
```

**Statistical properties:** ~91% normal, ~6% anomalous, ~3% cascade failures across 20–60 correlated events.
BPE vocab of 8,192 tokens trained on this corpus — field names like `latency_ms=`, `trace_id=` become single tokens.

---

## Visualize the Corpus

After building shards (real ingest or synthetic smoke), explore them:

```bash
# Full report + save corpus_overview.png
uv run python visualize_corpus.py

# Terminal-only (no plot)
uv run python visualize_corpus.py --no-plot

# Print 3 sample sessions (normal / anomalous / cascade)
uv run python visualize_corpus.py --samples
```

Sample terminal output:
```
  Session Distribution
  ─────────────────────────────────────────────
  normal       19,019  ( 90.6%)  ██████████████████████████████████████████████
  anomalous     1,260  (  6.0%)  ███
  cascade         721  (  3.4%)  █

  Telemetry Source Mix  (events, not sessions)
  ─────────────────────────────────────────────
  APMTracer             132,481  ( 24.9%)  ████████████
  LogStream             132,202  ( 24.8%)  ████████████
  OTel                  106,208  ( 19.9%)  ██████████
  NetIntel               79,716  ( 15.0%)  ███████
  WANController          53,234  ( 10.0%)  █████
  APMTracer-BizTxn       27,183  (  5.1%)  ██
```

The 4-panel `corpus_overview.png` shows session-type pie, source event counts,
session-length histogram, and normal vs anomalous latency distributions on one page.

To explore raw parquet data directly:

```python
import pyarrow.parquet as pq, os

data_dir = os.path.expanduser("~/.cache/autoresearch/data")
df = pq.read_table(f"{data_dir}/shard_00000.parquet").to_pandas()

# Print a session
print(df["text"].iloc[0])

# Find anomalous sessions
df[df["text"].str.contains("pagerduty=triggered|policy=VIOLATED")]
```

---

## The Model Architecture

Not vanilla nanoGPT. Built-in, state-of-the-art from day one:

| Component | What it does |
|-----------|-------------|
| **RoPE** | Rotary position embeddings — relative position, not absolute |
| **GQA** | Grouped Query Attention — fewer KV heads, memory efficient |
| **Sliding window** | `WINDOW_PATTERN="L"` — per-layer full or half-context attention |
| **MuonAdamW** | Muon (orthogonal updates) for matrices, AdamW for embeddings |
| **Value Embeddings** | ResFormer-style residual on alternating layers |
| **Logit softcapping** | `tanh(x/15)×15` — no gradient clipping needed |
| **RMSNorm** | Everywhere, no bias |

**What the agent explores:**

```python
DEPTH = 4               # layers: try 2, 3, 4, 6
ASPECT_RATIO = 64       # model_dim = DEPTH × ASPECT_RATIO
HEAD_DIM = 128          # attention head size
WINDOW_PATTERN = "L"    # try "SSL", "SSSL", "SL"

EMBEDDING_LR = 0.6      # 4-way learning rate split
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04        # Muon optimizer
SCALAR_LR = 0.5

WARMUP_RATIO = 0.0      # LR schedule shape
WARMDOWN_RATIO = 0.5
TOTAL_BATCH_SIZE = 2**16
```

---

## Empirical Results — Reference corpus v1 (Uber CRISP)

Honest recorded run on a **CRISP subset** (not a public accuracy claim, marketing number, or product benchmark).
README / factual documentation only — see [`docs/crisp-val-bpb-baseline.md`](docs/crisp-val-bpb-baseline.md).

Corpus: Uber CRISP ([Zenodo 13956078](https://doi.org/10.5281/zenodo.13956078), CC BY 4.0; cite Zhang et al., USENIX ATC'22),
`CRISP-main/data/bottom-up-trace` with `--max-spans 200000`.

| Date | Corpus | Hardware | Config | val_bpb | Notes |
|------|--------|----------|--------|---------|-------|
| 2026-09-14 | CRISP bottom-up-trace, `--max-spans 200000`, 20 train shards + val `shard_06542` | MacBook Pro Apple Silicon (MPS) | DEPTH=4, WINDOW=SSL, ~8.5M params, vocab 5206 | **0.458756** | Single 5-min run (`TIME_BUDGET=300`); no overnight `agent_loop` / no API keys |

| Metric | Value |
|--------|-------|
| `val_bpb` | **0.458756** |
| `training_seconds` | 300.1 |
| `total_seconds` (includes eval) | 401.4 |
| `num_steps` | 603 |
| `total_tokens_M` | 19.8 |
| `num_params_M` | 8.5 |
| depth / `window_pattern` | 4 / SSL |
| `vocab_size` | 5206 |
| Sessions | 2185 (1967 train / 218 val); 200000 spans; 2185 Jaeger JSON files |
| Windows | normal only (CRISP dump has no incident labels) |
| Provenance id | `crisp_zenodo_20260914T050751Z.json` |

**Do not 1:1 compare** this CRISP subset `val_bpb` to the synthetic smoke-era **0.3682** below — different data, tokenizer, and scale. Keep the two tables separate; neither is a public accuracy claim.

### Reproduce this baseline

```bash
uv run python -m corpus.ingest.fetch_crisp --download   # ~2.33 GB CRISP-main.zip

uv run python -m corpus.ingest.build_shards \
  --adapter crisp_zenodo \
  --input ~/.cache/autoresearch/corpus-v1/crisp/extracted \
  --max-spans 200000 \
  --num-train-shards 20 \
  --write-val-shard
# Subset path inside extract: CRISP-main/data/bottom-up-trace

uv run python prepare.py --num-shards 20
uv run python train.py
```

---

## Empirical Results — Synthetic / smoke-era (legacy)

> **Synthetic generator only** (`generate_observability_corpus.py`). Not the reference corpus product story.
> Numbers below are retained for historical overnight agent_loop runs on smoke data.

| Run | Hardware | Config | Best val_bpb | Experiments |
|-----|----------|--------|-------------|-------------|
| Night 3+ | MacBook Pro M-series | Focal loss + anomaly token weighting + Adam tuning | **0.3682** | 12 successful / 120 total |
| Night 2 | MacBook Pro M-series | DEPTH=4, WINDOW=SSL, EMBEDDING_LR=0.3 | 0.4297 | 6 successful / 14 total |
| Night 1 | MacBook Pro M-series | DEPTH=4, WINDOW=L | 0.4349 | 2 successful / 10 total |
| Baseline (5 min) | MacBook Pro M-series | Default | 0.4372 | 1 |
| Random model | — | — | ~8.0 | — |

Training throughput: ~63,000 tokens/sec on Apple Silicon MPS.
Each experiment cycle: ~10–15 minutes (Claude SDK call ~2–3 min + training 5 min + eval ~2 min).

**val_bpb progression across all successful experiments (synthetic corpus):**

| Exp | SHA | val_bpb | Δ | Change |
|-----|-----|---------|---|--------|
| Baseline | 2861c70 | 0.4372 | — | DEPTH=4, WINDOW=L |
| 5 | 5962a57 | 0.4349 | -0.0023 | Sliding window attention |
| 6 | 26cf5e3 | 0.4339 | -0.0010 | WINDOW_PATTERN = "SSL" |
| 1 | 7a4eb57 | 0.4323 | -0.0016 | Architecture sweep |
| 3 | 7c99e0e | 0.4300 | -0.0023 | TOTAL_BATCH_SIZE tuning |
| 13 | dc4df4a | 0.4297 | -0.0003 | EMBEDDING_LR 0.6 → 0.3 |
| **17** | **c2026cc** | **0.3950** | **-0.0347** | **Focal loss + anomaly token weighting** ← big jump |
| 18 | e1b363a | 0.3856 | -0.0094 | Focal loss refinement |
| 23 | 53b5b00 | 0.3855 | -0.0001 | WARMDOWN_RATIO 0.5 → 0.65 |
| 27 | 92e58f1 | 0.3771 | -0.0084 | Loss + config refinements |
| 28 | 3168d49 | 0.3771 | -0.0000 | FINAL_LR_FRAC reduction |
| 33 | c887b39 | 0.3697 | -0.0074 | Forward pass loss calc |
| 37 | da353a6 | 0.3692 | -0.0005 | Loss calc tweak |
| 66 | 96c3404 | 0.3692 | -0.0000 | Cascade detection refinement |
| 70 | fa2ee1e | 0.3691 | -0.0001 | Domain-aware loss tuning |
| 109 | 77e9eba | 0.3686 | -0.0005 | Adam betas tuning |
| 113 | 64b56fb | 0.3685 | -0.0001 | Adam betas refinement |
| **114** | **983ee44** | **0.3682** | **-0.0003** | **Adam optimizer tuning ← synthetic best** |

---

## val_bpb — The Only Metric That Matters

```
val_bpb = total_nats / (log(2) × total_bytes)
```

Bits-per-byte is vocabulary-independent within a fixed tokenizer/corpus.
**Do not treat scores from different corpora as interchangeable** (e.g. CRISP **0.458756** vs synthetic **0.3682**).

| val_bpb | What it means |
|---------|---------------|
| > 4.0 | Model barely beats random — hasn't learned field structure yet |
| 1.5 – 4.0 | Early convergence — learning token distributions |
| 0.8 – 1.5 | Good — model understands normal telemetry patterns |
| 0.4 – 0.8 | Strong — implicit anomaly detector, approaching production use |
| **0.458756** | **← CRISP subset run (2026-09-14); README fact only — not a marketing claim** |
| 0.4297 | ← synthetic Night 2 best (exp 13) |
| 0.3692 | ← synthetic exp 37–66 (focal loss + anomaly weighting) |
| 0.3691 | ← synthetic exp 70–113 (domain-aware loss tuning) |
| **0.3682** | **← synthetic smoke-era best (exp 114); separate table above — not comparable to CRISP** |
| < 0.35 | Excellent — deploy as zero-shot anomaly scorer |

The information-theoretic argument: minimizing val_bpb = minimizing KL(P_data ‖ P_model).
A model close to the true data distribution assigns high surprise to anomalous sequences automatically.
**The training objective IS the anomaly detection capability. No separate head. No labels.**

CRISP `val_bpb=0.458756` and synthetic `0.3682` are logged in separate tables above and must not be mixed or marketed as a single accuracy story.

---

## Observability Sources in the Corpus

| Source | Signal type |
|--------|-------------|
| **APMTracer** | APM traces, latency, error rates, GPU/AI agent cost drift |
| **NetIntel** | Network path quality, BGP stability, DNS, packet loss |
| **LogStream** | Structured logs, SIEM alerts, security events |
| **OpenTelemetry** | Distributed trace spans, service dependency chains |
| **WANController** | Branch WAN link health, QoS violations, bandwidth utilization |
| **APMTracer-BizTxn** | End-to-end business transaction health snapshots |

---

## Morning Report

```bash
uv run python morning_report.py --plot
```

```
========================================================================
  AUTONOMOUS OBSERVABILITY MODEL BREEDER — MORNING REPORT
========================================================================
  Experiments completed  : 120
  Starting val_bpb       : 0.4372
  Best val_bpb           : 0.3682  (15.8% improvement)
  val_bpb trend          : ▇▆▅▅▄▄▃▃▃▂▂▂▁▁▁▁

     #  SHA        val_bpb        Δ  Change
  ----  --------  --------  -------  ------------------------------------
     1  2861c704    0.4372           baseline — DEPTH=4, WINDOW=L
     5  5962a578    0.4349  -0.0023  sliding window attention
     6  26cf5e3a    0.4339  -0.0010  WINDOW_PATTERN = SSL
    13  dc4df4ab    0.4297  -0.0003  EMBEDDING_LR 0.6 → 0.3
    17  c2026ccf    0.3950  -0.0347  focal loss + anomaly token weighting
    18  e1b363af    0.3856  -0.0094  focal loss refinement
    27  92e58f1d    0.3771  -0.0084  loss + config refinements
    37  da353a6e    0.3692  -0.0005  loss calc tweak
    70  fa2ee1ef    0.3691  -0.0001  domain-aware loss tuning
   114  983ee440    0.3682  -0.0003  Adam optimizer tuning  ◀ BEST

  Restore best:    git checkout 983ee44 -- train.py
========================================================================
```

Generates `overnight_progress.png` convergence plot alongside the terminal report.

---

## The Agent Loop

`agent_loop.py` is the full autonomous orchestrator:

```bash
# Start
caffeinate -i uv run python agent_loop.py >> logs/agent_loop.log 2>&1 &

# Monitor
tail -f logs/agent_loop.log

# Stop cleanly
kill $(cat logs/agent_loop.pid)
```

Config (top of `agent_loop.py`):
```python
MAX_EXPERIMENTS = 120     # overnight cap
CLAUDE_TIMEOUT  = 300     # 5 min for Claude to respond
TRAIN_TIMEOUT   = 660     # 11 min for larger model variants
CLAUDE_MODEL    = "sonnet"  # default; override via AOMB_CLAUDE_MODELS env var
```

**API key configuration** (environment variables, or a `.env` file sourced before launch):
```bash
AOMB_ANTHROPIC_API_KEYS=sk-ant-...   # one or more keys, comma-separated
AOMB_OPENAI_API_KEYS=sk-proj-...     # optional OpenAI fallback
AOMB_CLAUDE_MODELS=sonnet            # or: opus, haiku, gpt-4o-mini
```

Every improvement is a git commit. The git log is the experiment log.
Every 10 successful experiments, results push automatically to GitHub.

---

## Scheduled Morning Report

```bash
# 1. Generate your machine-specific plist from the template
AOMB_DIR="$(pwd)"
sed "s|AOMB_DIR|${AOMB_DIR}|g" com.aomb.morning-report.plist.template \
  > ~/Library/LaunchAgents/com.aomb.morning-report.plist

# 2. Load the launchd job — fires at 8:00 AM daily
launchctl load ~/Library/LaunchAgents/com.aomb.morning-report.plist

# 3. Verify it's loaded
launchctl list com.aomb.morning-report
```

> The `.plist.template` uses `AOMB_DIR` as a placeholder.
> The generated `.plist` (with your actual paths) lives in `~/Library/LaunchAgents/` and is gitignored.

---

## Repository Lineage

```
karpathy/autoresearch          (original — H100, NVIDIA)
       │
       └── miolini/autoresearch-macos   (macOS/MPS port)
       │          │
       │          └── trevin-creator/autoresearch-mlx  (MLX native)
       │
       └── pandeyaby/AOMB  ← you are here
                   Domain: enterprise observability telemetry
                   Loop:   Fully autonomous (agent_loop.py drives everything)
                   Goal:   Minimize val_bpb → maximize implicit anomaly detection
```

---

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- `uv` package manager
- An Anthropic API key (`AOMB_ANTHROPIC_API_KEYS`) — or Claude Code installed for CLI fallback
- ~500 MB disk space for corpus + tokenizer

---

## License

MIT — builds on [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MIT)
which builds on [karpathy/autoresearch](https://github.com/karpathy/autoresearch) (MIT).

*Authored by Abhinav Pandey*
