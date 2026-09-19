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

# 2a. Prefer real Uber CRISP for training (see "Three lanes"). Details: docs/corpus-v1.md

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

## Three lanes (read this once)

AOMB uses **three separate lanes**. Do not mix their numbers.

| Lane | What it is | What you can say |
|------|------------|------------------|
| **1. Train (Uber CRISP)** | Real public Jaeger traces for training | Factual **`val_bpb`** only — CRISP has **no incident labels**, so it is **not** ranking accuracy |
| **2. Lab (private labeled)** | Your Docker stack + faults; optional redacted public pack | Ranking evidence in [`docs/lab/`](docs/lab/) — default **`not_published`** |
| **3. Public fixture card** | Tiny synthetic pack strangers can clone + CI | **`published_fixture_card`** = harness smoke that beat baselines — **not** production AUROC |

Details: CRISP train [`docs/corpus-v1.md`](docs/corpus-v1.md) · `val_bpb` facts [`docs/crisp-val-bpb-baseline.md`](docs/crisp-val-bpb-baseline.md) · fixture card [`docs/public-ranking-card-v1.md`](docs/public-ranking-card-v1.md) · lab [`docs/lab/`](docs/lab/) · protocol [`docs/public-accuracy-eval.md`](docs/public-accuracy-eval.md).

### Train on Uber CRISP

```bash
uv run python -m corpus.ingest.fetch_crisp                 # or --download (~2.33 GB, not CI)
uv run python -m corpus.ingest.build_shards \
  --adapter crisp_zenodo \
  --input ~/.cache/autoresearch/corpus-v1/crisp/extracted \
  --num-train-shards 8 --write-val-shard
uv run python prepare.py --num-shards 8
uv run python train.py
```

Current factual CRISP `val_bpb` (training fitness, **not** accuracy): **0.407753** (500k spans) · overnight 200k best **0.4309**. Full tables live in the CRISP doc above.

### Lab capture (labeled)

```bash
cd lab && docker compose up -d --build && ./scripts/run_capture_session.sh && cd ..
```

Private captures stay on your machine. A **redacted** public-safe pack is at [`corpus/fixtures/lab_public_pack_v0/`](corpus/fixtures/lab_public_pack_v0/) (`claim_status=not_published`). Do **not** put private lab AUROC on the README hero.

### Public ranking card (clone → verify)

```bash
./scripts/run_public_ranking_card_v1.sh
```

Synthetic fixture only. Soft status: fixture-card / harness smoke. See the card doc.

### DIPTYCH paired probes (full-8 harness smoke)

```bash
./scripts/run_diptych_full8.sh
```

All 8 operators × conforming/violating under `diptych-probes/` (`diptych_schema=0.2`). Coverage matrix: `coverage/matrix.json`. Docs: [`docs/paired-probes/`](docs/paired-probes/). **Not** lab AUROC; **not** a public ranking claim.

### BYO telemetry

Bring your own dumps: [`docs/byo-and-scorer.md`](docs/byo-and-scorer.md). Scoring ≠ a published accuracy claim.

### Synthetic generator = smoke / CI only

`generate_observability_corpus.py` is for fast loops — **not** the flagship train story.

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

## Results (where the numbers live)

- **CRISP `val_bpb` (train fitness):** [`docs/crisp-val-bpb-baseline.md`](docs/crisp-val-bpb-baseline.md)
- **Public fixture card (harness smoke):** [`docs/public-ranking-card-v1.md`](docs/public-ranking-card-v1.md) · `reports/public-ranking-card-v1/CARD.md`
- **Lab ranking (private / redacted pack):** [`docs/lab/`](docs/lab/) — `not_published` by default

Legacy overnight synthetic `val_bpb` history stays in git history / morning reports — not the product headline.

---

## val_bpb — The Only Metric That Matters

```
val_bpb = total_nats / (log(2) × total_bytes)
```

Bits-per-byte is vocabulary-independent within a fixed tokenizer/corpus.
**Do not treat scores from different corpora as interchangeable** (e.g. CRISP-500k **0.407753** vs overnight 200k **0.4309** vs synthetic **0.3682**).

| val_bpb | What it means |
|---------|---------------|
| > 4.0 | Model barely beats random — hasn't learned field structure yet |
| 1.5 – 4.0 | Early convergence — learning token distributions |
| 0.8 – 1.5 | Good — model understands normal telemetry patterns |
| 0.4 – 0.8 | Strong — implicit anomaly detector, approaching production use |
| **0.407753** | **← CRISP-500k `TIME_BUDGET` (7110 sessions, `crisp_zenodo_20260914T144906Z`); factual training metric only — not a public accuracy claim** |
| **0.4309** | **← CRISP overnight best on 200k subset (`73b1645`, exp 20); README fact only — not a public accuracy claim** |
| 0.458756 | ← CRISP pre-overnight 200k floor (`TIME_BUDGET` single run, 2026-09-14) |
| 0.4297 | ← synthetic Night 2 best (exp 13) |
| 0.3692 | ← synthetic exp 37–66 (focal loss + anomaly weighting) |
| 0.3691 | ← synthetic exp 70–113 (domain-aware loss tuning) |
| **0.3682** | **← synthetic smoke-era best (exp 114); separate table above — not comparable to CRISP** |
| < 0.35 | Excellent — deploy as zero-shot anomaly scorer |

The information-theoretic argument: minimizing val_bpb = minimizing KL(P_data ‖ P_model).
A model close to the true data distribution assigns high surprise to anomalous sequences automatically.
**The training objective IS the anomaly detection capability. No separate head. No labels.**

CRISP-500k **0.407753**, CRISP overnight 200k **0.4309** (and floor **0.458756**), and synthetic **0.3682** are logged in separate lanes/tables above and must not be mixed or marketed as a single accuracy story. See [`docs/public-accuracy-eval.md`](docs/public-accuracy-eval.md).

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
