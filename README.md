# Autonomous Observability Model Breeder (AOMB)

A self-improving AI research swarm that autonomously evolves tiny language models
specialized in Cisco / Splunk / AppDynamics observability telemetry.

Built on top of [autoresearch-macos](https://github.com/miolini/autoresearch-macos) —
runs entirely on Apple Silicon (MPS) with fixed 5-minute training experiments.

## What it does

The system trains a tiny GPT-variant (with RoPE, GQA, MuonAdamW, sliding window
attention) on synthetic observability telemetry sequences:

```
[ts=2026-03-08T18:00:00Z] [src=AppD] [svc=payment-gateway] latency_ms=420 error=timeout trace_id=a3f9 span_id=f1e2 http_status=500 gpu_util=0.92 drift_score=0.87
[ts=2026-03-08T18:00:01Z] [src=ThousandEyes] path=internet→aws-us-east-1 latency_ms=180 jitter_ms=12 packet_loss=0.003 bgp_changes=0 dns_ms=14
[ts=2026-03-08T18:00:02Z] [src=Splunk] host=web-03 level=ERROR svc=auth msg=token_expired latency_ms=9800 alert=true
[ts=2026-03-08T18:00:02Z] [src=OTel] trace_id=a3f9 span_id=b2c3 op=db.query svc=inventory-db duration_ms=390 status=ok
[ts=2026-03-08T18:00:03Z] [src=CiscoSDWAN] site=branch-07 link=mpls→inet latency_ms=55 loss=0.000 bw_util=0.71
```

A Claude agent autonomously runs experiments overnight, editing `train.py` to find
the architecture and hyperparameters that minimize `val_bpb` on this telemetry corpus.

**Why val_bpb?** Lower bits-per-byte means the model predicts normal telemetry better.
This creates a free implicit anomaly detector: anomalous sequences have high perplexity.

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Generate observability corpus (~21 parquet shards, ~150MB)
uv run python generate_observability_corpus.py

# 3. Train BPE tokenizer on telemetry vocab
uv run python prepare.py --num-shards 20

# 4. Smoke-test the baseline model (60 seconds)
timeout 60 uv run train.py

# 5. Start the agent loop (feed to Claude API or claude.ai)
# The agent reads program.md + current train.py and proposes improvements
# Run overnight for 50–100 experiments
```

## Files

| File | Purpose |
|------|---------|
| `train.py` | Model + training loop (agent edits this) |
| `prepare.py` | Tokenizer training + data loading (do not modify) |
| `program.md` | Agent research constitution (domain + experiment queue) |
| `generate_observability_corpus.py` | One-time synthetic telemetry corpus generator |
| `morning_report.py` | Next-morning progress summary from git log |
| `analysis.ipynb` | Experiment analysis notebook (from autoresearch-macos) |

## Architecture

The model is a sophisticated nanoGPT variant already featuring:
- **RoPE** (Rotary Positional Embeddings) — handles long sequences
- **GQA** (Grouped Query Attention) — memory efficient
- **Sliding window attention** — configurable per-layer pattern
- **MuonAdamW** — Muon optimizer for matrices, AdamW for embeddings
- **Value residual** (ResFormer-style) — alternating layers
- **Logit softcapping** — training stability

The agent explores: model depth, width, attention window patterns, learning rates,
schedule shape, and loss function modifications.

## Observability Sources Modeled

| Source | Events modeled |
|--------|----------------|
| AppDynamics | Business transactions, service health, agent traces |
| ThousandEyes | Network path latency, packet loss, BGP changes |
| Splunk | Structured logs, SIEM alerts, security events |
| OpenTelemetry | Distributed trace spans, service dependencies |
| Cisco SD-WAN | Branch link quality, QoS violations, bandwidth |
| AppD BizTxn | End-to-end transaction health snapshots |

## Morning Report

After an overnight run:
```bash
uv run python morning_report.py --plot
```

Prints a table of all experiments with val_bpb, delta, and the winning change.
Generates `overnight_progress.png` showing the convergence curve.

## Target val_bpb

| val_bpb | Meaning |
|---------|---------|
| > 1.5 | Early training, model learning field structure |
| 1.0–1.5 | Good — model understands normal telemetry patterns |
| 0.8–1.0 | Very good — strong implicit anomaly detector |
| < 0.8 | Excellent — production-ready zero-shot anomaly detector |

## License

Apache 2.0 — derived from [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos).
