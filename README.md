# AOMB — Autonomous Observability Model Breeder

**An AI research agent that improves a small language model for your telemetry while you sleep.**

[![stranger-verify](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml/badge.svg)](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml)
[![stranger-demo](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-demo.yml/badge.svg)](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-demo.yml)
[![diptych-adapter-gate](https://github.com/pandeyaby/AOMB/actions/workflows/diptych-adapter-gate.yml/badge.svg)](https://github.com/pandeyaby/AOMB/actions/workflows/diptych-adapter-gate.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Open in Codespaces](https://img.shields.io/badge/Open%20in-Codespaces-black?logo=github)](https://codespaces.new/pandeyaby/AOMB)

![Traces → small next-token Infrastructure Language Model → anomaly via surprise; Apple Silicon research loop](docs/assets/aomb-readme-hero.png)

AOMB trains an **Infrastructure Language Model (ILM)**: a small GPT that learns to predict the next token of observability telemetry (traces, logs, network and APM events). A model that predicts *normal* traffic well is surprised by *abnormal* traffic, so the surprise score (bits-per-byte) is an anomaly signal. That means no rules, no thresholds, and no labels.

The model isn't tuned by hand. An LLM agent (Claude) runs the research loop overnight on a Mac. It proposes a change to `train.py`, trains for 5 minutes on Apple Silicon, keeps the change if validation loss improved and reverts it if not. It repeats this dozens of times, and every improvement becomes a git commit. In the morning you get a report and a better model.

It's a domain-specific fork of [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch), via [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos).

> **Status: research prototype.** Training results on public real-world traces are measured and reproducible (below). Anomaly-detection *accuracy* (AUROC on labeled incidents) is **not yet published**. See [Claims & reproducibility](#claims--reproducibility).

---

## Results so far

All numbers are `val_bpb` (validation bits-per-byte, **lower is better**). This measures how well the model predicts held-out telemetry. Numbers from different datasets are **not comparable** with each other.

| Dataset | Setup | `val_bpb` | Details |
|---------|-------|-----------|---------|
| **Uber CRISP** (real Jaeger traces, 200k spans) | Single 5-min run → **20 overnight agent experiments** | 0.4588 → **0.4309** (−6.1%) | [`docs/crisp-val-bpb-baseline.md`](docs/crisp-val-bpb-baseline.md) |
| **Uber CRISP** (500k spans) | Single 5-min run, no agent | **0.4078** | [`docs/crisp-val-bpb-baseline.md`](docs/crisp-val-bpb-baseline.md) |
| **Uber Tale of Errors** (200k-span capped subset) | Single 5-min run | **1.3795** | [`docs/tale-val-bpb-baseline.md`](docs/tale-val-bpb-baseline.md) |
| Synthetic smoke corpus (first release) | 120 overnight experiments | 0.4372 → 0.3682 (−15.8%) | Legacy; used in the original write-up |

The first release was trained on a synthetic corpus, and that's where the 15.8% improvement quoted in the [original write-up](https://medium.com/@pandeyaby/i-let-an-ai-improve-itself-overnight-heres-what-i-woke-up-to-6db1905fc212) comes from. The project has since moved to public, real-world traces from Uber (CRISP and Tale of Errors).

**Does surprise actually find anomalies?** Run `uv run python demo_anomaly.py`. It trains briefly, then scores held-out sessions, and anomalous and cascade-failure sessions score higher bits-per-byte than normal ones. The walkthrough is in [`docs/anomaly-story.md`](docs/anomaly-story.md). Labeled lab evaluation is in progress in [`docs/lab/`](docs/lab/).

---

## Try it

### In 60 seconds, with no Mac and no API keys

Click **[Open in Codespaces](https://codespaces.new/pandeyaby/AOMB)**, or run it on any Linux or macOS machine:

```bash
git clone https://github.com/pandeyaby/AOMB.git && cd AOMB
uv sync
STRANGER_FAST=1 ./scripts/stranger_demo.sh
```

This runs the public test gates on CPU: the fixture pipeline, the scoring harness, and the DIPTYCH probe checks. It proves the code works end to end. It does **not** train a production model. The same checks run in CI on every push ([`stranger-verify`](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml)). More detail: [`docs/stranger-demo.md`](docs/stranger-demo.md).

### See the anomaly signal (~5 minutes, CPU is fine)

```bash
uv run python generate_observability_corpus.py   # small synthetic corpus
uv run python prepare.py --num-shards 20         # tokenizer + shards
uv run python demo_anomaly.py                    # train briefly, score normal vs anomalous sessions
```

### Run the overnight agent (Apple Silicon Mac)

```bash
./scripts/product_mac_smoke.sh     # check that MPS training works (no keys needed)

export AOMB_ANTHROPIC_API_KEYS=sk-ant-...
caffeinate -i uv run python agent_loop.py >> logs/agent_loop.log 2>&1 &

# next morning
uv run python morning_report.py --plot
```

---

## How it works

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT LOOP                           │
│                                                         │
│  program.md + train.py + git log                        │
│         │                                               │
│         ▼                                               │
│   Claude ──► proposed train.py                          │
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

Three files define the system:

| File | Who changes it | What it is |
|------|----------------|------------|
| `prepare.py` | Nobody (frozen) | Data pipeline, tokenizer, and the `evaluate_bpb` metric. It's frozen so scores stay comparable across experiments |
| `train.py` | The agent | Model architecture and training loop. This is the only file the agent edits |
| `program.md` | You, rarely | The agent's research brief: domain context, experiment ideas, constraints |

### Why `val_bpb` is the anomaly detector

```
val_bpb = total_nats / (log(2) × total_bytes)
```

Minimizing `val_bpb` minimizes the gap between the model's predictions and the real distribution of normal telemetry. The *same* quantity computed on a new session is its surprise score. So a lower training `val_bpb` means a sharper sense of normal, and a better anomaly signal. There's no separate detection head and no labels.

### The model

It isn't vanilla nanoGPT. The baseline the agent starts from includes:

| Component | Purpose |
|-----------|---------|
| RoPE | Rotary (relative) position embeddings |
| Grouped Query Attention | Memory-efficient attention |
| Sliding-window pattern | Full or half context per layer |
| Muon + AdamW | Muon for matrices, AdamW for embeddings |
| Value embeddings | ResFormer-style residual on alternating layers |
| Logit softcapping | `tanh(x/15)×15` |
| RMSNorm | Everywhere, no bias |

The agent explores depth, width, head size, attention windows, learning rates, schedules, batch size, and loss functions. Focal loss was an early breakthrough. See the top of `train.py`.

---

## Train on real data

### Uber CRISP (recommended)

Public Jaeger traces from Uber's CRISP dataset (~2.3 GB download).

```bash
uv run python -m corpus.ingest.fetch_crisp --download
uv run python -m corpus.ingest.build_shards \
  --adapter crisp_zenodo \
  --input ~/.cache/autoresearch/corpus-v1/crisp/extracted \
  --num-train-shards 8 --write-val-shard
uv run python prepare.py --num-shards 8
uv run python train.py
```

### Tale-scale (Uber Tale of Errors — train lane)

Much larger public dataset (CC BY 4.0, hundreds of GB), streamed and capped so it fits on a laptop. To try it on a fixture without downloading: `./scripts/tale_scale_smoke.sh`. Docs: [`docs/tale-scale.md`](docs/tale-scale.md).

**Capped subset measured** (Mac run, `max_spans=200000`): **`val_bpb=1.379520`**, `claim_status=measured_not_published`. This is train fitness only, not AUROC, and not a published accuracy claim. It isn't directly comparable to CRISP. Card: [`reports/tale-capped/measured_capped_200k.json`](reports/tale-capped/measured_capped_200k.json) · write-up: [`docs/tale-val-bpb-baseline.md`](docs/tale-val-bpb-baseline.md) · summary: [`docs/public-wins.md`](docs/public-wins.md) · one-liner: `./scripts/public_wins_tale_line.sh`.

### Your own telemetry

Point the scorer at an OTLP JSONL, Jaeger, or parquet dump and get per-session surprise scores:

```bash
./scripts/byo_score.sh /path/to/dump            # dry run: parse + validate
./scripts/byo_score.sh /path/to/dump --train-seconds 120
```

See [`docs/byo-and-scorer.md`](docs/byo-and-scorer.md).

---

## Configuration

**Agent loop:** set `MAX_EXPERIMENTS`, `CLAUDE_TIMEOUT`, `TRAIN_TIMEOUT` and `CLAUDE_MODEL` at the top of `agent_loop.py`. Environment variables:

```bash
AOMB_ANTHROPIC_API_KEYS=sk-ant-...   # comma-separated; falls back to the Claude Code CLI if unset
AOMB_OPENAI_API_KEYS=sk-proj-...     # optional fallback provider
AOMB_CLAUDE_MODELS=sonnet            # or: opus, haiku, gpt-4o-mini
```

Every improvement is a git commit. Results are pushed every 10 successes. To stop: `kill $(cat logs/agent_loop.pid)`.

**Scheduled morning report (launchd):**

```bash
sed "s|AOMB_DIR|$(pwd)|g" com.aomb.morning-report.plist.template \
  > ~/Library/LaunchAgents/com.aomb.morning-report.plist
launchctl load ~/Library/LaunchAgents/com.aomb.morning-report.plist
```

---

## Claims & reproducibility

This project keeps three kinds of evidence separate and never mixes their numbers:

| Lane | Data | What it can claim |
|------|------|-------------------|
| **1. Train** | Public real traces (Uber CRISP, Tale scale) | `val_bpb` training fitness only. These datasets have no incident labels, so no detection accuracy |
| **2. Lab** | Private Docker microservice stack with injected faults | Ranking accuracy (AUROC). **Not published yet**, see [`docs/lab/`](docs/lab/) and [`docs/public-accuracy-eval.md`](docs/public-accuracy-eval.md) |
| **3. Public fixture card** | Tiny synthetic pack, runs in CI | Proves the scoring harness works and beats trivial baselines. Not a real-world accuracy claim ([`docs/public-ranking-card-v1.md`](docs/public-ranking-card-v1.md)) |

**What CI proves** ([`stranger-verify`](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-verify.yml), [`stranger-demo`](https://github.com/pandeyaby/AOMB/actions/workflows/stranger-demo.yml)): the ingest → shard → score pipeline and the fixture ranking card run end to end on CPU, and the DIPTYCH probes pass. **What it doesn't prove:** production accuracy, MPS training, or the overnight agent, which needs a Mac and API keys. Compute details: [`docs/compute-paths.md`](docs/compute-paths.md) (Apple MPS is the product path; there's no CUDA claim yet).

Index of everything an outsider can check: [`docs/public-wins.md`](docs/public-wins.md) · 60-second cheatsheet: [`docs/stranger-60s.md`](docs/stranger-60s.md).

### Companion: DIPTYCH

[DIPTYCH](https://github.com/pandeyaby/DIPTYCH) is a separate project that grades model *calibration* with paired probes: two runs that share every input except one controlled perturbation. AOMB emits probes for all 8 DIPTYCH operators (`./scripts/run_diptych_full8.sh`, output in `diptych-probes/`), and the [`diptych-adapter-gate`](https://github.com/pandeyaby/AOMB/actions/workflows/diptych-adapter-gate.yml) workflow checks them on every push. Details: [`docs/paired-probes/`](docs/paired-probes/).

![AOMB breed/score → fixtures → DIPTYCH paired probes](docs/images/aomb-diptych-architecture.svg)

---

## Repository layout

```
agent_loop.py        overnight research agent (Claude proposes, train.py runs, git keeps winners)
train.py             model + training loop — the file the agent edits
prepare.py           frozen data pipeline, tokenizer, and val_bpb metric
program.md           research brief given to the agent
morning_report.py    overnight summary + progress plot
demo_anomaly.py      train briefly, then score normal vs anomalous sessions
score_session.py     score a telemetry dump (used by scripts/byo_score.sh)
corpus/ingest/       dataset fetchers + adapters (CRISP, Tale of Errors, OTLP, BYO)
eval/                ranking card, scoring CLI, DIPTYCH emitters
lab/                 Docker fault-injection lab for labeled evaluation
scripts/             one-command entry points (demo, verify, Mac smoke, Tale pipeline)
docs/                design notes, baselines, and reproducibility docs
tests/               pytest suite (run: uv run pytest)
```

---

## Requirements

- **Demo and CI path:** Linux or macOS, Python 3.10+, [`uv`](https://docs.astral.sh/uv/). CPU only, no API keys.
- **Overnight agent:** a Mac with Apple Silicon (M1 or later), an Anthropic API key or the Claude Code CLI, and ~500 MB of disk (more for the CRISP and Tale datasets). Bridge from CPU to MPS: [`docs/product-mac-path.md`](docs/product-mac-path.md).

## Contributing

Issues and PRs are welcome. Start with [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/contributing-stranger.md`](docs/contributing-stranger.md). Run `uv run pytest` before opening a PR. Please don't modify `prepare.py`, because it's the fixed measuring stick, and don't add accuracy numbers that the harness didn't produce.

Security: [`SECURITY.md`](SECURITY.md) · Support: [`SUPPORT.md`](SUPPORT.md) · Code of conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)

## Lineage

```
karpathy/autoresearch            original (H100 / NVIDIA)
└── miolini/autoresearch-macos   macOS / MPS port
    └── pandeyaby/AOMB           observability telemetry + fully autonomous overnight loop
```

## Citation & license

If you use AOMB, please cite it via [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button).

MIT. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for upstream attributions. Built by [Abhinav Pandey](https://github.com/pandeyaby).
