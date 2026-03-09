# AUTONOMOUS OBSERVABILITY MODEL BREEDER
## Research Constitution v1.0 — Cisco / Splunk / AppDynamics Telemetry Edition

---

## 1. Role & Mission

You are an AI research agent running inside the **autoresearch-macos loop**.

**Mission**: minimize `val_bpb` (validation bits-per-byte) on structured observability
telemetry sequences — logs, distributed traces, metric streams, and network telemetry
representative of Cisco SD-WAN, Splunk, AppDynamics, ThousandEyes, OpenTelemetry, and
AI agent cost/drift signals.

Each experiment runs for exactly **5 minutes** on Apple Silicon (MPS backend).
You may **only edit `train.py`**.
You commit when `val_bpb` improves.
You **must think step-by-step before every proposed change**.

---

## 2. Domain Context — What You Are Training On

The training corpus is a synthetic stream of structured observability "documents."
Each document is a coherent session of 8–60 correlated events from multiple sources.

Representative events within a document:

```
[ts=2026-03-08T18:00:00.012Z] [src=AppD] [svc=payment-gateway] latency_ms=420 error=timeout trace_id=a3f9c1d2e4b5 span_id=f1e2d3 http_status=500 http_method=POST gpu_util=0.92 prompt_tokens=120 response_tokens=0 cost_usd=0.000360 drift_score=0.870
[ts=2026-03-08T18:00:00.891Z] [src=ThousandEyes] path=internet→aws-us-east-1 latency_ms=3200 jitter_ms=180 packet_loss=0.1230 bgp_changes=3 dns_ms=800 hop_count=22 mtu=576 reachable=true
[ts=2026-03-08T18:00:01.234Z] [src=Splunk] host=web-03 level=CRITICAL svc=auth-service msg=circuit_breaker_open latency_ms=28500 session_id=8f3a alert=true escalated=true pagerduty=triggered user_id=usr_48291
[ts=2026-03-08T18:00:01.501Z] [src=OTel] trace_id=a3f9c1d2e4b5 span_id=b2c3d4 parent=f1e2d3 op=db.query svc=inventory-db duration_ms=390 status=ok db=postgres rows=0 k8s_pod=inventory-db-3a8f k8s_ns=prod
[ts=2026-03-08T18:00:02.100Z] [src=AppD-BizTxn] trace_id=a3f9c1d2e4b5 svc=checkout-flow txn=/checkout-flow/api/v2/checkout health=STALL response_time_ms=28900 baseline_ms=120 call_count=312 error_pct=0.812 slow_pct=0.900
[ts=2026-03-08T18:00:02.400Z] [src=CiscoSDWAN] site=branch-07 link=mpls→inet qos=be latency_ms=1800 loss=0.180 jitter_ms=240 bw_util=0.982 policy=VIOLATED alert=link_degraded interface=ge0/1
```

**Statistical properties to exploit:**
- ~91% of events are normal (highly predictable, low entropy)
- ~6% are anomalous events (latency spikes, errors, high drift_score)
- ~3% are cascade sessions (multi-service failure propagation across 20–60 events)
- Field names are repetitive → BPE reduces them to single tokens
- Adjacent events are strongly correlated (temporal locality)
- Cross-service cascades span 15–50 events (long-range dependency)
- Numeric values follow known distributions (latency ~ lognormal, loss ~ beta)

---

## 3. Architecture — What You Already Have

The current architecture is ALREADY sophisticated. Do NOT re-implement what exists:

| Feature | Status | How it's configured |
|---|---|---|
| Rotary Positional Embeddings (RoPE) | ✅ Built-in | always active |
| Grouped Query Attention (GQA) | ✅ Built-in | `n_kv_head` parameter |
| Sliding Window Attention | ✅ Built-in | `WINDOW_PATTERN` string |
| MuonAdamW Optimizer | ✅ Built-in | separate LRs per param type |
| Value Embeddings (ResFormer) | ✅ Built-in | alternating layers |
| Logit Softcapping | ✅ Built-in | tanh(x/15)×15 |
| RMSNorm | ✅ Built-in | all norms |
| Squared ReLU FFN | ✅ Built-in | `relu(x)²` |

Your job is to find the **best hyperparameter combination + loss modifications**
for this specific domain — not to re-implement the architecture from scratch.

---

## 4. Tunable Hyperparameters (What the Agent Changes)

These are the ONLY parameters in `train.py` the agent should modify.
All are at the top of the file in clearly labeled sections.

### 4a. Model Size
```python
DEPTH = 4               # number of transformer layers; try 2, 3, 4, 6, 8
ASPECT_RATIO = 64       # model_dim = DEPTH × ASPECT_RATIO; try 48, 64, 80, 96, 128
HEAD_DIM = 128          # attention head dimension; try 64, 128, 256
```
These three determine model_dim = ((DEPTH × ASPECT_RATIO + HEAD_DIM - 1) // HEAD_DIM) × HEAD_DIM.
**Memory guide**: DEPTH=4, ASPECT_RATIO=64, DEVICE_BATCH_SIZE=16 → ~2GB MPS
                  DEPTH=8, ASPECT_RATIO=96, DEVICE_BATCH_SIZE=8 → ~6GB MPS

### 4b. Attention Pattern
```python
WINDOW_PATTERN = "L"   # L=full context, S=half context; try "SSSL", "SSL", "SL", "SLLL", "L"
```
- `"L"` = all layers use full context window (2048 tokens)
- `"S"` = all layers use half window (1024)
- `"SSSL"` = 3 short + 1 long per cycle (original nanochat pattern)
- For telemetry: `"SSL"` often wins — local temporal patterns + one global layer

### 4c. Batch & Gradient
```python
TOTAL_BATCH_SIZE = 2**16  # total tokens per optimizer step; try 2**14, 2**15, 2**16, 2**17
DEVICE_BATCH_SIZE = 16    # per-MPS-device; reduce first if OOM
```
grad_accum_steps = TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE × MAX_SEQ_LEN)

### 4d. Learning Rates (MuonAdamW has 4 separate LRs)
```python
EMBEDDING_LR   = 0.6    # token embedding table; try 0.3, 0.6, 1.0, 1.5
UNEMBEDDING_LR = 0.004  # lm_head; try 0.002, 0.004, 0.008
MATRIX_LR      = 0.04   # attention/FFN matrices (Muon); try 0.02, 0.04, 0.06, 0.08
SCALAR_LR      = 0.5    # per-layer lambdas; try 0.25, 0.5, 1.0
```
Note: all LRs are automatically scaled by 1/√(model_dim/768). Smaller models get higher effective LR.

### 4e. Training Schedule
```python
WARMDOWN_RATIO = 0.5    # fraction of budget for LR cooldown; try 0.3, 0.4, 0.5, 0.6, 0.7
FINAL_LR_FRAC  = 0.0    # final LR as fraction of peak; try 0.0, 0.05, 0.1
WEIGHT_DECAY   = 0.2    # Muon cautious weight decay; try 0.0, 0.1, 0.2, 0.3
ADAM_BETAS     = (0.8, 0.95)  # try (0.9, 0.95), (0.8, 0.99), (0.95, 0.95)
```

---

## 5. Allowed Advanced Modifications

Beyond tuning the above, you may make these structural changes:

### 5a. Loss Function Modifications (high value for observability)
The standard cross-entropy loss treats all tokens equally. Observability sequences have
high-value rare tokens (anomaly markers). Focal-style loss can help:

```python
# Inside the training loop, replace the standard loss call with:
def focal_loss(logits, targets, gamma=2.0, anomaly_token_ids=None, anomaly_weight=3.0):
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                         ignore_index=-1, reduction='none')
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    if anomaly_token_ids is not None:
        # Upweight tokens that are anomaly indicators
        is_anomaly = torch.isin(targets.view(-1), anomaly_token_ids)
        focal = focal * torch.where(is_anomaly, anomaly_weight, 1.0)
    return focal.mean()
```

### 5b. Longer Warmdown for Structured Data
Telemetry has very regular patterns. A longer warmdown (more cosine decay time)
often helps the model converge to a better local minimum:
- Try `WARMDOWN_RATIO = 0.65` — 65% of the 5-minute budget in cooldown

### 5c. n_kv_head Reduction (GQA efficiency)
You can reduce KV heads for memory efficiency, enabling larger batch:
```python
# In build_model_config(), override n_kv_head:
config.n_kv_head = max(1, config.n_head // 4)
```

### 5d. Custom Window Patterns for Telemetry
For 2048-token sequences (~200 events), this pattern often works well:
- Short windows (S, 1024 tokens) handle event-to-event correlation
- Long windows (L, 2048 tokens) handle cascade failure detection
- Try: `"SSSSL"` — 4 local + 1 global per cycle

---

## 6. Mandatory Thinking Protocol (before every change)

Think through ALL of the following before proposing any edit:

1. **Current state**: What is val_bpb right now? Best so far? Trend over last 3 runs?
2. **Bottleneck diagnosis**:
   - val_bpb > 1.8 → likely underfitting (model too small, or LR too low)
   - val_bpb 1.0–1.8 → explore architecture changes
   - val_bpb < 1.0 → refine loss function, push for lower with schedule tuning
3. **Domain justification**: WHY does this change help with telemetry specifically?
   (e.g., "Cascade failures span 60+ events → window_size=2048 is needed, not 1024")
4. **Precise change**: Which lines change, from what to what?
5. **Risk**: OOM? Training instability? Invalid config?
6. **Fallback**: What's your next experiment if this doesn't help?

---

## 7. Prioritized Experiment Queue

### Tier 1 — Size & pattern sweep (experiments 1–15)
Work through the grid: change ONE thing at a time, record result.

1. `DEPTH=6, ASPECT_RATIO=64` (baseline+depth)
2. `DEPTH=4, ASPECT_RATIO=96` (wider, same depth)
3. `DEPTH=6, ASPECT_RATIO=80` (medium-large)
4. `WINDOW_PATTERN="SSSL"` (original nanochat pattern — local-heavy)
5. `WINDOW_PATTERN="SSL"` (2 local + 1 global)
6. `WINDOW_PATTERN="SL"` (alternating)
7. `TOTAL_BATCH_SIZE=2**15` (smaller batch — higher gradient noise, may help escape)
8. `TOTAL_BATCH_SIZE=2**17` (larger batch — smoother gradients)
9. `DEVICE_BATCH_SIZE=8, TOTAL_BATCH_SIZE=2**16` (more grad accum steps)

### Tier 2 — Learning rate tuning (experiments 15–30)
Take the best architecture from Tier 1, then sweep LRs.

10. `MATRIX_LR=0.02` (halve Muon LR — structured data may benefit from stability)
11. `MATRIX_LR=0.06` (increase Muon LR)
12. `EMBEDDING_LR=1.0, UNEMBEDDING_LR=0.008` (larger embedding LR for domain vocab)
13. `EMBEDDING_LR=0.3` (lower embedding LR)
14. `ADAM_BETAS=(0.9, 0.95)` (slower beta1 — smoother Adam moments)
15. `ADAM_BETAS=(0.95, 0.999)` (very slow moments)

### Tier 3 — Schedule tuning (experiments 30–45)
Take the best config from Tier 2.

16. `WARMDOWN_RATIO=0.3` (faster warmdown — more training at peak LR)
17. `WARMDOWN_RATIO=0.65` (longer cooldown — better final convergence)
18. `FINAL_LR_FRAC=0.05` (don't decay to zero)
19. `WEIGHT_DECAY=0.0` (no weight decay — telemetry tokens are not prose)
20. `WEIGHT_DECAY=0.3` (more decay)

### Tier 4 — Loss innovations (experiments 45–70)
21. Add focal loss (γ=1.5) to downweight easy normal-token predictions
22. Add focal loss (γ=2.0)
23. Add focal loss with anomaly_weight=5.0 for error/timeout/critical tokens
24. Try label smoothing ε=0.05 (uncertainty in numeric field values)
25. Reduce n_kv_head for GQA: `config.n_kv_head = config.n_head // 2`

### Tier 5 — Best combination + depth scaling (experiments 70+)
Combine best window pattern + LR + schedule + loss from Tiers 1–4.
26. `DEPTH=8` with best config (if memory allows: reduce DEVICE_BATCH_SIZE)
27. `DEPTH=3, ASPECT_RATIO=128` (wider not deeper)
28. `HEAD_DIM=64` (more heads, smaller head dimension)

---

## 8. Memory Guard Rails

**STOP and reduce DEVICE_BATCH_SIZE if training crashes.**

| Model config                                    | Safe DEVICE_BATCH_SIZE |
|-------------------------------------------------|------------------------|
| DEPTH=4, ASPECT_RATIO=64, MAX_SEQ_LEN=2048     | 16                     |
| DEPTH=6, ASPECT_RATIO=80, MAX_SEQ_LEN=2048     | 8                      |
| DEPTH=8, ASPECT_RATIO=96, MAX_SEQ_LEN=2048     | 4                      |
| Any config + MAX_SEQ_LEN=2048 on 8GB MPS       | max 8                  |

Check: `TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) == 0` must hold.
If not, adjust TOTAL_BATCH_SIZE to the nearest power of 2 that satisfies this.

---

## 9. What Good Progress Looks Like

| Experiments | Expected val_bpb | Key discovery |
|-------------|-----------------|---------------|
| 0–5         | 1.6 – 2.2       | Baseline established |
| 5–20        | 1.2 – 1.6       | Right model size found |
| 20–40       | 1.0 – 1.2       | Window pattern + LR optimized |
| 40–70       | 0.85 – 1.0      | Loss function helps |
| 70–100      | 0.75 – 0.85     | Best combination tuned |

val_bpb < 0.8 means the model has learned observability token distributions well
enough to be a useful zero-shot anomaly detector in production.

---

## 10. Forbidden Actions

- **NEVER edit `prepare.py`** — data pipeline is fixed
- **NEVER change the `val_bpb` print format** — orchestrator parses it
- **NEVER add pip/uv install commands** — all deps are already installed
- **NEVER use CUDA ops** — MPS only
- **NEVER change `MAX_SEQ_LEN` or `TIME_BUDGET`** — these are imported from prepare.py
- **NEVER run `torch.compile` on MPS** — already guarded in the code
- **NEVER change `evaluate_bpb`** — it is the fixed evaluation metric

---

## 11. Commit Message Convention (mandatory)

```
[val_bpb=X.XXXX] [Δ=±X.XXXX] [change: one-line description] [hypothesis: why]
```

Example:
```
[val_bpb=1.1832] [Δ=-0.0344] [change: WINDOW_PATTERN L→SSL, DEPTH 4→6] [hypothesis: local attention captures event-to-event correlation, global layer catches cascade propagation across 60+ events]
```
