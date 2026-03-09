# Prompt: Build Autonomous Observability Model Breeder using autoresearch-macos

You are an expert AI systems engineer with deep knowledge of observability (Splunk, AppDynamics, OpenTelemetry, Cisco networking telemetry), time-series anomaly detection, tiny language models, and agentic coding workflows.

Your task is to **produce a complete, ready-to-run fork/setup of https://github.com/miolini/autoresearch-macos** that transforms the original nanochat LLM research swarm into an **Autonomous Observability Model Breeder** for Cisco/Splunk/AppDynamics-style telemetry.

## Goal of the New System
Create a swarm of tiny, efficient models (starting from nanochat architecture) that **autonomously evolve overnight** to become specialists in:
- Compressing observability telemetry sequences while preserving root-cause signals
- Detecting anomalies / drift in AI-agent + infrastructure traces (latency spikes, cost explosions, GPU util anomalies, PII leaks in prompts, policy violations)
- Distinguishing real incidents from benign noise in distributed traces (AppD business transactions + ThousandEyes paths + Splunk logs + OTel spans)
- Generating plausible "what-if" synthetic failure patterns to stress-test detection

All of this must run autonomously on a single Apple Silicon Mac (MPS backend), using the exact autoresearch-macos mechanics:
- Agent edits **only train.py**
- Fixed **5-minute training runs**
- Metric = **val_bpb** (lower is better) — but we will reinterpret it meaningfully for observability sequences
- Git branching per experiment
- Agent guided entirely via **program.md** instructions

## Deliverables — Produce ALL of these in your response

1. **Modified program.md**  
   Full text content — this is the critical "research constitution" file that tells the agent what to do. Make it detailed, structured, and obsessive about observability goals. Include:
   - Role & mission
   - Success criteria (improve val_bpb on observability-like sequences)
   - Allowed kinds of changes to train.py (architecture, optimizer, attention, quantization awareness, custom loss components, etc.)
   - Forbidden changes (never touch prepare.py, don't break MPS compatibility)
   - Thinking style: chain-of-thought before proposing edits
   - Experiment ideas to try first (deeper temporal attention for traces, better handling of sparse events, adversarial robustness, etc.)
   - How to generate / mutate synthetic telemetry in code if needed

2. **Custom data preparation instructions**  
   Explain exactly how to modify (or extend) `prepare.py` **minimally** so that instead of TinyStories tokens we train on **observability telemetry sequences**.  
   Provide sample code snippets to add. Suggested format for sequences:
   [ts=2026-03-08T18:00:00Z] [src=AppD] [svc=payment-gateway] latency_ms=420 error=timeout trace_id=abc123 span_id=def456 http_status=500 gpu_util=0.92 prompt_tokens=120 response_tokens=45 drift_score=0.87
   [ts=2026-03-08T18:00:01Z] [src=ThousandEyes] path=internet→aws latency_ms=180 jitter_ms=12 packet_loss=0.3 ...

- Use a small public / synthetic dataset (~few MB) — e.g. generate synthetic traces or point to open OTel/JSON log samples.
- Keep BPE tokenizer training — it should learn observability tokens well (IP addresses, trace IDs, metric names, etc.).

3. **Baseline train.py modifications (initial seed)**  
Provide the minimal diff / code to make the very first run already somewhat observability-friendly:
- Increase context length if possible within memory (traces are longer-range)
- Suggestion for sliding-window causal attention if not already present
- Any small tweaks to batch size / learning rate known to work better on MPS

4. **Evaluation reinterpretation & hooks**  
Explain how val_bpb still makes sense here (next-token prediction on telemetry → good compression & understanding of patterns).  
Optional: suggest adding a secondary logged metric (e.g. synthetic anomaly recall) that the agent can read from logs but not directly optimize.

5. **Setup & Run Instructions**  
Step-by-step bash / terminal commands to:
- Clone miolini/autoresearch-macos
- Apply your changes (program.md + prepare.py snippets)
- Install & prepare
- Kick off the agent (example Claude prompt to start the loop)
- What to watch for overnight (~50–100 experiments)

6. **Demo / Success Visualization Ideas**  
Suggest how to read the git history + logs the next morning to see progress (e.g. val_bpb dropping = better compression/understanding of telemetry chaos).

## Constraints & Style Rules
- Stay **extremely faithful** to autoresearch-macos mechanics — no new files, no changing prepare.py logic fundamentally, agent **only edits train.py**.
- Keep everything runnable on Apple Silicon Mac (MPS) — mention memory / batch size caution.
- Make program.md **dense, hierarchical, and agent-friendly** (use markdown headings, numbered lists, examples).
- Output should be **copy-paste ready** — full file contents in fenced code blocks.
- Be ambitious: aim for the agent to discover better tiny architectures / tricks for observability that could seed real improvements in Splunk AppD Cisco agentic tools.

Now go — output the complete set of deliverables above.