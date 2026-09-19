# Compute paths — honest hardware claims

**This PR / doc does not run a GPU job.** It is the **plan gate** for what is real today vs what a future CUDA claim would require. No invented CUDA benchmarks. No invented AUROC.

`prepare.py` remains **sacred** — never edited to “enable CUDA.”

---

## Today (real)

| Path | Hardware | Entry | Honest claim |
|------|----------|-------|--------------|
| **Stranger demo / verify** | **CPU** Linux / GitHub Actions / Codespaces | `./scripts/stranger_demo.sh`, `./scripts/stranger_verify.sh` | Public gates only (full-8 + `gate_axis_mutate`, ranking-card harness smoke). **No MPS. No API keys. No CUDA.** |
| **Product overnight breed / `TIME_BUDGET` train** | **Apple Silicon MPS** (documented Mac path) | `train.py` / agent loop / CRISP train as in README + [`crisp-val-bpb-baseline.md`](crisp-val-bpb-baseline.md) | Train fitness **`val_bpb`** (and overnight breed story) on MPS — **not** a CUDA claim, **not** lab AUROC |

**Do not invent:** CUDA wall-clock numbers, CUDA `val_bpb`, or any AUROC (lab stays `not_published` until a separate published checklist is green).

Related: [`stranger-60s.md`](stranger-60s.md) · [`public-wins.md`](public-wins.md) · [`stranger-demo.md`](stranger-demo.md) · [`stranger-verify.md`](stranger-verify.md) · [`public-ranking-card-v1.md`](public-ranking-card-v1.md) · [`lab/`](lab/) · [DIPTYCH](https://github.com/pandeyaby/DIPTYCH).

---

## Future CUDA claim checklist (requirements only)

A public CUDA path may be claimed **only after** all boxes below are filled with **measured** evidence on a named run. Empty boxes = **no claim**. This section is a gate, not a result.

### 1. Hardware

- [ ] GPU class named (e.g. consumer 24GB / datacenter A10 / A100 — pick one and stick to it for the claim)
- [ ] Driver / CUDA / torch build recorded (versions in the run log)
- [ ] Same corpus / shard recipe as the MPS baseline being compared (or explicitly “not comparable”)

### 2. Script entrypoint

- [ ] Wrapper script named (e.g. `scripts/cuda_train.sh` or similar) that calls existing `train.py` / product train path
- [ ] **`prepare.py` untouched** — sacred; wrapper only
- [ ] Env flags documented (`CUDA_VISIBLE_DEVICES`, torch device selection) — no silent fallback to invent numbers

### 3. CI job shape

- [ ] Runner chosen: **self-hosted** GPU machine **or** cloud GPU Actions runner (named)
- [ ] Workflow file path recorded; manual `workflow_dispatch` first, then optional push triggers
- [ ] Artifact / log retention enough to cite a run URL

### 4. Spend + approval

- [ ] Spend ceiling set (USD or GPU-hours) before the run
- [ ] **Abhinav yes** recorded (explicit go-ahead for that spend)
- [ ] Run stopped or not started if ceiling would be exceeded

### 5. Allowed metrics (when claiming)

| Allowed after checklist green | Never from CUDA alone |
|------------------------------|------------------------|
| Wall-clock for a fixed `TIME_BUDGET` / step budget on the **same** corpus recipe | Lab AUROC (`not_published` until lab publish checklist is green) |
| `val_bpb` train fitness on that corpus (same honesty as MPS CRISP docs) | Production / field ranking accuracy |
| Throughput notes (tokens/sec) if measured on that run | Invented or cross-hardware “speedup” without both runs cited |

**Lab AUROC:** still forbidden until the lab publish path is independently green — CUDA does not unlock it.

---

## Explicit non-claims for this doc

- No GPU (CUDA or otherwise) was run to produce this file.
- No CUDA benchmark numbers appear here on purpose.
- Product overnight / CRISP `TIME_BUDGET` today = **MPS**, not CUDA.
- Stranger paths today = **CPU** only.
