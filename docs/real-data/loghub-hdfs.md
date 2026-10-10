# LogHub HDFS_v1: first real-data result

> **claim_status=`published`** (2026-09-27).
> On real Hadoop production logs, the model ranks anomalous blocks better than every hand-built check (AUROC 0.878 vs 0.822; PR-AUC 0.740 vs 0.605). At the best single threshold it's level with them (F1 0.787 vs 0.795). It is **not** state of the art on HDFS.

> **Update (2026-10-09).** Two things changed after this page was written. An end-of-session marker lifts the model to 0.964–0.974 AUROC, and a simple "too short" rule added to the baselines lifts the best hand-built check to **0.977 / PR-AUC 0.833**. So the model now *matches* the best simple check on HDFS; the "ranks better than every hand-built check" statement below held only against the baselines that existed then. See [`end-of-session-marker.md`](end-of-session-marker.md).

## Data

[LogHub HDFS_v1](https://github.com/logpai/loghub) (Xu et al., SOSP 2009; Zhu et al., ISSRE 2023; CC BY 4.0; [Zenodo](https://zenodo.org/records/8196385)). It's 11.2M log lines from a Hadoop cluster running benchmark workloads. 575,061 blocks are labelled Normal or Anomaly by the dataset authors' hand-crafted rules, and 2.93% are anomalous. One session = one block's log lines.

The data isn't vendored. To rebuild the exact sessions:

```bash
# download HDFS_v1.zip (187 MB) from Zenodo and unzip to ~/.cache/aomb-datasets/loghub/HDFS_v1
uv run python -m corpus.ingest.loghub_hdfs --input ~/.cache/aomb-datasets/loghub/HDFS_v1 \
  --out ~/.cache/aomb-datasets/loghub/hdfs_sessions.jsonl
# expected sessions_sha256: f445a5355ba1dcae1e12448d65a5685cabf63f5a9705193330ba8a8ef122f18b
uv run python -m eval.in_domain --sessions ~/.cache/aomb-datasets/loghub/hdfs_sessions.jsonl \
  --seeds 0..4 --train-seconds 120 --train-chunk-lines 24 --out-dir <out>
```

## Protocol

| Item | Value |
|------|-------|
| Split | Temporal (DeepLog-style). **Train:** the first 5,000 *normal* blocks in first-appearance order. **Eval:** 10,000 blocks sampled (seed 0) from all blocks after them, at the natural rate: 9,729 normal, 271 anomalous |
| Normalised before anything sees the text | Block ids, IPs/ports, job and task ids (they embed a timestamp), **job output directories** (workload names that change over time), part numbers, data sub-directories, thread ids |
| Masked from model scoring | Timestamps (as in the lab evals) |
| Model | Tokenizer + `train.py` GPT fit on the 5,000 training blocks only (split into ≤24-line chunks, none cropped; context 960 tokens); 120 s × 5 seeds, MPS |
| Code | `corpus/ingest/loghub_hdfs.py`, `eval/in_domain.py` |

## Results (5 seeds)

| Method | Kind | AUROC | PR-AUC | Best F1 (P / R) |
|--------|------|-------|--------|-----------------|
| Session length | baseline | 0.569 | 0.123 | |
| Error/latency rule (no ERROR lines or spans in HDFS) | baseline | 0.500 | 0.027 | |
| Event-multiset novelty | baseline | 0.531 | 0.029 | |
| Template novelty | baseline | 0.791 | 0.580 | 0.730 (0.975 / 0.583) |
| **Value rarity (best baseline)** | baseline | 0.822 | 0.605 | **0.795** (0.872 / 0.731) |
| Model, masked session mean | model | **0.878 ± 0.022** | 0.631 ± 0.014 | 0.632 ± 0.013 |
| Model, max per-event | model | 0.855 ± 0.003 | **0.740 ± 0.018** | 0.784 ± 0.000 (0.973 / 0.656) |
| Model, per-field | model | 0.802 ± 0.032 | 0.714 ± 0.017 | 0.787 ± 0.011 (0.876 / 0.715) |

Best F1 uses the single best threshold chosen on the eval set itself, so it's an upper bound for each method and is shown only to compare with published work, which reports F1.

## What it shows

1. **On ranking, the model beats every hand-built check.** Every masked model variant has higher PR-AUC than the best baseline (0.63–0.74 vs 0.61). The masked session mean has the best AUROC (0.878).
2. **At a single operating point it's level with them, not better.** Best F1 is 0.787 for the model against 0.795 for value rarity.
3. **It's well short of published HDFS detectors.** Papers report F1 around 0.9 or higher (DeepLog: 0.96). Protocols differ (they parse log keys, and splits vary), so the numbers aren't directly comparable. Even so, a general, unconfigured model isn't matching specialised log-anomaly methods here.
4. **The best scoring variant depends on the data.** Per-field scoring won on the lab; on HDFS the whole-session and per-event scores rank better. There's no single score to recommend yet.
5. **The misses are anomalies by absence.** The lowest-ranked anomalies are 2-line blocks (allocated, then nothing), where the lifecycle simply stops. Next-token surprise doesn't score what *didn't* happen. An explicit end-of-session token is the obvious next thing to try.

## Correction during this eval

A first run scored 0.957 AUROC. Looking at the top-ranked anomalies showed they all came from job directories (`/user/root/randtxt4/…`) that first appear after the training window, so a new *workload name* looked like an anomaly. Job directories are now normalised; the confound was worth about 0.08 AUROC. After the fix, 158 of 271 anomalies still contain a log template never seen in training, against 4 of 9,729 normal blocks. That's why template novelty is a strong baseline here.

Reports: [`reports/public-accuracy/loghub-hdfs-in-domain-20260927/`](../../reports/public-accuracy/loghub-hdfs-in-domain-20260927/).
