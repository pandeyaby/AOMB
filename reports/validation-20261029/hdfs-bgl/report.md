# HDFS / BGL validation (blinded re-measurement)

Repo main @ b5538b7. All compute run under run_locked.sh. Device: MPS (fixture trainer).

**Blinding note:** my context auto-loaded the user's memory file (~/.claude/.../MEMORY.md), which contains previously published HDFS/BGL numbers. I did not open it deliberately, but I was not fully blind. The numbers below come from my own runs only.

## HDFS

```
uv run python -m corpus.ingest.loghub_hdfs --input ~/.cache/aomb-datasets/loghub/HDFS_v1 --out /private/tmp/claude-501/-Users-abhinavpandey-Documents-GitHub-AOMB/93ef5deb-e083-4dcd-9de4-3b697e12752e/scratchpad/validation/hdfs-bgl/hdfs_sessions.jsonl
uv run python -m eval.in_domain --sessions /private/tmp/claude-501/-Users-abhinavpandey-Documents-GitHub-AOMB/93ef5deb-e083-4dcd-9de4-3b697e12752e/scratchpad/validation/hdfs-bgl/hdfs_sessions.jsonl --seeds 0..4 --train-seconds 120 --train-chunk-lines 24 --end-marker --out-dir /private/tmp/claude-501/-Users-abhinavpandey-Documents-GitHub-AOMB/93ef5deb-e083-4dcd-9de4-3b697e12752e/scratchpad/validation/hdfs-bgl/hdfs
```

- Wall: ingest 8s; eval 1657s after lock acquired (HDFS total incl. lock wait was ~4.1 h).
- sessions sha256 rebuilt = cached = `f445a5355ba1dcae1e12448d65a5685cabf63f5a9705193330ba8a8ef122f18b` (**match**)
- n_train 5000 (all normal), n_eval 10000, n_incident 271
- steps per seed [449, 453, 453, 456, 461]; max_seq_len 960; training sessions cropped: [0, 0, 0, 0, 0] (none)
- best baseline: **rarity_or_short**

| Method | kind | AUROC | PR-AUC |
|---|---|---|---|
| length | baseline | 0.569 | 0.123 |
| error_lines | baseline | 0.500 | 0.027 |
| duration_z | baseline | 0.500 | 0.027 |
| rule | baseline | 0.500 | 0.027 |
| novelty | baseline | 0.791 | 0.580 |
| heuristic | baseline | 0.791 | 0.580 |
| value_novelty | baseline | 0.766 | 0.534 |
| value_rarity | baseline | 0.822 | 0.605 |
| value_pair | baseline | 0.791 | 0.580 |
| sequence_novelty | baseline | 0.531 | 0.029 |
| too_short | baseline | 0.677 | 0.372 |
| rarity_or_short | baseline | 0.977 | 0.833 |
| bpb_mean | model | 0.800 ± 0.048 | 0.440 ± 0.039 |
| bpb_content | model | 0.974 ± 0.004 | 0.648 ± 0.023 |
| bpb_max_event | model | 0.954 ± 0.024 | 0.747 ± 0.060 |
| bpb_top10 | model | 0.966 ± 0.012 | 0.630 ± 0.025 |
| bits_max_field | model | 0.759 ± 0.047 | 0.673 ± 0.066 |
| bits_end | model | 0.737 ± 0.110 | 0.087 ± 0.049 |

## BGL

```
uv run python -m corpus.ingest.loghub_bgl --input ~/.cache/aomb-datasets/loghub/BGL --out /private/tmp/claude-501/-Users-abhinavpandey-Documents-GitHub-AOMB/93ef5deb-e083-4dcd-9de4-3b697e12752e/scratchpad/validation/hdfs-bgl/bgl_sessions.jsonl
uv run python -m eval.in_domain --sessions /private/tmp/claude-501/-Users-abhinavpandey-Documents-GitHub-AOMB/93ef5deb-e083-4dcd-9de4-3b697e12752e/scratchpad/validation/hdfs-bgl/bgl_sessions.jsonl --seeds 0..4 --train-seconds 120 --train-chunk-lines 24 --end-marker --out-dir /private/tmp/claude-501/-Users-abhinavpandey-Documents-GitHub-AOMB/93ef5deb-e083-4dcd-9de4-3b697e12752e/scratchpad/validation/hdfs-bgl/bgl
```

- Wall: ingest 12s; eval 1812s after lock acquired (HDFS total incl. lock wait was ~4.1 h).
- sessions sha256 rebuilt = cached = `e70abf326fabbb929be8b8487f33477559e18948258ecdfbf9aebc0827e99f09` (**match**)
- n_train 5000 (all normal), n_eval 10000, n_incident 863
- steps per seed [532, 522, 525, 527, 527]; max_seq_len 2240; training sessions cropped: [0, 0, 0, 0, 0] (none)
- best baseline: **heuristic**

| Method | kind | AUROC | PR-AUC |
|---|---|---|---|
| length | baseline | 0.624 | 0.117 |
| error_lines | baseline | 0.925 | 0.386 |
| duration_z | baseline | 0.500 | 0.086 |
| rule | baseline | 0.925 | 0.386 |
| novelty | baseline | 0.733 | 0.155 |
| heuristic | baseline | 0.932 | 0.427 |
| value_novelty | baseline | 0.484 | 0.085 |
| value_rarity | baseline | 0.890 | 0.326 |
| value_pair | baseline | 0.484 | 0.085 |
| sequence_novelty | baseline | 0.739 | 0.153 |
| too_short | baseline | 0.500 | 0.086 |
| rarity_or_short | baseline | 0.890 | 0.326 |
| bpb_mean | model | 0.790 ± 0.016 | 0.170 ± 0.010 |
| bpb_content | model | 0.798 ± 0.004 | 0.175 ± 0.004 |
| bpb_max_event | model | 0.777 ± 0.008 | 0.159 ± 0.004 |
| bpb_top10 | model | 0.829 ± 0.021 | 0.227 ± 0.027 |
| bits_max_field | model | 0.668 ± 0.047 | 0.119 ± 0.013 |
| bits_end | model | 0.739 ± 0.149 | 0.288 ± 0.219 |

## Leakage checks

- Label columns: HDFS labels come only from anomaly_label.csv and set the `label` field; BGL's first column (alert category) is split off and only sets `label`. Text never contains 'Anomaly'/'Normal'/'incident'/'label'/'fault' in any session. No BGL alert-category token appears in the text except `MMCS`, which is a component name (svc=MMCS..., 201 normal vs 3 incident sessions, so it is not a leak).
- Train/eval disjoint: 0 overlapping session ids, 0 duplicate eval ids, all train sessions normal (loader also enforces this).
- Temporal: BGL max train window index 5513 < min eval 5520; max train timestamp 1117990965 < min eval 1117990968. HDFS split is by block first appearance: latest train-block start 081109T204415 < earliest eval-block start 081109T204417 (0 eval blocks start earlier). Lines of 2075 eval blocks do fall inside the training time range, because HDFS blocks run concurrently and are long-lived. That is inherent to block sessions, and it is the stated design.
- Identifiers: HDFS block ids/IPs/job/task/dirs normalised; 1 session keeps `blk_1`/`blk_2` (a PacketResponder message pattern, not an id). BGL: 22 sessions keep raw midplane ids like `R73-M1`, a pattern the NODE regex misses; harmless for labels.
- What is kept is real telemetry, not leakage: BGL `level=` (FATAL/SEVERE…). BGL alerts are almost all FATAL (16131/17260 incident lines), but FATAL is also common in normal windows (20208 lines), so severity is a strong signal but not the label.

## Surprising / caveats

- HDFS: 4842/5000 training blocks exceed 24 lines, so the model is trained on 24-line chunks; 1304 eval sessions (97 incidents) exceed 24 lines and may be scored with context windows reset at max_seq_len (eval scoring has no crop check; it windows).
- HDFS: the rule-style baselines (error_lines/duration_z/rule) are exactly 0.500 (no ERROR levels or durations in HDFS). The best hand check, rarity_or_short (0.977 / PR 0.833), edges out bpb_top10 on AUROC and clearly beats it on PR-AUC. bpb_content (0.974) is the best model AUROC; bpb_max_event has the best model PR-AUC (0.747).
- BGL: bpb_top10 (0.829) is well below heuristic (0.932) and error_lines (0.925); every model view also loses on PR-AUC.
- bits_end has a large seed sd (0.11–0.15 AUROC; 0.22 PR on BGL): unstable.
- An exploratory leak2.py ran from 14:2x to 14:29. Training started at 18:04, so it did not compete with training.
