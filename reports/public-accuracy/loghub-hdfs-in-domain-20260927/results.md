# In-domain lab eval — `hdfs_sessions`

Train: 5000 earlier normal sessions (no incidents). Eval: 9729 later normals + 271 incidents. Model: 120s × seeds [0, 1, 2, 3, 4]. Git `cadddd3`.

| Method | AUROC | PR-AUC | P@10 | `hdfs_v1` |
|---|---|---|---|---|
| length (baseline) | 0.569 | 0.123 | 0.800 | 0.569 |
| error_lines (baseline) | 0.500 | 0.027 | 0.000 | 0.500 |
| duration_z (baseline) | 0.500 | 0.027 | 0.000 | 0.500 |
| rule (baseline) | 0.500 | 0.027 | 0.000 | 0.500 |
| novelty (baseline) | 0.791 | 0.580 | 1.000 | 0.791 |
| heuristic (baseline) | 0.791 | 0.580 | 1.000 | 0.791 |
| value_novelty (baseline) | 0.766 | 0.534 | 1.000 | 0.766 |
| value_rarity (baseline) | 0.822 | 0.605 | 0.900 | 0.822 |
| value_pair (baseline) | 0.791 | 0.580 | 1.000 | 0.791 |
| sequence_novelty (baseline) | 0.531 | 0.029 | 0.000 | 0.531 |
| bpb_mean (model) | 0.758 ± 0.027 | 0.426 ± 0.027 | 1.000 ± 0.000 | 0.758 ± 0.027 |
| bpb_content (model) | 0.878 ± 0.022 | 0.631 ± 0.014 | 1.000 ± 0.000 | 0.878 ± 0.022 |
| bpb_max_event (model) | 0.855 ± 0.003 | 0.740 ± 0.018 | 1.000 ± 0.000 | 0.855 ± 0.003 |
| bpb_top10 (model) | 0.851 ± 0.010 | 0.624 ± 0.008 | 1.000 ± 0.000 | 0.851 ± 0.010 |
| bits_max_field (model) | 0.802 ± 0.032 | 0.714 ± 0.017 | 1.000 ± 0.000 | 0.802 ± 0.032 |
