# In-domain lab eval — `hdfs_sessions`

Train: 5000 earlier normal sessions (no incidents). Eval: 9729 later normals + 271 incidents. Model: 120s × seeds [0, 1, 2, 3, 4]. Git `e18d888`.

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
| too_short (baseline) | 0.677 | 0.372 | 1.000 | 0.677 |
| rarity_or_short (baseline) | 0.977 | 0.833 | 1.000 | 0.977 |
| bpb_mean (model) | 0.775 ± 0.040 | 0.421 ± 0.034 | 1.000 ± 0.000 | 0.775 ± 0.040 |
| bpb_content (model) | 0.974 ± 0.006 | 0.639 ± 0.054 | 1.000 ± 0.000 | 0.974 ± 0.006 |
| bpb_max_event (model) | 0.963 ± 0.015 | 0.785 ± 0.040 | 1.000 ± 0.000 | 0.963 ± 0.015 |
| bpb_top10 (model) | 0.964 ± 0.020 | 0.621 ± 0.063 | 1.000 ± 0.000 | 0.964 ± 0.020 |
| bits_max_field (model) | 0.778 ± 0.027 | 0.661 ± 0.069 | 1.000 ± 0.000 | 0.778 ± 0.027 |
| bits_end (model) | 0.756 ± 0.063 | 0.158 ± 0.101 | 0.220 ± 0.259 | 0.756 ± 0.063 |
