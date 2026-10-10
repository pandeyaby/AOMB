# In-domain lab eval — `bgl_sessions`

Train: 5000 earlier normal sessions (no incidents). Eval: 9137 later normals + 863 incidents. Model: 120s × seeds [0, 1, 2, 3, 4]. Git `0fc33eb`.

| Method | AUROC | PR-AUC | P@10 | `bgl` |
|---|---|---|---|---|
| length (baseline) | 0.624 | 0.117 | 0.000 | 0.624 |
| error_lines (baseline) | 0.925 | 0.386 | 0.500 | 0.925 |
| duration_z (baseline) | 0.500 | 0.086 | 0.100 | 0.500 |
| rule (baseline) | 0.925 | 0.386 | 0.500 | 0.925 |
| novelty (baseline) | 0.733 | 0.155 | 0.000 | 0.733 |
| heuristic (baseline) | 0.932 | 0.427 | 0.100 | 0.932 |
| value_novelty (baseline) | 0.484 | 0.085 | 0.000 | 0.484 |
| value_rarity (baseline) | 0.890 | 0.326 | 0.000 | 0.890 |
| value_pair (baseline) | 0.484 | 0.085 | 0.000 | 0.484 |
| sequence_novelty (baseline) | 0.739 | 0.153 | 0.100 | 0.739 |
| too_short (baseline) | 0.500 | 0.086 | 0.100 | 0.500 |
| rarity_or_short (baseline) | 0.890 | 0.326 | 0.000 | 0.890 |
| bpb_mean (model) | 0.793 ± 0.014 | 0.172 ± 0.010 | 0.000 ± 0.000 | 0.793 ± 0.014 |
| bpb_content (model) | 0.800 ± 0.003 | 0.177 ± 0.003 | 0.000 ± 0.000 | 0.800 ± 0.003 |
| bpb_max_event (model) | 0.779 ± 0.009 | 0.160 ± 0.005 | 0.040 ± 0.055 | 0.779 ± 0.009 |
| bpb_top10 (model) | 0.814 ± 0.019 | 0.203 ± 0.025 | 0.000 ± 0.000 | 0.814 ± 0.019 |
| bits_max_field (model) | 0.689 ± 0.048 | 0.126 ± 0.015 | 0.480 ± 0.045 | 0.689 ± 0.048 |
| bits_end (model) | 0.732 ± 0.119 | 0.160 ± 0.052 | 0.000 ± 0.000 | 0.732 ± 0.119 |
