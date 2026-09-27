# In-domain lab eval — `pooled-20260926-valuedrift`

Train: 723 earlier normal sessions (no incidents). Eval: 726 later normals + 1443 incidents. Model: 120s × seeds [0, 1, 2, 3, 4]. Git `c2d1aa4`.

| Method | AUROC | PR-AUC | P@10 | `0926-currency_swap` | `60926-pricing_flip` | `0260926-region_new` |
|---|---|---|---|---|---|---|
| length (baseline) | 0.500 | 0.664 | 0.900 | 0.498 | 0.500 | 0.501 |
| error_lines (baseline) | 0.500 | 0.665 | 1.000 | 0.500 | 0.500 | 0.500 |
| duration_z (baseline) | 0.562 | 0.714 | 0.900 | 0.485 | 0.499 | 0.692 |
| rule (baseline) | 0.562 | 0.714 | 0.900 | 0.485 | 0.499 | 0.692 |
| novelty (baseline) | 0.501 | 0.666 | 1.000 | 0.501 | 0.501 | 0.501 |
| heuristic (baseline) | 0.564 | 0.717 | 0.900 | 0.487 | 0.501 | 0.694 |
| value_novelty (baseline) | 0.543 | 0.694 | 1.000 | 0.501 | 0.501 | 0.626 |
| value_rarity (baseline) | 0.586 | 0.740 | 1.000 | 0.498 | 0.621 | 0.640 |
| value_pair (baseline) | 0.584 | 0.722 | 1.000 | 0.626 | 0.501 | 0.626 |
| bpb_mean (model) | 0.655 ± 0.043 | 0.781 ± 0.031 | 0.800 ± 0.187 | 0.609 ± 0.066 | 0.790 ± 0.055 | 0.564 ± 0.015 |
| bpb_content (model) | 0.559 ± 0.009 | 0.703 ± 0.008 | 0.860 ± 0.055 | 0.529 ± 0.007 | 0.507 ± 0.010 | 0.641 ± 0.028 |
| bpb_max_event (model) | 0.563 ± 0.011 | 0.729 ± 0.013 | 0.860 ± 0.055 | 0.524 ± 0.012 | 0.497 ± 0.014 | 0.669 ± 0.027 |
| bpb_top10 (model) | 0.552 ± 0.007 | 0.700 ± 0.005 | 0.860 ± 0.089 | 0.524 ± 0.007 | 0.506 ± 0.008 | 0.627 ± 0.024 |
| bits_max_field (model) | 0.588 ± 0.023 | 0.764 ± 0.017 | 1.000 ± 0.000 | 0.539 ± 0.032 | 0.535 ± 0.020 | 0.685 ± 0.021 |
