# In-domain lab eval — `pooled-20260926-valuedrift`

Train: 723 earlier normal sessions (no incidents). Eval: 726 later normals + 1443 incidents. Model: 120s × seeds [0, 1, 2, 3, 4]. Git `783b8b1`.

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
| bpb_mean (model) | 0.650 ± 0.055 | 0.777 ± 0.033 | 0.820 ± 0.164 | 0.602 ± 0.067 | 0.776 ± 0.073 | 0.568 ± 0.026 |
| bpb_content (model) | 0.564 ± 0.012 | 0.709 ± 0.008 | 0.800 ± 0.122 | 0.528 ± 0.014 | 0.517 ± 0.025 | 0.647 ± 0.022 |
| bpb_max_event (model) | 0.569 ± 0.007 | 0.737 ± 0.013 | 0.880 ± 0.084 | 0.516 ± 0.017 | 0.511 ± 0.025 | 0.681 ± 0.016 |
| bpb_top10 (model) | 0.558 ± 0.011 | 0.706 ± 0.007 | 0.840 ± 0.114 | 0.525 ± 0.011 | 0.516 ± 0.022 | 0.633 ± 0.021 |
