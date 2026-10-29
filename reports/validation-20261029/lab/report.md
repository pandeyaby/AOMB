# Lab in-domain validation (blinded re-measurement)

Repo /Users/abhinavpandey/Documents/GitHub/AOMB at b5538b7 (git_head recorded in every results.json). Host: Apple Silicon, MPS. All commands were run from the repo dir and wrapped in run_locked.sh, one at a time via driver.sh in this folder.

## Blinding disclosure
Before I started, the harness auto-injected the user's memory file (MEMORY.md) into my context. It contains published lab numbers: for example, the 2026-09-26 in-domain error/latency model 0.740 vs rule 0.776, rule-proof checkout model 0.944 / novelty 0.875 / rule+novelty 0.890, value-drift checkout per-fault figures, and 'model at chance 0.48'. I did not open any README, docs, reports or git log. My numbers come only from the runs below, and no analysis choices were changed to match anything. The lab capture hashes were checked with tests/test_published_captures.py, which passed (3 tests, 6 subtests). That test compares the hashes to report files, but it prints no numbers. I also recomputed the hashes independently.

## Commands
```
OUT=<this folder>; cd /Users/abhinavpandey/Documents/GitHub/AOMB
run_locked.sh uv run python -m eval.in_domain --capture lab/published/<pool> --seeds 0..4 --train-seconds 120 --end-marker --out-dir $OUT/<pool>
run_locked.sh uv run python -m eval.in_domain --capture lab/published/<pool> --out-dir $OUT/<pool> --subset-marker "op=GET_/api/checkout"
```
The checkout subset was run for all three pools. The subset mode reports AUROC only, not PR-AUC.

## pooled-20260918

- content_sha256 `effb9a3a70075d7e0681c3de6d4a9e236de8677b14dba3acd851a945b18e53f5` (independent recompute matches)
- n_train 726 normals; n_eval 2174 (730 normal + 1444 incident; prevalence 0.66)
- steps per seed 0..4: [459, 463, 477, 481, 480]; n_train_cropped [0, 0, 0, 0, 0]; device mps, max_seq_len 768, vocab 512
- wall time: full run 778s (after lock acquired), subset 1s; rc 0/0
- NaN model scores: 0
- Best baseline (whole, by AUROC): **rule**; checkout subset (n=545, 360 incidents): **rule**

| Method | kind | AUROC | PR-AUC | P@10 | checkout AUROC |
|---|---|---|---|---|---|
| length | baseline | 0.538 ± 0.000 | 0.682 ± 0.000 | 0.000 ± 0.000 | 0.605 ± 0.000 |
| error_lines | baseline | 0.621 ± 0.000 | 0.746 ± 0.000 | 1.000 ± 0.000 | 0.682 ± 0.000 |
| duration_z | baseline | 0.715 ± 0.000 | 0.869 ± 0.000 | 1.000 ± 0.000 | 0.793 ± 0.000 |
| rule | baseline | 0.776 ± 0.000 | 0.896 ± 0.000 | 1.000 ± 0.000 | 0.857 ± 0.000 |
| novelty | baseline | 0.561 ± 0.000 | 0.705 ± 0.000 | 1.000 ± 0.000 | 0.682 ± 0.000 |
| heuristic | baseline | 0.776 ± 0.000 | 0.896 ± 0.000 | 1.000 ± 0.000 | 0.857 ± 0.000 |
| value_novelty | baseline | 0.623 ± 0.000 | 0.747 ± 0.000 | 1.000 ± 0.000 | 0.682 ± 0.000 |
| value_rarity | baseline | 0.614 ± 0.000 | 0.766 ± 0.000 | 1.000 ± 0.000 | 0.682 ± 0.000 |
| value_pair | baseline | 0.623 ± 0.000 | 0.747 ± 0.000 | 1.000 ± 0.000 | 0.682 ± 0.000 |
| sequence_novelty | baseline | 0.575 ± 0.000 | 0.714 ± 0.000 | 1.000 ± 0.000 | 0.738 ± 0.000 |
| too_short | baseline | 0.500 ± 0.000 | 0.664 ± 0.000 | 1.000 ± 0.000 | 0.500 ± 0.000 |
| rarity_or_short | baseline | 0.614 ± 0.000 | 0.766 ± 0.000 | 1.000 ± 0.000 | 0.682 ± 0.000 |
| bpb_mean | model | 0.689 ± 0.016 | 0.835 ± 0.015 | 1.000 ± 0.000 | 0.784 ± 0.019 |
| bpb_content | model | 0.735 ± 0.009 | 0.871 ± 0.006 | 1.000 ± 0.000 | 0.799 ± 0.017 |
| bpb_max_event | model | 0.739 ± 0.012 | 0.875 ± 0.007 | 1.000 ± 0.000 | 0.806 ± 0.011 |
| bpb_top10 | model | 0.734 ± 0.009 | 0.871 ± 0.006 | 1.000 ± 0.000 | 0.798 ± 0.017 |
| bits_max_field | model | 0.780 ± 0.008 | 0.902 ± 0.003 | 1.000 ± 0.000 | 0.853 ± 0.002 |
| bits_end | model | 0.609 ± 0.037 | 0.789 ± 0.034 | 1.000 ± 0.000 | 0.717 ± 0.041 |

## pooled-20260925-ruleproof

- content_sha256 `21970ed8827cb55e0b55308087fde0a95af9f06a6dd2c86c0840102c2a750188` (independent recompute matches)
- n_train 324 normals; n_eval 972 (328 normal + 644 incident; prevalence 0.66)
- steps per seed 0..4: [480, 480, 478, 478, 477]; n_train_cropped [0, 0, 0, 0, 0]; device mps, max_seq_len 512, vocab 512
- wall time: full run 685s (after lock acquired), subset 1s; rc 0/0
- NaN model scores: 0
- Best baseline (whole, by AUROC): **heuristic**; checkout subset (n=244, 160 incidents): **heuristic**

| Method | kind | AUROC | PR-AUC | P@10 | checkout AUROC |
|---|---|---|---|---|---|
| length | baseline | 0.541 ± 0.000 | 0.732 ± 0.000 | 1.000 ± 0.000 | 0.750 ± 0.000 |
| error_lines | baseline | 0.500 ± 0.000 | 0.663 ± 0.000 | 1.000 ± 0.000 | 0.500 ± 0.000 |
| duration_z | baseline | 0.608 ± 0.000 | 0.790 ± 0.000 | 1.000 ± 0.000 | 0.601 ± 0.000 |
| rule | baseline | 0.608 ± 0.000 | 0.790 ± 0.000 | 1.000 ± 0.000 | 0.601 ± 0.000 |
| novelty | baseline | 0.627 ± 0.000 | 0.748 ± 0.000 | 1.000 ± 0.000 | 0.875 ± 0.000 |
| heuristic | baseline | 0.680 ± 0.000 | 0.841 ± 0.000 | 1.000 ± 0.000 | 0.890 ± 0.000 |
| value_novelty | baseline | 0.565 ± 0.000 | 0.707 ± 0.000 | 1.000 ± 0.000 | 0.750 ± 0.000 |
| value_rarity | baseline | 0.580 ± 0.000 | 0.731 ± 0.000 | 1.000 ± 0.000 | 0.750 ± 0.000 |
| value_pair | baseline | 0.565 ± 0.000 | 0.707 ± 0.000 | 1.000 ± 0.000 | 0.750 ± 0.000 |
| sequence_novelty | baseline | 0.627 ± 0.000 | 0.748 ± 0.000 | 1.000 ± 0.000 | 0.875 ± 0.000 |
| too_short | baseline | 0.500 ± 0.000 | 0.663 ± 0.000 | 1.000 ± 0.000 | 0.500 ± 0.000 |
| rarity_or_short | baseline | 0.580 ± 0.000 | 0.731 ± 0.000 | 1.000 ± 0.000 | 0.750 ± 0.000 |
| bpb_mean | model | 0.650 ± 0.031 | 0.806 ± 0.018 | 1.000 ± 0.000 | 0.819 ± 0.017 |
| bpb_content | model | 0.701 ± 0.013 | 0.833 ± 0.011 | 1.000 ± 0.000 | 0.940 ± 0.027 |
| bpb_max_event | model | 0.696 ± 0.022 | 0.841 ± 0.013 | 1.000 ± 0.000 | 0.969 ± 0.025 |
| bpb_top10 | model | 0.695 ± 0.012 | 0.825 ± 0.009 | 1.000 ± 0.000 | 0.930 ± 0.028 |
| bits_max_field | model | 0.711 ± 0.005 | 0.856 ± 0.004 | 1.000 ± 0.000 | 0.962 ± 0.011 |
| bits_end | model | 0.644 ± 0.038 | 0.771 ± 0.054 | 0.920 ± 0.110 | 0.883 ± 0.065 |

## pooled-20260926-valuedrift

- content_sha256 `088c818399acc4d6b35cdc223501f52e040fc9657740329295900ff953b92dab` (independent recompute matches)
- n_train 723 normals; n_eval 2169 (726 normal + 1443 incident; prevalence 0.67)
- steps per seed 0..4: [478, 477, 466, 445, 466]; n_train_cropped [0, 0, 0, 0, 0]; device mps, max_seq_len 576, vocab 512
- wall time: full run 765s (after lock acquired), subset 0s; rc 0/0
- NaN model scores: 0
- Best baseline (whole, by AUROC): **value_rarity**; checkout subset (n=543, 360 incidents): **value_pair**

| Method | kind | AUROC | PR-AUC | P@10 | checkout AUROC |
|---|---|---|---|---|---|
| length | baseline | 0.500 ± 0.000 | 0.664 ± 0.000 | 0.900 ± 0.000 | 0.500 ± 0.000 |
| error_lines | baseline | 0.500 ± 0.000 | 0.665 ± 0.000 | 1.000 ± 0.000 | 0.500 ± 0.000 |
| duration_z | baseline | 0.562 ± 0.000 | 0.714 ± 0.000 | 0.900 ± 0.000 | 0.565 ± 0.000 |
| rule | baseline | 0.562 ± 0.000 | 0.714 ± 0.000 | 0.900 ± 0.000 | 0.565 ± 0.000 |
| novelty | baseline | 0.501 ± 0.000 | 0.666 ± 0.000 | 1.000 ± 0.000 | 0.500 ± 0.000 |
| heuristic | baseline | 0.564 ± 0.000 | 0.717 ± 0.000 | 0.900 ± 0.000 | 0.565 ± 0.000 |
| value_novelty | baseline | 0.543 ± 0.000 | 0.694 ± 0.000 | 1.000 ± 0.000 | 0.667 ± 0.000 |
| value_rarity | baseline | 0.586 ± 0.000 | 0.740 ± 0.000 | 1.000 ± 0.000 | 0.792 ± 0.000 |
| value_pair | baseline | 0.584 ± 0.000 | 0.722 ± 0.000 | 1.000 ± 0.000 | 0.833 ± 0.000 |
| sequence_novelty | baseline | 0.501 ± 0.000 | 0.666 ± 0.000 | 1.000 ± 0.000 | 0.500 ± 0.000 |
| too_short | baseline | 0.500 ± 0.000 | 0.665 ± 0.000 | 1.000 ± 0.000 | 0.500 ± 0.000 |
| rarity_or_short | baseline | 0.586 ± 0.000 | 0.740 ± 0.000 | 1.000 ± 0.000 | 0.792 ± 0.000 |
| bpb_mean | model | 0.667 ± 0.027 | 0.789 ± 0.017 | 0.800 ± 0.122 | 0.733 ± 0.036 |
| bpb_content | model | 0.569 ± 0.011 | 0.712 ± 0.010 | 0.780 ± 0.084 | 0.723 ± 0.022 |
| bpb_max_event | model | 0.568 ± 0.003 | 0.739 ± 0.008 | 0.900 ± 0.122 | 0.699 ± 0.016 |
| bpb_top10 | model | 0.564 ± 0.008 | 0.708 ± 0.007 | 0.800 ± 0.100 | 0.722 ± 0.021 |
| bits_max_field | model | 0.589 ± 0.013 | 0.767 ± 0.010 | 1.000 ± 0.000 | 0.838 ± 0.024 |
| bits_end | model | 0.501 ± 0.007 | 0.664 ± 0.006 | 0.760 ± 0.089 | 0.520 ± 0.016 |

## Per-fault checkout AUROC (from subset per_capture_auroc)
- ruleproof, bpb_top10: db_failover 0.976, retry_storm 0.855, silent_fallback 1.000, skip_cache 0.865. Heuristic: 0.662 / 1.000 / 1.000 / 1.000. bpb_max_event: 1.000 / 0.912 / 1.000 / 0.947.
- valuedrift, bpb_top10: currency_swap 0.670, pricing_flip 0.544, region_new 0.944. value_pair: 1.000 / 0.500 / 1.000. value_rarity: 0.493 / 0.886 / 1.000. bits_max_field: 0.854 / 0.645 / 0.999.

## Anomalies / observations
- No sessions were cropped, there were no NaNs, every rc was 0, and the step counts were consistent (445-481 per 120 s). The 20260918 full run waited about 3 h for the lock (14:26 -> 17:27) and then ran 778 s, so no compute was contended.
- Incidents make up about 66% of each eval set, so the PR-AUC floor is about 0.66 and P@10 = 1.000 is near-trivial for most methods.
- Many baselines sit exactly at 0.500 (error_lines on ruleproof and valuedrift; too_short everywhere; length, novelty and sequence_novelty on valuedrift). These are constant-score ties, as expected for faults those checks cannot see.
- Several per-fault AUROCs are perfect: 1.000 for silent_fallback (most model scores and several baselines), region_new (value checks), and kill_redis in 20260918. In 20260918's first capture, 'length' scores 0.000 per-capture on checkout, a perfect inversion. On ruleproof skip_cache, length is also 0.000 and rule 0.217. These look like small, uniform per-capture groups, not bugs.
- valuedrift: bpb_mean, which includes IDs and timestamps, is the best model score on the whole set (0.667 vs bpb_top10 0.564). Its strength comes mostly from pricing_flip (0.919 on checkout). That may be a confound from unmasked ID or timestamp content and is worth checking.
- 20260918: the best model score is bits_max_field (0.780), not bpb_top10 (0.734). The rule and heuristic baselines are identical, because novelty adds nothing there.
- Model variance across seeds is small (sd at most 0.04).
