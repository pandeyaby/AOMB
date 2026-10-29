# Validator report: agent-kept train.py changes vs base on labelled HDFS

## Blinding disclosure
Project auto-memory loaded into my context at session start already contained published numbers for this experiment (real HDFS bpb_top10 AUROC before/after, plus val_bpb direction). I did not open README, docs/, reports/, or git log, and I did not compare against those numbers. My measurements below come only from my own runs.

## Setup
- Worktree at $OUT/wt, temp branch validation/agent-run-tmp from fdd9b57. Applied all_kept_changes.patch (git apply --check clean) and committed as ecf014b. The diff touches only train.py: focal/anomaly-weighted loss replaced by plain CE, TOTAL_BATCH_SIZE 2^15->2^14, MATRIX_LR 0.02->0.06, DEVICE_BATCH_SIZE 16->8. Saved to applied_diff.txt.
- prepare.py, eval/in_domain.py and eval/agent_commits.py are identical between fdd9b57 and main b5538b7.
- Command, run through run_locked.sh inside swap_and_eval.sh (cache swap with a trap restore):
    cd /Users/abhinavpandey/Documents/GitHub/AOMB
    caffeinate -i uv run python -m eval.agent_commits --base fdd9b57 --branch validation/agent-run-tmp \
      --sessions ~/.cache/aomb-datasets/loghub/hdfs_sessions.jsonl --out-dir $OUT/eval \
      --end-marker --primary bpb_top10 --repeat-ends 2
  This gave 3 independent trainings of each commit. Total time was 3151 s. Raw output is in eval/runs/*.json, eval/summary.json and eval_run.log.

## Per run (10000 eval blocks, 271 anomalies; sessions sha256 f445a535... in every run). Cells are AUROC/PR-AUC.
| run | commit | steps | val_bpb | top10 | content | max_event | mean | max_field | end |
|---|---|---|---|---|---|---|---|---|---|
| 0 | base fdd9b57 | 722 | 0.4485 | 0.860/0.342 | 0.857/0.482 | 0.942/0.856 | 0.722/0.355 | 0.806/0.710 | 0.773/0.213 |
| 2 | base | 737 | 0.4568 | 0.856/0.415 | 0.850/0.501 | 0.941/0.874 | 0.717/0.370 | 0.788/0.710 | 0.703/0.225 |
| 3 | base | 734 | 0.4248 | 0.851/0.404 | 0.847/0.465 | 0.940/0.829 | 0.705/0.370 | 0.809/0.719 | 0.614/0.122 |
| 1 | patched ecf014b | 1425 | 0.5634 | 0.974/0.645 | 0.973/0.647 | 0.953/0.829 | 0.793/0.436 | 0.786/0.633 | 0.651/0.454 |
| 4 | patched | 1398 | 0.6551 | 0.925/0.605 | 0.918/0.587 | 0.891/0.566 | 0.752/0.427 | 0.795/0.698 | 0.726/0.051 |
| 5 | patched | 1417 | 0.5785 | 0.966/0.606 | 0.966/0.613 | 0.909/0.618 | 0.807/0.459 | 0.829/0.697 | 0.793/0.095 |

## Aggregate (AUROC, mean +/- sd over 3 runs)
| score | base | patched | delta |
|---|---|---|---|
| bpb_top10 (primary) | 0.856 +/- 0.005 (PR 0.387) | 0.955 +/- 0.026 (PR 0.618) | +0.100 |
| bpb_content | 0.852 +/- 0.005 | 0.953 +/- 0.030 | +0.101 |
| bpb_mean | 0.715 +/- 0.009 | 0.784 +/- 0.029 | +0.069 |
| bpb_max_event | 0.941 +/- 0.001 (PR 0.853) | 0.918 +/- 0.032 (PR 0.671) | -0.023 |
| bits_max_field | 0.801 +/- 0.011 | 0.804 +/- 0.023 | +0.003 |
| bits_end | 0.697 +/- 0.080 | 0.724 +/- 0.071 | +0.027 (within noise) |
| val_bpb (lower is better) | 0.443 | 0.599 | worse by 0.156 |

On the primary score the ranges do not overlap: the worst patched run (0.925) beats the best base run (0.860). Patched runs are about 5x noisier. Spearman between val_bpb and primary AUROC over the 6 runs is +0.714, so the run with lower val_bpb did not have higher AUROC.

## Fairness
- Objective data. The objective (eval/objective.py, set up by scripts/agent_inject_experiment.sh) scores 300 normal blocks from hdfs_sessions.val.jsonl. It fits its injection pool from split=="train" normals of hdfs_sessions.jsonl.
  - The val file is 1000 normals, all with split=val.
  - Its session IDs overlap 0 eval blocks and 0 train blocks, and its texts overlap 0 eval texts.
  - The agent's objective never scores the labelled eval blocks.
- Training shards. I checked the training shards in the HDFS cache.
  - The 8 train shards hold 5000 docs, all from the train split.
  - The val shard holds 1000 docs, all from the val file.
  - No shard contains any eval-block text, whether normal or incident.
- Identical conditions. Base and patched used the same sessions file (same sha256 in every run), the same scoring code (main checkout), the same shards, and the same 300 s training budget (training_seconds 300.0 to 300.4 in every run).
- Caveat 1, in-process train.py. The agent's train.py runs in the same process as the objective. It could in principle read env AOMB_OBJECTIVE_TRAIN, which points to the full sessions file including labelled eval rows. The kept patch does nothing like that: it only changes the loss and hyperparameters.
- Caveat 2, agent file access. I did not verify whether the claude --print agent could read files on disk.

## Cache
- cache_before.txt and cache_after.txt hold ls -la plus shasum of every file in ~/.cache/autoresearch/{data,tokenizer}.
- All file checksums and all data/tokenizer entries are identical.
- The only difference is the '..' line, which is the mtime of the parent ~/.cache/autoresearch: Oct 11 13:37 before, Oct 28 17:27 after. The script created and removed the hold dir there.
- The HDFS source cache is also unchanged (hdfs_src_before.txt and hdfs_src_after.txt are identical, and the in-place copy matched the source).

## Surprises
- Step count. Patched runs reach about 2x the optimizer steps (1398 to 1425 vs 722 to 737) because of the halved batch.
- val_bpb vs detection. val_bpb gets clearly worse while the primary AUROC improves.
- Not every score improves. max-event AUROC and PR-AUC fall, and per-field is flat.
- Patched noise. Patched run 4 is much weaker than runs 1 and 5.

## Cleanup
Worktree removed and branch validation/agent-run-tmp deleted. git worktree list shows only /Users/abhinavpandey/Documents/GitHub/AOMB b5538b7 [main].
