"""Reproducer: code running inside train.py (exec'd in-process by eval/objective.py:80)
can rebind eval.in_domain.* before inject_scores imports them (objective.py:51), and can
read AOMB_OBJECTIVE_VAL to know which texts are the clean ones."""
import sys, json, os
sys.path.insert(0, os.getcwd())
val = "/Users/abhinavpandey/.cache/aomb-datasets/loghub/hdfs_sessions.val.jsonl"
clean = [json.loads(l)["text"] for l in open(val)][:50]
ref = [json.loads(l)["text"] for l in open("/Users/abhinavpandey/.cache/aomb-datasets/loghub/hdfs_sessions.jsonl") if '"split": "train"' in l][:500]
# ---- what a hostile train.py could do (no model at all) ----
import eval.in_domain as d
known = {t + "\n" + d.END_MARKER for t in clean}        # read from the env-var path
d.token_surprise = lambda *a, **k: []
d.model_scores_for = lambda toks, text: {"bpb_top10": 0.0 if text in known else 1.0}
# ---- objective's own scorer ----
from eval.objective import inject_scores
print(inject_scores(None, None, None, 0, clean, ref))
