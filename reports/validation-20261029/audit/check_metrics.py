import sys, random, math
sys.path.insert(0, ".")
from eval.metrics import auroc, pr_auc
from eval.agent_commits import spearman
try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    roc_auc_score = None
rng = random.Random(0); worst_a = worst_p = 0
for t in range(300):
    n = rng.randint(5, 200); y = [int(rng.random() < rng.uniform(.05,.6)) for _ in range(n)]
    if sum(y) in (0, n): continue
    s = [round(rng.gauss(0,1), rng.choice([0,1,3])) for _ in range(n)]  # heavy ties
    if roc_auc_score:
        worst_a = max(worst_a, abs(auroc(y,s)-roc_auc_score(y,s)))
        worst_p = max(worst_p, abs(pr_auc(y,s)-average_precision_score(y,s)))
print("sklearn available:", bool(roc_auc_score), "max |auroc diff|", worst_a, "max |pr_auc diff| vs sklearn AP", worst_p)
print("all tied scores auroc:", auroc([0,1,0,1],[1,1,1,1]), "pr_auc:", pr_auc([0,1,0,1],[1,1,1,1]))
print("perfect inverted:", auroc([0,0,1,1],[3,4,1,2]))
print("one class:", auroc([1,1],[0.1,0.2]))
print("spearman with ties x=[1,1,1,2], y=[1,2,3,4]:", spearman([1,1,1,2],[1,2,3,4]), "(scipy-style avg-rank would be 0.775)")
# ranking_metrics NaN dropping
from eval.in_domain import ranking_metrics
print("ranking_metrics drops NaN rows:", ranking_metrics([1,0,0,1],[float('nan'),0.1,0.2,0.9]))
