"""
Pick one scoring rule from the six development views (see
docs/real-data/scoring-rule-preregistration.md), or evaluate every candidate
on one finished run.

    uv run python -m eval.select_rule                       # select on the six views
    uv run python -m eval.select_rule --run <report dir>    # all candidates on one run
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from eval.metrics import auroc, mean_std, pr_auc

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports" / "public-accuracy"
DATA = Path(os.path.expanduser("~/.cache/aomb-datasets"))
CHECKOUT = "op=GET_/api/checkout"

SINGLE = ("bpb_content", "bpb_max_event", "bpb_top10", "bits_max_field", "bits_end")
COMBOS = {"combo_field_end": ("bits_max_field", "bits_end"), "combo_content_end": ("bpb_content", "bits_end")}
CANDIDATES = SINGLE + tuple(COMBOS)

# (name, report dir, how to choose the view)
VIEWS = [
    ("lab error/latency", "lab-pooled-20260918-eos-20261009", ("capture", "lab/published/pooled-20260918", None)),
    ("lab rule-proof (checkout)", "lab-pooled-20260925-ruleproof-eos-20261009", ("capture", "lab/published/pooled-20260925-ruleproof", CHECKOUT)),
    ("lab value drift (checkout)", "lab-pooled-20260926-valuedrift-eos-20261009", ("capture", "lab/published/pooled-20260926-valuedrift", CHECKOUT)),
    ("HDFS", "loghub-hdfs-eos-20261009", ("all", None, None)),
    ("RCAEval Online Boutique (faulted service)", "rcaeval-re3-online-boutique-eos-20261009", ("sessions", "rcaeval/sessions/re3_online-boutique.jsonl", None)),
    ("RCAEval Train Ticket (faulted service)", "rcaeval-re3-train-ticket-eos-20261009", ("sessions", "rcaeval/sessions/re3_train-ticket.jsonl", None)),
]


def percentile_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    for rank, i in enumerate(order):
        out[i] = rank / max(1, len(values) - 1)
    return out


def candidate_scores(seed_scores: dict[str, list[float]], idx: list[int]) -> dict[str, list[float]]:
    out = {m: [seed_scores[m][i] for i in idx] for m in SINGLE}
    for name, (a, b) in COMBOS.items():
        ra, rb = percentile_ranks(out[a]), percentile_ranks(out[b])
        out[name] = [x + y for x, y in zip(ra, rb)]
    return out


def view_indices(saved: dict, how: tuple) -> list[int]:
    kind, src, marker = how
    ids = saved["session_ids"]
    if kind == "all":
        return list(range(len(ids)))
    if kind == "sessions":
        from eval.in_domain import load_session_file

        _train, _ev, _groups, meta = load_session_file(DATA / src)
        return [i for i, sid in enumerate(ids) if sid in meta["subset_ids"]]
    from eval.in_domain import temporal_split
    from eval.lab_breakdown import session_capture_ids
    from eval.labels import filter_scorable, load_lab_sessions

    sessions, _ = load_lab_sessions(ROOT / src)
    _y, kept = filter_scorable(sessions)
    _train, eval_set = temporal_split(kept, session_capture_ids(ROOT / src))
    if [s.session_id for s in eval_set] != ids:
        raise RuntimeError(f"eval set of {src} differs from the saved run")
    return [i for i, s in enumerate(eval_set) if marker is None or marker in s.text]


def evaluate(report_dir: Path, how: tuple) -> dict[str, dict]:
    saved = json.loads((report_dir / "sessions.json").read_text(encoding="utf-8"))
    idx = view_indices(saved, how)
    y = [saved["labels"][i] for i in idx]
    per: dict[str, dict[str, list[float]]] = {c: {"auroc": [], "pr_auc": []} for c in CANDIDATES}
    for seed_scores in saved["model_scores_by_seed"].values():
        for c, sc in candidate_scores(seed_scores, idx).items():
            per[c]["auroc"].append(auroc(y, sc))
            per[c]["pr_auc"].append(pr_auc(y, sc))
    return {c: {k: mean_std(v) for k, v in d.items()} for c, d in per.items()} | {"_n": len(y), "_pos": sum(y)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", help="evaluate every candidate on one report dir (all eval sessions)")
    p.add_argument("--out", help="write the result JSON here")
    a = p.parse_args()

    if a.run:
        res = evaluate(Path(a.run), ("all", None, None))
        for c in CANDIDATES:
            print(f"{c:20} AUROC {res[c]['auroc']['mean']:.3f} ± {res[c]['auroc']['std']:.3f}   "
                  f"PR-AUC {res[c]['pr_auc']['mean']:.3f} ± {res[c]['pr_auc']['std']:.3f}")
        out = {"run": a.run, "candidates": {c: res[c] for c in CANDIDATES}, "n": res["_n"], "n_incident": res["_pos"]}
    else:
        table = {name: evaluate(R / d, how) for name, d, how in VIEWS}
        means = {c: sum(table[v][c]["auroc"]["mean"] for v in table) / len(table) for c in CANDIDATES}
        print(f"{'candidate':20} " + " ".join(f"{v[:14]:>14}" for v in table) + f" {'MEAN':>7}")
        for c in sorted(CANDIDATES, key=means.get, reverse=True):
            print(f"{c:20} " + " ".join(f"{table[v][c]['auroc']['mean']:>14.3f}" for v in table) + f" {means[c]:>7.3f}")
        winner = max(means, key=means.get)
        print(f"\nSelected rule: {winner} (six-view mean AUROC {means[winner]:.3f})")
        out = {"selected": winner, "six_view_mean_auroc": means,
               "views": {v: {c: table[v][c]["auroc"] for c in CANDIDATES} | {"n": table[v]["_n"], "n_incident": table[v]["_pos"]} for v in table}}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
