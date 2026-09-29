"""
Does the overnight agent's val_bpb progress buy better anomaly detection?

For the branch-start commit and every later commit that changed ``train.py``
(i.e. every change the agent kept because val_bpb improved), this:

1. runs that commit's own ``train.py`` unmodified, in a fresh process, on the
   current prepare.py cache (so the exact training loop the agent kept);
2. records the val_bpb it prints (measured on normal-only val shards);
3. scores every eval session with the trained model and reports AUROC / PR-AUC
   for each scoring variant in eval.in_domain.

The primary variant is ``bpb_content`` (masked session mean bits-per-byte),
fixed before the run because it is the same quantity val_bpb measures.
The summary reports the rank correlation between val_bpb and primary AUROC
across commits.

Usage (after an agent-loop run on branch <branch> starting at <base>):
    uv run python -m eval.agent_commits --base <sha> --branch <branch> \\
        --sessions ~/.cache/aomb-datasets/loghub/hdfs_sessions.jsonl \\
        --out-dir reports/public-accuracy/agent-loop-hdfs-<date>
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "bpb_content"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def commits_to_evaluate(base: str, branch: str) -> list[dict]:
    """Base commit, then every commit on base..branch that touched train.py (oldest first)."""
    rows = [{"sha": git("rev-parse", base).strip(), "subject": "branch start (before any agent change)"}]
    log = git("log", "--reverse", "--format=%H%x09%s", f"{base}..{branch}", "--", "train.py")
    for line in log.splitlines():
        sha, subject = line.split("\t", 1)
        rows.append({"sha": sha, "subject": subject})
    return rows


# ---------------------------------------------------------------- one commit (child process)


def run_one(sha: str, sessions: str, max_eval: int | None, out: Path) -> None:
    import torch

    from eval.in_domain import MODEL_METHODS, load_session_file, model_scores_for, ranking_metrics, token_surprise

    source = git("show", f"{sha}:train.py")
    module = types.ModuleType("_aomb_agent_commit")
    module.__file__ = str(ROOT / "train.py")
    sys.modules[module.__name__] = module  # lets @dataclass resolve its module
    t0 = time.time()
    exec(compile(source, f"train.py@{sha[:7]}", "exec"), module.__dict__)
    train_seconds = time.time() - t0

    import prepare

    ns = module.__dict__
    model, tokenizer = ns["model"], ns["tokenizer"]
    device = next(model.parameters()).device
    token_bytes = prepare.get_token_bytes(device=str(device))
    model.eval()

    _train, eval_set, _groups, meta = load_session_file(sessions)
    if max_eval:
        eval_set = eval_set[:max_eval]
    y = [int(s.binary) for s in eval_set]
    scores: dict[str, list[float]] = {m: [] for m in MODEL_METHODS}
    with torch.no_grad():
        for s in eval_set:
            toks = token_surprise(model, tokenizer, token_bytes, s.text, prepare.MAX_SEQ_LEN)
            for m, v in model_scores_for(toks, s.text).items():
                scores[m].append(v)
    result = {
        "sha": sha,
        "val_bpb": float(ns["val_bpb"]),
        "num_steps": int(ns.get("step", -1)),
        "train_and_val_seconds": round(train_seconds, 1),
        "sessions_sha256": meta["content_sha256"],
        "n_eval": len(y),
        "n_incident": sum(y),
        "metrics": {m: ranking_metrics(y, scores[m]) for m in MODEL_METHODS},
    }
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- driver


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r

    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return cov / den if den else float("nan")


def render(summary: dict) -> str:
    lines = [
        "# Agent loop vs detection",
        "",
        f"Primary score (fixed in advance): `{PRIMARY}`. Spearman ρ(val_bpb, primary AUROC) = "
        f"**{summary['spearman_valbpb_vs_auroc']:.3f}** over {len(summary['rows'])} runs "
        "(negative = lower val_bpb goes with higher AUROC).",
        "",
        "| # | Commit | val_bpb | AUROC (primary) | PR-AUC (primary) | AUROC (max event) | AUROC (per-field) | Change |",
        "|---|--------|---------|-----------------|------------------|-------------------|-------------------|--------|",
    ]
    for i, r in enumerate(summary["rows"]):
        m = r["metrics"]
        lines.append(
            f"| {i} | `{r['sha'][:7]}` | {r['val_bpb']:.4f} | {m[PRIMARY]['auroc']:.3f} | "
            f"{m[PRIMARY]['pr_auc']:.3f} | {m['bpb_max_event']['auroc']:.3f} | "
            f"{m['bits_max_field']['auroc']:.3f} | {r['subject'][:70]} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sessions", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--base", help="branch-start commit (before the agent's first change)")
    p.add_argument("--branch", help="branch the agent committed to")
    p.add_argument("--max-eval", type=int, default=None, help="score only the first N eval sessions")
    p.add_argument("--repeat-ends", type=int, default=1, help="extra runs of first and last commit (noise)")
    p.add_argument("--run-one", metavar="SHA", help=argparse.SUPPRESS)
    p.add_argument("--result", help=argparse.SUPPRESS)
    a = p.parse_args(argv)

    if a.run_one:
        run_one(a.run_one, a.sessions, a.max_eval, Path(a.result))
        return 0
    if not (a.base and a.branch):
        p.error("--base and --branch are required")

    out = Path(a.out_dir)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    commits = commits_to_evaluate(a.base, a.branch)
    plan = list(commits)
    if len(commits) > 1:
        plan += [commits[0]] * a.repeat_ends + [commits[-1]] * a.repeat_ends
    rows = []
    for n, c in enumerate(plan):
        path = out / "runs" / f"{n:03d}_{c['sha'][:7]}.json"
        if not path.exists():
            print(f"[{n + 1}/{len(plan)}] {c['sha'][:7]} {c['subject'][:60]}", flush=True)
            cmd = [sys.executable, "-m", "eval.agent_commits", "--sessions", a.sessions,
                   "--run-one", c["sha"], "--result", str(path)]
            if a.max_eval:
                cmd += ["--max-eval", str(a.max_eval)]
            subprocess.run(cmd, cwd=ROOT, check=True)
        r = json.loads(path.read_text(encoding="utf-8"))
        rows.append({**r, "subject": c["subject"]})

    xs = [r["val_bpb"] for r in rows]
    ys = [r["metrics"][PRIMARY]["auroc"] for r in rows]
    summary = {"primary": PRIMARY, "base": commits[0]["sha"], "branch": a.branch,
               "spearman_valbpb_vs_auroc": spearman(xs, ys), "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md = render(summary)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
