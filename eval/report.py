"""Emit JSON + markdown reports for public accuracy eval (no invented claim language)."""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def git_sha(repo_root: str | Path | None = None, path: str = "HEAD") -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", path],
            cwd=str(repo_root) if repo_root else None,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def file_sha(repo_root: str | Path, relpath: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "hash-object", relpath],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def build_report(
    *,
    metrics: dict[str, Any],
    corpus: dict[str, Any],
    seed: int,
    score_method: str,
    hardware: dict[str, Any] | None = None,
    train_meta: dict[str, Any] | None = None,
    repo_root: str | Path | None = None,
    claim_status: str = "not_published",
    notes: str = "",
) -> dict[str, Any]:
    root = Path(repo_root) if repo_root else Path.cwd()
    report = {
        "protocol": "docs/public-accuracy-eval.md",
        "claim_status": claim_status,
        "disclaimer": (
            "Scaffolding / measurement report only. "
            "Do not cite as a public accuracy claim until the protocol checklist passes. "
            "Do not cite CRISP val_bpb=0.458756 or synthetic 0.3682 as public accuracy."
        ),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": seed,
        "score_method": score_method,
        "git": {
            "head": git_sha(root),
            "train_py": file_sha(root, "train.py"),
            "prepare_py": file_sha(root, "prepare.py"),
        },
        "corpus": corpus,
        "hardware": hardware or {},
        "train": train_meta or {},
        "metrics": metrics,
        "notes": notes,
    }
    return report


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return f"{x:.6f}"
    return str(x)


def render_markdown(report: dict[str, Any]) -> str:
    m = report.get("metrics") or {}
    corpus = report.get("corpus") or {}
    rb = m.get("random_baseline") or {}
    lines = [
        "# Public accuracy eval report",
        "",
        f"**Claim status:** `{report.get('claim_status', 'not_published')}`",
        "",
        report.get("disclaimer", ""),
        "",
        "## Identity",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Generated (UTC) | {report.get('generated_at')} |",
        f"| Seed | {report.get('seed')} |",
        f"| Score method | {report.get('score_method')} |",
        f"| Git HEAD | `{((report.get('git') or {}).get('head'))}` |",
        f"| train.py SHA | `{((report.get('git') or {}).get('train_py'))}` |",
        f"| prepare.py SHA | `{((report.get('git') or {}).get('prepare_py'))}` |",
        f"| Corpus capture_id | {corpus.get('capture_id')} |",
        f"| Corpus content SHA-256 | `{corpus.get('content_sha256')}` |",
        f"| Sessions (scorable) | {m.get('n')} "
        f"(pos={m.get('n_positive')}, neg={m.get('n_negative')}) |",
        "",
        "## Metrics (this seed)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| AUROC | {_fmt(m.get('auroc'))} |",
        f"| PR-AUC | {_fmt(m.get('pr_auc'))} |",
    ]
    for k, v in sorted((m.get("precision_at_k") or {}).items(), key=lambda t: int(t[0])):
        lines.append(f"| precision@{k} | {_fmt(v)} |")
    sm = m.get("score_mean_by_label") or {}
    lines += [
        f"| mean score (label 0) | {_fmt(sm.get('0'))} |",
        f"| mean score (label 1) | {_fmt(sm.get('1'))} |",
        "",
        "## Random ranking baseline",
        "",
        f"Draws: {rb.get('n_draws')} (seed={rb.get('seed')})",
        "",
        f"| Metric | mean | std |",
        f"|--------|------|-----|",
    ]
    for name in ("auroc", "pr_auc"):
        block = rb.get(name) or {}
        lines.append(
            f"| {name} | {_fmt(block.get('mean'))} | {_fmt(block.get('std'))} |"
        )
    for k, block in sorted(
        (rb.get("precision_at_k") or {}).items(), key=lambda t: int(t[0])
    ):
        lines.append(
            f"| precision@{k} | {_fmt(block.get('mean'))} | {_fmt(block.get('std'))} |"
        )
    notes = report.get("notes") or ""
    if notes:
        lines += ["", "## Notes", "", notes]
    lines += [
        "",
        "## Checklist reminder",
        "",
        "See `docs/public-accuracy-eval.md`. Multi-seed mean±std required before any "
        "public claim language. This file alone is not a published claim.",
        "",
    ]
    return "\n".join(lines)


def _repo_relpath(path: str | Path, repo_root: str | Path | None = None) -> str:
    """Prefer repo-relative paths in committed reports (no machine-absolute paths)."""
    p = Path(path)
    root = Path(repo_root) if repo_root else Path.cwd()
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except ValueError:
        # Outside repo — keep as-is but strip common absolute prefixes for hygiene
        s = str(p)
        for prefix in ("/workspace/", str(Path.home()) + "/"):
            if s.startswith(prefix):
                return s[len(prefix) :]
        return s


def write_report(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Hygiene: never commit machine-absolute capture_dir
    corpus = report.get("corpus")
    if isinstance(corpus, dict) and "capture_dir" in corpus:
        corpus = dict(corpus)
        corpus["capture_dir"] = _repo_relpath(corpus["capture_dir"])
        report = dict(report)
        report["corpus"] = corpus
    json_path = out / "report.json"
    md_path = out / "report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def aggregate_seed_reports(
    paths: list[Path],
    *,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate per-seed report.json files → mean±std (no invented extras)."""
    from eval.metrics import mean_std

    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    if not reports:
        raise ValueError("no seed reports to aggregate")

    root = Path(repo_root) if repo_root else Path.cwd()
    aurocs = [r["metrics"]["auroc"] for r in reports]
    pras = [r["metrics"]["pr_auc"] for r in reports]
    # union of k keys
    k_keys: set[str] = set()
    for r in reports:
        k_keys.update((r["metrics"].get("precision_at_k") or {}).keys())
    prec_agg = {
        k: mean_std([r["metrics"]["precision_at_k"][k] for r in reports if k in r["metrics"].get("precision_at_k", {})])
        for k in sorted(k_keys, key=int)
    }
    return {
        "protocol": "docs/public-accuracy-eval.md",
        "claim_status": "not_published",
        "disclaimer": (
            "Multi-seed aggregate only. Public claim language forbidden until "
            "docs/public-accuracy-eval.md checklist passes."
        ),
        "n_seeds": len(reports),
        "seeds": [r.get("seed") for r in reports],
        "score_method": reports[0].get("score_method"),
        "git_heads": [r.get("git", {}).get("head") for r in reports],
        "metrics_mean_std": {
            "auroc": mean_std(aurocs),
            "pr_auc": mean_std(pras),
            "precision_at_k": prec_agg,
        },
        "per_seed": [
            {
                "seed": r.get("seed"),
                "auroc": r["metrics"]["auroc"],
                "pr_auc": r["metrics"]["pr_auc"],
                "precision_at_k": r["metrics"].get("precision_at_k"),
                "path": _repo_relpath(paths[i], root),
            }
            for i, r in enumerate(reports)
        ],
    }


def render_aggregate_markdown(agg: dict[str, Any]) -> str:
    ms = agg.get("metrics_mean_std") or {}
    lines = [
        "# Public accuracy eval — multi-seed aggregate",
        "",
        f"**Claim status:** `{agg.get('claim_status')}`",
        "",
        agg.get("disclaimer", ""),
        "",
        f"Seeds ({agg.get('n_seeds')}): {agg.get('seeds')}",
        f"Score method: {agg.get('score_method')}",
        "",
        "## mean ± std",
        "",
        "| Metric | mean | std | n |",
        "|--------|------|-----|---|",
    ]
    for name in ("auroc", "pr_auc"):
        b = ms.get(name) or {}
        lines.append(
            f"| {name} | {_fmt(b.get('mean'))} | {_fmt(b.get('std'))} | {b.get('n')} |"
        )
    for k, b in (ms.get("precision_at_k") or {}).items():
        lines.append(
            f"| precision@{k} | {_fmt(b.get('mean'))} | {_fmt(b.get('std'))} | {b.get('n')} |"
        )
    lines += ["", "## Per-seed", ""]
    for row in agg.get("per_seed") or []:
        lines.append(
            f"- seed={row.get('seed')}: AUROC={_fmt(row.get('auroc'))} "
            f"PR-AUC={_fmt(row.get('pr_auc'))}"
        )
    lines.append("")
    return "\n".join(lines)
