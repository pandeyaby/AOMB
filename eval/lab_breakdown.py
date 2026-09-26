"""
Per-capture (per-fault) ranking breakdown for a pooled lab eval.

Pooled captures mix several fault runs. A pooled AUROC can hide a run where
the model fails, or be propped up by between-run differences (different day,
different load). This breakdown maps every session back to its source capture
window by its first event timestamp and reports AUROC *within* each capture:
that capture's incident sessions vs that same capture's normal sessions.

Usage:
    uv run python -m eval.lab_breakdown \\
        --capture lab/captures/pooled-20260918 \\
        --seeds-dir reports/public-accuracy/<run>/model-seeds
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from eval.labels import load_lab_sessions, load_provenance
from eval.metrics import auroc, mean_std

_TS = re.compile(r"\[ts=([0-9T:\-\.]+)Z\]")


def _parse_ts(s: str) -> datetime:
    s = s.rstrip("Z")
    fmt = "%Y-%m-%dT%H:%M:%S.%f" if "." in s else "%Y-%m-%dT%H:%M:%S"
    return datetime.strptime(s, fmt)


def session_capture_ids(capture_dir: str | Path) -> dict[str, str]:
    """session_id → source capture_id, via first event timestamp ∈ window."""
    sessions, _ = load_lab_sessions(capture_dir)
    prov = load_provenance(Path(capture_dir))
    default_cid = str(prov.get("capture_id") or Path(capture_dir).name)
    windows = [
        (
            w.get("capture_id") or default_cid,
            w["label"],
            _parse_ts(w["start"]),
            _parse_ts(w["end"]),
        )
        for w in prov.get("windows") or []
    ]
    out: dict[str, str] = {}
    for s in sessions:
        m = _TS.search(s.text)
        if not m:
            continue
        ts = _parse_ts(m.group(1))
        for cid, label, start, end in windows:
            # windows are second-resolution; include the whole end second
            if label == s.label and start <= ts.replace(microsecond=0) <= end:
                out[s.session_id] = cid
                break
    return out


def breakdown(capture_dir: str | Path, seed_reports: list[Path]) -> dict:
    cap_of = session_capture_ids(capture_dir)
    per_capture: dict[str, dict] = {}
    for rp in seed_reports:
        rows = json.loads(rp.read_text(encoding="utf-8"))["sessions"]
        by_cap: dict[str, tuple[list[int], list[float]]] = {}
        for r in rows:
            cid = cap_of.get(r["session_id"])
            if cid is None or r["binary"] is None or r["score"] is None:
                continue
            ys, ss = by_cap.setdefault(cid, ([], []))
            ys.append(int(r["binary"]))
            ss.append(float(r["score"]))
        for cid, (ys, ss) in by_cap.items():
            d = per_capture.setdefault(
                cid, {"n_normal": ys.count(0), "n_incident": ys.count(1), "auroc": []}
            )
            d["auroc"].append(auroc(ys, ss))
    for d in per_capture.values():
        d["auroc"] = mean_std(d["auroc"])
    return {
        "n_sessions_mapped": len(cap_of),
        "n_seeds": len(seed_reports),
        "per_capture": dict(sorted(per_capture.items())),
    }


def render_markdown(result: dict, capture_dir: str | Path) -> str:
    prov = load_provenance(Path(capture_dir))
    default_cid = str(prov.get("capture_id") or Path(capture_dir).name)
    faults = {
        w.get("capture_id") or default_cid: w.get("fault", "")
        for w in prov.get("windows") or []
        if w.get("fault")
    }
    lines = [
        "| Capture | Fault | Normal | Incident | Within-capture AUROC (mean ± std) |",
        "|---------|-------|--------|----------|-----------------------------------|",
    ]
    for cid, d in result["per_capture"].items():
        a = d["auroc"]
        lines.append(
            f"| `{cid}` | {faults.get(cid, '')} | {d['n_normal']} | {d['n_incident']} "
            f"| {a['mean']:.4f} ± {a['std']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--capture", required=True)
    p.add_argument("--seeds-dir", required=True, help="dir holding seed-*/report.json")
    p.add_argument("--out", default=None, help="write JSON + .md next to this path")
    args = p.parse_args(argv)

    reports = sorted(Path(args.seeds_dir).glob("seed-*/report.json"))
    if not reports:
        print(f"ERROR: no seed-*/report.json under {args.seeds_dir}", file=sys.stderr)
        return 2
    result = breakdown(args.capture, reports)
    md = render_markdown(result, args.capture)
    print(md)
    out = Path(args.out) if args.out else Path(args.seeds_dir) / "breakdown.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(f"Wrote {out} (+ .md) — mapped {result['n_sessions_mapped']} sessions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
