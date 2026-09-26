#!/usr/bin/env python3
"""
Pool several lab captures into one capture dir for labelled evaluation.

Concatenates each capture's merged traces.jsonl / logs.jsonl and merges the
provenance windows, tagging every window with its source capture_id so
eval.lab_breakdown / eval.in_domain can split and report per capture.

Usage:
    python3 lab/scripts/pool_captures.py lab/captures/pooled-<id> \\
        lab/captures/<capture-a> lab/captures/<capture-b> ...
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def pool(out: Path, captures: list[Path], pool_id: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    windows, pooled_from = [], []
    for name in ("traces.jsonl", "logs.jsonl"):
        with (out / name).open("w", encoding="utf-8") as dst:
            for cap in captures:
                src = cap / name
                if src.is_file():
                    text = src.read_text(encoding="utf-8")
                    dst.write(text if text.endswith("\n") or not text else text + "\n")
    for cap in captures:
        prov = json.loads((cap / "provenance.json").read_text(encoding="utf-8"))
        cid = prov.get("capture_id") or cap.name
        faults = sorted({w.get("fault", "") for w in prov.get("windows", []) if w.get("fault")})
        pooled_from.append({"capture_id": cid, "fault": ",".join(faults), "path": str(cap)})
        for w in prov.get("windows", []):
            windows.append({**w, "notes": f"{w.get('notes', '')} [from {cid}]", "capture_id": cid})
    provenance = {
        "corpus_version": "v1",
        "source_id": "lab-aomb-stack-pooled",
        "source_kind": "lab_capture",
        "license": "Apache-2.0",
        "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "citation": "AOMB lab stack pooled captures for labelled ranking eval",
        "capture_id": pool_id,
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "capture_tool": "lab/scripts/pool_captures.py",
        "signals": ["traces", "logs"],
        "windows": windows,
        "pooled_from": pooled_from,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("out")
    p.add_argument("captures", nargs="+")
    args = p.parse_args()
    out = Path(args.out)
    prov = pool(out, [Path(c) for c in args.captures], out.name)
    print(f"Pooled {len(prov['pooled_from'])} captures, {len(prov['windows'])} windows → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
