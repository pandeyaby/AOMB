"""
RCAEval (RE2/RE3) → labelled AOMB trace sessions, one JSONL file per system.

Source: RCAEval (Pham et al., WWW'25 companion; arXiv 2412.17015), MIT license,
https://huggingface.co/datasets/phamquiluan/RCAEval. Three microservice
benchmark systems (Online Boutique, Sock Shop, Train Ticket), each case = ~15
minutes of normal traffic, a fault injected into one service at ``inject_time``,
then ~15 minutes more. RE3's 90 cases are code-level faults designed by the
RCAEval authors — nobody on this project chose them, which is the point.
Download it yourself; this repo never vendors it.

One session per trace (spans sorted by start time), rendered in AOMB's span
format. Per case (all temporal): the earlier half of pre-fault traces are
training candidates, the later half are eval normals, post-fault traces are
eval incidents. Up to ``per_case`` of each are sampled (seeded).

Every session records ``touches_root_cause``: whether the trace includes the
faulted service. Traces that never touch it can't show the fault, so the eval
also reports that subset — chosen by service, not by label, for both classes.

Usage:
    uv run python -m corpus.ingest.rcaeval --input ~/.cache/aomb-datasets/rcaeval \\
        --suite re3 --out-dir ~/.cache/aomb-datasets/rcaeval/sessions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

SYSTEMS = {"ob": "online-boutique", "ss": "sock-shop", "tt": "train-ticket"}
_IDLIKE = [
    (re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"), "UUID"),
    (re.compile(r"\b[0-9a-fA-F]{16,}\b"), "HEX"),
    (re.compile(r"\d{4,}"), "NUM"),
]


def normalise(op: str) -> str:
    for rx, rep in _IDLIKE:
        op = rx.sub(rep, op)
    return "_".join(op.split()) or "unknown"


def render_trace(spans: list[dict]) -> str:
    lines = []
    for sp in sorted(spans, key=lambda r: (r["startTime"], r["spanID"])):
        status = sp.get("statusCode")
        status = "ok" if status in (None, "None", "", 0) else str(status)
        parent = sp.get("parentSpanID")
        parent = "n/a" if parent in (None, "None", "") else parent
        lines.append(
            f"[ts={sp['startTimeMillis']}] [src=OTel] trace_id={sp['traceID']} span_id={sp['spanID']} "
            f"parent={parent} op={normalise(str(sp['operationName']))} svc={sp['serviceName']} "
            f"duration_ms={int(sp['duration']) // 1000} status={status}"
        )
    return "\n".join(lines)


def case_sessions(case_dir: Path, root_cause: str, per_case: int, rng: random.Random) -> list[dict]:
    import pyarrow.parquet as pq

    inject_ms = int((case_dir / "inject_time.txt").read_text().strip()) * 1000
    cols = ["traceID", "spanID", "parentSpanID", "serviceName", "operationName",
            "startTimeMillis", "startTime", "duration", "statusCode"]
    spans = pq.read_table(case_dir / "traces.parquet", columns=cols).to_pylist()
    traces: dict[str, list[dict]] = defaultdict(list)
    for sp in spans:
        traces[sp["traceID"]].append(sp)
    start = {t: min(s["startTimeMillis"] for s in v) for t, v in traces.items()}
    pre = sorted((t for t in traces if start[t] < inject_ms), key=start.get)
    post = [t for t in traces if start[t] >= inject_ms]
    half = len(pre) // 2
    groups = {
        "train": pre[:half],
        "eval_normal": pre[half:],
        "eval_incident": post,
    }
    out = []
    for role, ids in groups.items():
        for t in rng.sample(ids, min(per_case, len(ids))):
            out.append(
                {
                    "session_id": f"{case_dir.name}:{t}",
                    "label": "incident" if role == "eval_incident" else "normal",
                    "split": "train" if role == "train" else "eval",
                    "group": case_dir.name,
                    "touches_root_cause": any(s["serviceName"] == root_cause for s in traces[t]),
                    "text": render_trace(traces[t]),
                }
            )
    return out


def build(input_dir: Path, suite: str, out_dir: Path, per_case: int, seed: int) -> dict:
    import pyarrow.parquet as pq

    meta_rows = pq.read_table(input_dir / "cases.parquet").to_pylist()
    root_of = {r["case"]: r["root_cause_service"] for r in meta_rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for code, name in SYSTEMS.items():
        cases = sorted(p for p in input_dir.glob(f"{suite}{code}_*") if (p / "traces.parquet").is_file())
        if not cases:
            continue
        rng = random.Random(seed)
        rows = []
        for c in cases:
            rows.extend(case_sessions(c, root_of[c.name], per_case, rng))
        path = out_dir / f"{suite}_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summary[name] = {
            "cases": len(cases),
            "train": sum(r["split"] == "train" for r in rows),
            "eval_normal": sum(r["split"] == "eval" and r["label"] == "normal" for r in rows),
            "eval_incident": sum(r["label"] == "incident" for r in rows),
            "sessions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    meta = {
        "source": "RCAEval (MIT) https://huggingface.co/datasets/phamquiluan/RCAEval",
        "suite": suite,
        "per_case": per_case,
        "seed": seed,
        "systems": summary,
    }
    (out_dir / f"{suite}_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", required=True, help="dir with cases.parquet and <case>/ folders")
    p.add_argument("--suite", default="re3")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--per-case", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    meta = build(Path(args.input).expanduser(), args.suite, Path(args.out_dir).expanduser(), args.per_case, args.seed)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
