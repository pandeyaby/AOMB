"""
LogHub BGL → labelled AOMB sessions (fixed windows of consecutive log lines).

Source: LogHub BGL (Oliner & Stearley, DSN 2007; Zhu et al., ISSRE 2023), CC BY 4.0,
https://zenodo.org/records/8196385. 4.75M log lines from a Blue Gene/L
supercomputer at LLNL; each line is labelled by its first column ("-" = normal,
anything else = an alert category). 7.3% of lines are alerts. Download it
yourself; this repo never vendors it.

Protocol (fixed before any model was run on BGL):

- Session = ``window`` consecutive lines in file (chronological) order, default 20.
  A session is an incident if any of its lines is an alert.
- Split (temporal, as for HDFS): the first ``n_train`` all-normal windows train
  the model; ``n_eval`` windows are sampled (seeded) from the windows after the
  last training window, at the natural rate.

The label column is dropped and never enters the text. Identifiers are
normalised: node / location ids, hex addresses, IPs, and runs of 4+ digits.
The severity level (INFO / WARNING / ERROR / SEVERE / FAILURE / FATAL) is kept —
it is real telemetry, and a simple severity rule is a baseline the model must beat.

Usage:
    uv run python -m corpus.ingest.loghub_bgl --input ~/.cache/aomb-datasets/loghub/BGL \\
        --out ~/.cache/aomb-datasets/loghub/bgl_sessions.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

_NORMALISE = [
    (re.compile(r"\bR\d+-M\d+-[A-Z0-9]+(?:-[A-Z0-9]+)?(?::J\d+-U\d+)?\b"), "NODE"),
    (re.compile(r"0x[0-9a-fA-F]+"), "HEX"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "IP"),
    (re.compile(r"\b[0-9a-fA-F]{8,}\b"), "HEX"),
    (re.compile(r"\d{4,}"), "NUM"),
]


def normalise(msg: str) -> str:
    for rx, rep in _NORMALISE:
        msg = rx.sub(rep, msg)
    return msg


def render(line: str) -> tuple[bool, str] | None:
    """(is_alert, rendered line) or None for a malformed line."""
    parts = line.rstrip("\n").split(" ", 9)
    if len(parts) < 9:
        return None
    if len(parts) == 9:  # a line with no message text
        parts.append("")
    label, unix_ts, _date, _node, _dt, _node2, kind, component, level, msg = parts
    msg = "_".join(normalise(msg).split()) or "empty"
    return (
        label != "-",
        f"[ts={unix_ts}] [src=OTelLog] level={level} svc={kind}.{component} msg={msg}",
    )


def build(input_dir: Path, out: Path, window: int, n_train: int, n_eval: int, seed: int) -> dict:
    windows: list[tuple[bool, list[str]]] = []
    cur: list[str] = []
    alert = False
    log_sha = hashlib.sha256()
    with open(input_dir / "BGL.log", "rb") as f:
        for raw in f:
            log_sha.update(raw)
            r = render(raw.decode("utf-8", "replace"))
            if r is None:
                continue
            alert = alert or r[0]
            cur.append(r[1])
            if len(cur) == window:
                windows.append((alert, cur))
                cur, alert = [], False

    train_idx: list[int] = []
    for i, (is_alert, _lines) in enumerate(windows):
        if not is_alert:
            train_idx.append(i)
            if len(train_idx) == n_train:
                break
    later = list(range(train_idx[-1] + 1, len(windows)))
    eval_idx = random.Random(seed).sample(later, min(n_eval, len(later)))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for split, idxs in (("train", train_idx), ("eval", eval_idx)):
            for i in idxs:
                is_alert, lines = windows[i]
                f.write(
                    json.dumps(
                        {
                            "session_id": f"bgl:{i}",
                            "label": "incident" if is_alert else "normal",
                            "split": split,
                            "group": "bgl",
                            "text": "\n".join(lines),
                        }
                    )
                    + "\n"
                )
    meta = {
        "source": "LogHub BGL (CC BY 4.0) https://zenodo.org/records/8196385",
        "bgl_log_sha256": log_sha.hexdigest(),
        "sessions_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "window_lines": window,
        "n_windows": len(windows),
        "n_train_normal": len(train_idx),
        "n_eval": len(eval_idx),
        "n_eval_incident": sum(windows[i][0] for i in eval_idx),
        "seed": seed,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", required=True, help="unpacked BGL directory (contains BGL.log)")
    p.add_argument("--out", required=True)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-eval", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    meta = build(Path(a.input).expanduser(), Path(a.out).expanduser(), a.window, a.n_train, a.n_eval, a.seed)
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
