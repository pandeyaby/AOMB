"""
LogHub HDFS_v1 → labelled AOMB sessions (one session per HDFS block).

Source: LogHub HDFS_v1 (Xu et al., SOSP 2009; Zhu et al., ISSRE 2023), CC BY 4.0,
https://zenodo.org/records/8196385. 11.2M lines of real Hadoop logs from a
private cloud running benchmark workloads; 575,061 blocks labelled Normal /
Anomaly (2.93% anomalous). Download it yourself; this repo never vendors it.

Split (temporal, DeepLog-style): the first ``n_train`` *normal* blocks in
first-appearance order train the model; ``n_eval`` blocks are sampled uniformly
(seeded) from all blocks that first appear after the last training block, at
the natural anomaly rate. ``anomaly_label.csv`` is already in first-appearance
order (checked against HDFS.log).

Each log line is rendered in AOMB's session format::

    [ts=081109T203518] [src=OTelLog] level=INFO svc=dfs.DataNode$DataXceiver msg=Receiving_block_blk_src:_IP_dest:_IP

Identifiers carry no health signal and several are clock-like, so they are
normalised before anything sees the text: block ids, IPs/ports, job and task ids
(which embed a start timestamp), job output directories (workload names that
change over time), part numbers and data sub-directories. The
thread id is dropped. Message words and sizes are kept.

Usage:
    uv run python -m corpus.ingest.loghub_hdfs \\
        --input ~/.cache/aomb-datasets/loghub/HDFS_v1 \\
        --out ~/.cache/aomb-datasets/loghub/hdfs_sessions.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

_BLK = re.compile(r"blk_-?\d+")
_NORMALISE = [
    # job output dirs name the workload (rand, randtxt4, sortrand2...), which changes
    # over time: an identifier, not a health signal
    (re.compile(r"/user/root/[^/\s]+/"), "/user/root/JOB/"),
    (re.compile(r"_?task_\d+_\d+_[mr]_\d+_\d+"), "_task"),
    (re.compile(r"job_\d+_\d+"), "job"),
    (_BLK, "blk"),
    (re.compile(r"/?\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?"), "IP"),
    (re.compile(r"part-\d+"), "part"),
    (re.compile(r"subdir\d+"), "subdir"),
]
_LINE = re.compile(r"^(\d{6}) (\d{6}) \d+ (\w+) ([^:]+): (.*)$")


def normalise(msg: str) -> str:
    for rx, rep in _NORMALISE:
        msg = rx.sub(rep, msg)
    return msg


def render(line: str) -> str | None:
    m = _LINE.match(line.rstrip("\n"))
    if not m:
        return None
    date, time, level, component, msg = m.groups()
    msg = "_".join(normalise(msg).split())
    return f"[ts={date}T{time}] [src=OTelLog] level={level} svc={component} msg={msg}"


def select_blocks(labels: list[tuple[str, str]], n_train: int, n_eval: int, seed: int):
    """(train block ids, eval block ids) — see module docstring."""
    train, last = [], -1
    for i, (blk, lab) in enumerate(labels):
        if lab == "Normal":
            train.append(blk)
            last = i
            if len(train) == n_train:
                break
    later = [blk for blk, _ in labels[last + 1 :]]
    rng = random.Random(seed)
    return train, rng.sample(later, min(n_eval, len(later)))


def select_val(labels: list[tuple[str, str]], exclude: set[str], n_val: int, seed: int) -> list[str]:
    """
    ``n_val`` normal blocks for a val_bpb shard, disjoint from train and eval.

    Drawn after the eval sample (separate RNG) so adding a val set never
    changes the published eval set.
    """
    if n_val <= 0:
        return []
    pool = [b for b, lab in labels if lab == "Normal" and b not in exclude]
    return random.Random(seed + 1).sample(pool, min(n_val, len(pool)))


def build(
    input_dir: Path, out: Path, n_train: int, n_eval: int, seed: int, n_val: int = 0
) -> dict:
    with open(input_dir / "preprocessed" / "anomaly_label.csv", newline="") as f:
        labels = [(r["BlockId"], r["Label"]) for r in csv.DictReader(f)]
    label_of = dict(labels)
    train, evals = select_blocks(labels, n_train, n_eval, seed)
    role = {b: "train" for b in train} | {b: "eval" for b in evals}
    # val candidates must come after the training window, like eval
    later = {b for b, _ in labels[labels.index((train[-1], "Normal")) + 1 :]} if train else set()
    val = select_val([(b, l) for b, l in labels if b in later], set(role), n_val, seed)
    role |= {b: "val" for b in val}

    lines: dict[str, list[str]] = {b: [] for b in role}
    log_sha = hashlib.sha256()
    with open(input_dir / "HDFS.log", "rb") as f:
        for raw in f:
            log_sha.update(raw)
            text = raw.decode("utf-8", "replace")
            blks = {b for b in _BLK.findall(text) if b in role}
            if not blks:
                continue
            rendered = render(text)
            if rendered is None:
                continue
            for b in blks:
                lines[b].append(rendered)

    out.parent.mkdir(parents=True, exist_ok=True)
    if val:
        val_out = out.with_suffix(".val.jsonl")
        with val_out.open("w", encoding="utf-8") as f:
            for b in val:
                f.write(json.dumps({"session_id": b, "label": "normal", "split": "val",
                                    "group": "hdfs_v1", "text": "\n".join(lines[b])}) + "\n")
    with out.open("w", encoding="utf-8") as f:
        for b in train + evals:
            f.write(
                json.dumps(
                    {
                        "session_id": b,
                        "label": "normal" if label_of[b] == "Normal" else "incident",
                        "split": role[b],
                        "group": "hdfs_v1",
                        "text": "\n".join(lines[b]),
                    }
                )
                + "\n"
            )
    n_eval_anom = sum(label_of[b] == "Anomaly" for b in evals)
    meta = {
        "source": "LogHub HDFS_v1 (CC BY 4.0) https://zenodo.org/records/8196385",
        "hdfs_log_sha256": log_sha.hexdigest(),
        "sessions_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "n_train_normal": len(train),
        "n_eval": len(evals),
        "n_eval_anomaly": n_eval_anom,
        "seed": seed,
        "split": "first n_train normal blocks (first-appearance order); eval sampled from later blocks",
        "n_val_normal": len(val),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", required=True, help="unpacked HDFS_v1 directory")
    p.add_argument("--out", required=True, help="output sessions .jsonl")
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--n-eval", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n-val",
        type=int,
        default=0,
        help="Also write <out>.val.jsonl: normal blocks disjoint from train and eval (for a val_bpb shard)",
    )
    args = p.parse_args()
    meta = build(
        Path(args.input).expanduser(), Path(args.out).expanduser(), args.n_train, args.n_eval, args.seed, args.n_val
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
