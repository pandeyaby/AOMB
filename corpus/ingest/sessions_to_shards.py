"""
Write prepared session JSONL files as prepare.py training shards.

Train shards come from the ``split == "train"`` sessions of ``--train``; the pinned
val shard (``shard_06542``) comes from every session in ``--val``. Only normal
sessions are accepted, so an agent optimising val_bpb on these shards never sees
an incident. Afterwards run ``prepare.py --num-shards N`` to fit the tokenizer.

Refuses to write into a data dir that already has shards: move the current
cache aside first (``scripts/agent_hdfs_experiment.sh`` does this reversibly).

Usage:
    uv run python -m corpus.ingest.sessions_to_shards \\
        --train ~/.cache/aomb-datasets/loghub/hdfs_sessions.jsonl \\
        --val ~/.cache/aomb-datasets/loghub/hdfs_sessions.val.jsonl \\
        --data-dir ~/.cache/autoresearch/data --num-train-shards 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from corpus.ingest.build_shards import VAL_SHARD, chunked, write_parquet_shard


def read_texts(path: Path, split: str | None) -> list[str]:
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if split is not None and r.get("split") != split:
            continue
        if r.get("label") != "normal":
            raise ValueError(f"{path}: session {r.get('session_id')} is not normal")
        if r["text"]:
            texts.append(r["text"])
    return texts


def write(train_path: Path, val_path: Path, data_dir: Path, num_train_shards: int) -> dict:
    if data_dir.exists() and any(data_dir.glob("shard_*.parquet")):
        raise SystemExit(f"Refusing: {data_dir} already has shards. Move them aside first.")
    train = read_texts(train_path, "train")
    val = read_texts(val_path, None)
    per_shard = -(-len(train) // num_train_shards)
    written = [
        write_parquet_shard(i, docs, str(data_dir)) for i, docs in enumerate(chunked(train, per_shard))
    ]
    written.append(write_parquet_shard(VAL_SHARD, val, str(data_dir)))
    return {"train_docs": len(train), "val_docs": len(val), "shards": [Path(w).name for w in written]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--num-train-shards", type=int, default=8)
    a = p.parse_args()
    out = write(Path(a.train).expanduser(), Path(a.val).expanduser(), Path(a.data_dir).expanduser(), a.num_train_shards)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
