"""
Write AOMB parquet shards from a SourceAdapter.

Output matches prepare.py:
  ~/.cache/autoresearch/data/shard_NNNNN.parquet  (column: text)
  val shard index 6542 when --write-val-shard is set

Also writes provenance JSON under
  ~/.cache/autoresearch/corpus-v1/provenance/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Callable

import pyarrow as pa
import pyarrow.parquet as pq

from corpus.ingest.adapters.base import SourceAdapter, SourceBundle
from corpus.ingest.adapters.byo import ByoAdapter
from corpus.ingest.adapters.crisp_zenodo import CrispZenodoAdapter
from corpus.ingest.adapters.lab_capture import LabCaptureAdapter
from corpus.ingest.adapters.tale_of_errors import TaleOfErrorsAdapter
from corpus.ingest.otlp_to_sessions import bundle_to_sessions

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
PROV_DIR = os.path.join(CACHE_DIR, "corpus-v1", "provenance")
VAL_SHARD = 6542

ADAPTERS: dict[str, Callable[[], SourceAdapter]] = {
    "crisp_zenodo": CrispZenodoAdapter,
    "tale_of_errors": TaleOfErrorsAdapter,
    "lab_capture": LabCaptureAdapter,
    "byo": ByoAdapter,
}


def write_parquet_shard(shard_index: int, docs: list[str], data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    filename = f"shard_{shard_index:05d}.parquet"
    filepath = os.path.join(data_dir, filename)
    table = pa.table({"text": pa.array(docs, type=pa.string())})
    pq.write_table(table, filepath, compression="snappy")
    return filepath


def chunked(items: list[str], n: int) -> list[list[str]]:
    if n <= 0:
        return [items] if items else []
    return [items[i : i + n] for i in range(0, len(items), n)]


def build(
    adapter_name: str,
    input_path: str,
    *,
    num_train_shards: int = 8,
    docs_per_shard: int = 0,
    write_val_shard: bool = False,
    data_dir: str = DATA_DIR,
    max_spans: int = 0,
    max_logs: int = 0,
    include_meta: bool = True,
) -> dict:
    if adapter_name not in ADAPTERS:
        raise SystemExit(
            f"Unknown adapter {adapter_name!r}. Choose from: {sorted(ADAPTERS)}"
        )
    adapter = ADAPTERS[adapter_name]()
    sessions: list[str] = []
    window_counts: dict[str, int] = {}
    bundles: list[SourceBundle] = []

    # BYO adapter may yield ready session texts (parquet) via iter_sessions
    if hasattr(adapter, "iter_sessions"):
        seen_bundle_ids: set[int] = set()
        for text, window, bundle in adapter.iter_sessions(
            input_path,
            max_spans=max_spans,
            max_logs=max_logs,
            include_meta=include_meta,
        ):
            if id(bundle) not in seen_bundle_ids:
                bundles.append(bundle)
                seen_bundle_ids.add(id(bundle))
            sessions.append(text)
            window_counts[window.label] = window_counts.get(window.label, 0) + 1
    else:
        for bundle in adapter.load(
            input_path, max_spans=max_spans, max_logs=max_logs
        ):
            bundles.append(bundle)
            for text, window in bundle_to_sessions(
                bundle, include_meta=include_meta
            ):
                sessions.append(text)
                window_counts[window.label] = window_counts.get(window.label, 0) + 1

    if not sessions:
        raise SystemExit("No sessions produced — check input path / adapter.")

    # Split: last 10% (or at least 1) for val if requested
    val_docs: list[str] = []
    train_docs = sessions
    if write_val_shard:
        n_val = max(1, len(sessions) // 10)
        val_docs = sessions[-n_val:]
        train_docs = sessions[:-n_val] or sessions[:1]

    if docs_per_shard <= 0:
        docs_per_shard = max(1, (len(train_docs) + num_train_shards - 1) // num_train_shards)

    train_chunks = chunked(train_docs, docs_per_shard)[:num_train_shards]
    # Pad by cycling if fewer chunks than requested (small fixtures)
    while len(train_chunks) < num_train_shards and train_docs:
        train_chunks.append(train_docs[:docs_per_shard])

    shard_indices: list[int] = []
    written: list[str] = []
    for i, docs in enumerate(train_chunks):
        path = write_parquet_shard(i, docs, data_dir)
        shard_indices.append(i)
        written.append(path)
        print(f"  wrote {path} ({len(docs)} docs)")

    if write_val_shard and val_docs:
        path = write_parquet_shard(VAL_SHARD, val_docs, data_dir)
        shard_indices.append(VAL_SHARD)
        written.append(path)
        print(f"  wrote {path} ({len(val_docs)} docs) [val]")

    os.makedirs(PROV_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prov = {
        "corpus_version": "v1",
        "built_at": stamp,
        "adapter": adapter_name,
        "input_path": os.path.abspath(input_path),
        "session_count": len(sessions),
        "train_sessions": len(train_docs),
        "val_sessions": len(val_docs),
        "window_counts": window_counts,
        "shard_indices": shard_indices,
        "data_dir": os.path.abspath(data_dir),
        "sources": [
            {
                "source_id": b.source_id,
                "source_kind": b.source_kind,
                "license": b.license,
                "license_url": b.license_url,
                "citation": b.citation,
                "capture_id": b.capture_id,
                "span_count": len(b.spans),
                "log_count": len(b.logs),
                "windows": [
                    {
                        "label": w.label,
                        "start": w.start.isoformat() if w.start else None,
                        "end": w.end.isoformat() if w.end else None,
                        "fault": w.fault,
                        "notes": w.notes,
                    }
                    for w in b.windows
                ],
                "extra": b.extra_provenance,
            }
            for b in bundles
        ],
        "converter": "corpus/ingest/otlp_to_sessions.py",
    }
    prov_path = os.path.join(PROV_DIR, f"{adapter_name}_{stamp}.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    print(f"  provenance → {prov_path}")
    return prov


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build AOMB corpus shards from real OTLP")
    p.add_argument(
        "--adapter",
        required=True,
        choices=sorted(ADAPTERS),
        help="Source adapter",
    )
    p.add_argument("--input", required=True, help="Input directory for the adapter")
    p.add_argument("--num-train-shards", type=int, default=8)
    p.add_argument(
        "--docs-per-shard",
        type=int,
        default=0,
        help="Docs per train shard (0 = auto-split across num-train-shards)",
    )
    p.add_argument(
        "--write-val-shard",
        action="store_true",
        help=f"Also write pinned val shard_{VAL_SHARD:05d}.parquet",
    )
    p.add_argument("--data-dir", default=DATA_DIR)
    p.add_argument("--max-spans", type=int, default=0, help="Cap spans (0=all)")
    p.add_argument("--max-logs", type=int, default=0, help="Cap logs (0=all)")
    p.add_argument(
        "--no-meta",
        action="store_true",
        help="Omit # aomb_meta provenance lines from sessions",
    )
    args = p.parse_args(argv)

    print("=" * 60)
    print("AOMB corpus v1 — build shards (real telemetry)")
    print(f"  adapter: {args.adapter}")
    print(f"  input:   {args.input}")
    print(f"  output:  {args.data_dir}")
    print("=" * 60)
    build(
        args.adapter,
        args.input,
        num_train_shards=args.num_train_shards,
        docs_per_shard=args.docs_per_shard,
        write_val_shard=args.write_val_shard,
        data_dir=args.data_dir,
        max_spans=args.max_spans,
        max_logs=args.max_logs,
        include_meta=not args.no_meta,
    )
    print("\nNext: uv run python prepare.py --num-shards <N>")


if __name__ == "__main__":
    main()
