"""
Streaming capped extractor for Uber Tale of Errors ``.tar.zst`` archives.

Zenodo ships split ``trace1_*`` / ``trace2_*`` pieces that reassemble to
``trace{N}-sanitized.tar.zst``. Full ``zstd -d`` materializes **300–500 GB**
per archive — out of scope on a ~315 Gi free Mac.

This module:
  1. Accepts a directory of sorted ``trace1_*`` (or ``trace2_*``) pieces,
     optionally concatenating them to a named ``.tar.zst`` without writing
     the decompressed tree; OR a path to an already-concatenated ``.tar.zst``.
  2. Stream-decompresses with ``zstandard`` and iterates tar members.
  3. Writes only enough Jaeger JSON under ``out/traces/`` to hit
     ``--max-spans`` / ``--max-files`` / ``--max-bytes``, then **stops**.
  4. Produces a local tree the ``tale_of_errors`` adapter / ``build_shards``
     can load (same discovery as CRISP: ``**/traces/**/*.json`` / ``**/*.json``).

Loud refusals: ``--auroc``, full decompress without caps, inventing metrics.
Does **not** invent ``val_bpb``. ``prepare.py`` is sacred — not invoked here.

Usage::

  uv run python -m corpus.ingest.tale_stream_extract --help
  uv run python -m corpus.ingest.tale_stream_extract \\
    --input ~/.cache/autoresearch/corpus-v1/tale_of_errors \\
    --out /tmp/aomb-tale-capped --prefix trace1_ \\
    --max-spans 50000 --max-files 200
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Optional

# Span counting is inlined (avoid importing corpus.ingest.jaeger → circular
# adapters import). Layout matches CrispZenodoAdapter discovery.

SKIP_JSON_NAMES = frozenset(
    {
        "package.json",
        "package-lock.json",
        "composer.json",
        "tsconfig.json",
        ".eslintrc.json",
        "provenance.json",
        "extract_provenance.json",
    }
)

DECOMPRESSED_DISK_NOTE = "300–500 GB per archive after full zstd decompress"
DISK_CAP_NOTE = (
    "Maintainer Mac ~315 Gi free → full decompress OUT OF SCOPE. "
    "Use --max-spans / --max-files / --max-bytes and stop early."
)

_REFUSED_METRIC_FLAGS = frozenset(
    {
        "--auroc",
        "--lab-auroc",
        "--accuracy",
        "--ranking",
        "--publish",
        "--claim",
        "--val-bpb",
        "--invent-val-bpb",
        "--claim-val-bpb",
    }
)


class ConcatReader(io.RawIOBase):
    """Read-only sequential view over sorted split pieces (no on-disk cat)."""

    def __init__(self, paths: list[str]) -> None:
        super().__init__()
        if not paths:
            raise ValueError("ConcatReader requires at least one path")
        self._paths = list(paths)
        self._idx = 0
        self._fh: Optional[BinaryIO] = open(self._paths[0], "rb")
        self._closed = False

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        if self._closed or self._fh is None:
            return b""
        if size == 0:
            return b""
        chunks: list[bytes] = []
        remaining: Optional[int] = size if size is not None and size >= 0 else None
        while True:
            assert self._fh is not None
            data = self._fh.read() if remaining is None else self._fh.read(remaining)
            if data:
                chunks.append(data)
                if remaining is not None:
                    remaining -= len(data)
                    if remaining <= 0:
                        break
            if not data or remaining is None:
                self._fh.close()
                self._fh = None
                self._idx += 1
                if self._idx >= len(self._paths):
                    break
                self._fh = open(self._paths[self._idx], "rb")
                if remaining is not None and remaining <= 0:
                    break
                if remaining is None:
                    continue
                if not data:
                    continue
            elif remaining is not None and remaining <= 0:
                break
        return b"".join(chunks)

    def readinto(self, b: bytearray | memoryview) -> int:  # type: ignore[override]
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def close(self) -> None:
        if not self._closed:
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            self._closed = True
        super().close()


def discover_trace_pieces(input_dir: str, prefix: str = "trace1_") -> list[str]:
    """Return sorted absolute paths matching ``prefix*`` under ``input_dir``."""
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    pieces = sorted(
        p for p in root.iterdir() if p.is_file() and p.name.startswith(prefix)
    )
    return [str(p.resolve()) for p in pieces]


def resolve_archive_source(
    input_path: str,
    *,
    prefix: str = "trace1_",
    concat_out: Optional[str] = None,
) -> tuple[str, Optional[ConcatReader], list[str]]:
    """
    Resolve ``input_path`` to a streamable ``.tar.zst`` source.

    Returns ``(label, concat_reader_or_none, piece_paths)``.
    If ``concat_reader_or_none`` is set, caller must stream from it (and close).
    If None, ``label`` is a filesystem path to an existing ``.tar.zst``.
    """
    path = os.path.abspath(input_path)
    if os.path.isfile(path):
        lower = path.lower()
        if not (lower.endswith(".tar.zst") or lower.endswith(".zst")):
            raise ValueError(
                f"Expected a .tar.zst archive or a directory of {prefix}* pieces, "
                f"got file: {path}"
            )
        return path, None, [path]

    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Input not found: {path}. Pass a directory of {prefix}* pieces "
            "or a path to an already-concatenated .tar.zst."
        )

    pieces = discover_trace_pieces(path, prefix=prefix)
    if not pieces:
        raise FileNotFoundError(
            f"No '{prefix}*' pieces under {path}. "
            "Download selectively via: "
            "uv run python -m corpus.ingest.fetch_tale_of_errors --download <KEY> "
            "or pass --input pointing at a .tar.zst."
        )

    if concat_out:
        dest = os.path.abspath(concat_out)
        parent = os.path.dirname(dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        print(
            f"Concatenating {len(pieces)} pieces → {dest} "
            "(compressed archive only; not full decompress)",
            file=sys.stderr,
        )
        with open(dest, "wb") as out_f:
            for piece in pieces:
                with open(piece, "rb") as in_f:
                    while True:
                        chunk = in_f.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        out_f.write(chunk)
        return dest, None, pieces

    reader = ConcatReader(pieces)
    return f"<concat:{prefix}*{len(pieces)}>", reader, pieces


def _looks_like_jaeger(doc: object) -> bool:
    if isinstance(doc, dict):
        if "data" in doc and isinstance(doc["data"], list):
            return True
        if "spans" in doc and (
            "traceID" in doc or "traceId" in doc or "processes" in doc
        ):
            return True
    if isinstance(doc, list) and doc:
        return _looks_like_jaeger(doc[0])
    return False


def _iter_traces(doc: object):
    if doc is None:
        return
    if isinstance(doc, list):
        for item in doc:
            yield from _iter_traces(item)
        return
    if not isinstance(doc, dict):
        return
    if "data" in doc and isinstance(doc["data"], list):
        for item in doc["data"]:
            if isinstance(item, dict) and ("spans" in item or "traceID" in item or "traceId" in item):
                yield item
        return
    if "spans" in doc or "traceID" in doc or "traceId" in doc:
        yield doc


def _count_spans(doc: object) -> int:
    if not _looks_like_jaeger(doc):
        return 0
    total = 0
    for trace in _iter_traces(doc):
        spans = trace.get("spans") or []
        if isinstance(spans, list):
            total += len(spans)
    return total


def _safe_member_relpath(member_name: str) -> Optional[str]:
    """Sanitize tar member path; return None to skip."""
    name = member_name.replace("\\", "/").lstrip("/")
    if not name or name.endswith("/"):
        return None
    parts = [p for p in name.split("/") if p and p != "."]
    if not parts or any(p == ".." for p in parts):
        return None
    base = parts[-1]
    if not base.endswith(".json"):
        return None
    if base in SKIP_JSON_NAMES:
        return None
    return "/".join(parts)


def _output_relpath(member_rel: str) -> str:
    """
    Place JSON under ``traces/`` so CRISP/Tale preferred globs hit.

    If the member already lives under a ``traces/`` or ``trace/`` segment,
    preserve from that segment; else ``traces/<flattened>``.
    """
    parts = member_rel.split("/")
    for i, part in enumerate(parts):
        if part in {"traces", "trace"}:
            return "/".join(parts[i:])
    return f"traces/{'__'.join(parts)}"


@dataclass
class ExtractStats:
    files_written: int = 0
    spans_written: int = 0
    bytes_written: int = 0
    members_seen: int = 0
    members_skipped: int = 0
    stopped_reason: str = ""
    written_paths: list[str] = field(default_factory=list)


@dataclass
class CapConfig:
    max_spans: int = 0
    max_files: int = 0
    max_bytes: int = 0

    def any_set(self) -> bool:
        return bool(self.max_spans or self.max_files or self.max_bytes)

    def hit(self, stats: ExtractStats) -> Optional[str]:
        if self.max_spans and stats.spans_written >= self.max_spans:
            return "max-spans"
        if self.max_files and stats.files_written >= self.max_files:
            return "max-files"
        if self.max_bytes and stats.bytes_written >= self.max_bytes:
            return "max-bytes"
        return None


def stream_extract_capped(
    archive_fh: BinaryIO,
    out_dir: str,
    caps: CapConfig,
) -> ExtractStats:
    """
    Stream-decompress ``archive_fh`` (``.tar.zst`` bytes) and write capped JSON.

    Stops as soon as any cap is hit — never materializes the full tree.
    """
    import zstandard as zstd

    stats = ExtractStats()
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "traces").mkdir(parents=True, exist_ok=True)

    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(archive_fh) as reader:
        with tarfile.open(fileobj=reader, mode="r|") as tar:
            for member in tar:
                stats.members_seen += 1
                reason = caps.hit(stats)
                if reason:
                    stats.stopped_reason = reason
                    break

                rel = _safe_member_relpath(member.name)
                if rel is None or not member.isfile():
                    stats.members_skipped += 1
                    continue

                extracted = tar.extractfile(member)
                if extracted is None:
                    stats.members_skipped += 1
                    continue
                try:
                    raw = extracted.read()
                finally:
                    extracted.close()

                try:
                    doc = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    stats.members_skipped += 1
                    continue

                n_spans = _count_spans(doc)
                if n_spans <= 0:
                    stats.members_skipped += 1
                    continue

                dest_rel = _output_relpath(rel)
                dest = out_root / dest_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                payload = raw if raw.endswith(b"\n") else raw + b"\n"
                dest.write_bytes(payload)

                stats.files_written += 1
                stats.spans_written += n_spans
                stats.bytes_written += len(payload)
                stats.written_paths.append(str(dest))

                reason = caps.hit(stats)
                if reason:
                    stats.stopped_reason = reason
                    break
            else:
                if not stats.stopped_reason:
                    stats.stopped_reason = "archive-exhausted"

    prov = {
        "tool": "corpus.ingest.tale_stream_extract",
        "source": "uber-tale-of-errors",
        "note": (
            "Capped streaming extract only. Not a full decompress. "
            "No val_bpb / AUROC invented here."
        ),
        "disk_note": DECOMPRESSED_DISK_NOTE,
        "caps": {
            "max_spans": caps.max_spans,
            "max_files": caps.max_files,
            "max_bytes": caps.max_bytes,
        },
        "stats": {
            "files_written": stats.files_written,
            "spans_written": stats.spans_written,
            "bytes_written": stats.bytes_written,
            "members_seen": stats.members_seen,
            "members_skipped": stats.members_skipped,
            "stopped_reason": stats.stopped_reason,
        },
        "claim_status": "not_a_metric — extract only; train later for factual val_bpb",
    }
    (out_root / "extract_provenance.json").write_text(
        json.dumps(prov, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def extract_from_input(
    input_path: str,
    out_dir: str,
    *,
    prefix: str = "trace1_",
    concat_out: Optional[str] = None,
    max_spans: int = 0,
    max_files: int = 0,
    max_bytes: int = 0,
) -> ExtractStats:
    caps = CapConfig(max_spans=max_spans, max_files=max_files, max_bytes=max_bytes)
    if not caps.any_set():
        raise SystemExit(
            "ERROR: Refusing uncapped extract (would chase a full decompress).\n"
            f"  {DISK_CAP_NOTE}\n"
            "  Pass at least one of: --max-spans / --max-files / --max-bytes.\n"
            "  Full zstd -d of Tale archives is OUT OF SCOPE on ~315 Gi free disks."
        )

    label, concat_reader, pieces = resolve_archive_source(
        input_path, prefix=prefix, concat_out=concat_out
    )
    print(
        f"Source: {label} ({len(pieces)} piece(s)); "
        f"caps spans={max_spans or chr(8734)} files={max_files or chr(8734)} "
        f"bytes={max_bytes or chr(8734)}",
        file=sys.stderr,
    )
    try:
        if concat_reader is not None:
            buffered = io.BufferedReader(concat_reader)
            try:
                return stream_extract_capped(buffered, out_dir, caps)
            finally:
                buffered.close()
        with open(label, "rb") as fh:
            return stream_extract_capped(fh, out_dir, caps)
    finally:
        if concat_reader is not None and not concat_reader.closed:
            concat_reader.close()


def _refuse_loud_flags(argv: list[str]) -> None:
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in _REFUSED_METRIC_FLAGS:
            raise SystemExit(
                f"ERROR: Refusing '{key}'.\n"
                "  Streaming extract is disk-safe prep for the *train* lane only.\n"
                "  Tale dumps have no AOMB incident labels → no AUROC.\n"
                "  This tool never invents val_bpb / ranking accuracy.\n"
                "  See docs/tale-val-bpb-baseline.md · docs/tale-scale.md"
            )
        if key in {"--full-decompress", "--decompress-all", "--uncapped"}:
            raise SystemExit(
                f"ERROR: Refusing '{key}'.\n"
                f"  {DECOMPRESSED_DISK_NOTE}\n"
                f"  {DISK_CAP_NOTE}\n"
                "  Use --max-spans / --max-files / --max-bytes instead."
            )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m corpus.ingest.tale_stream_extract",
        description=(
            "Stream-decompress Tale of Errors .tar.zst (or cat'd trace*_ pieces) "
            "and write a capped Jaeger JSON tree for tale_of_errors / build_shards. "
            "Never materializes the full 300–500 GB tree."
        ),
        epilog=(
            "Honesty: no AUROC, no invented val_bpb. prepare.py is sacred "
            "(invoke later on Mac; not from this module)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        help=(
            "Directory of downloaded trace1_* / trace2_* pieces, "
            "OR path to an already-concatenated .tar.zst"
        ),
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output directory (writes traces/*.json loadable by tale_of_errors)",
    )
    p.add_argument(
        "--prefix",
        default="trace1_",
        help="Piece filename prefix when --input is a directory (default: trace1_)",
    )
    p.add_argument(
        "--concat-out",
        default=None,
        metavar="PATH",
        help=(
            "Optional path to write the concatenated .tar.zst "
            "(compressed only; still not a full decompress)"
        ),
    )
    p.add_argument(
        "--max-spans",
        type=int,
        default=0,
        help="Stop after writing approximately this many spans (required unless other cap)",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Stop after writing this many Jaeger JSON files",
    )
    p.add_argument(
        "--max-bytes",
        type=int,
        default=0,
        help="Stop after writing this many bytes of JSON payload",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    _refuse_loud_flags(raw)

    args = build_parser().parse_args(raw)
    for name, val in (
        ("--max-spans", args.max_spans),
        ("--max-files", args.max_files),
        ("--max-bytes", args.max_bytes),
    ):
        if val < 0:
            print(f"ERROR: {name} must be >= 0", file=sys.stderr)
            return 2

    stats = extract_from_input(
        args.input,
        args.out,
        prefix=args.prefix,
        concat_out=args.concat_out,
        max_spans=args.max_spans,
        max_files=args.max_files,
        max_bytes=args.max_bytes,
    )
    print(
        f"OK: wrote {stats.files_written} JSON file(s), "
        f"{stats.spans_written} span(s), {stats.bytes_written} byte(s) → {args.out}"
    )
    print(f"Stopped: {stats.stopped_reason}")
    print(
        "Next: uv run python -m corpus.ingest.build_shards "
        f"--adapter tale_of_errors --input {args.out} --max-spans N ..."
    )
    print(
        "No val_bpb invented here. Mac path: prepare.py → train.py → record printed val_bpb."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
