"""Tests for streaming capped Tale of Errors extract (synthetic .tar.zst)."""

from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _jaeger_doc(trace_id: str, n_spans: int = 2) -> dict:
    spans = []
    for i in range(n_spans):
        spans.append(
            {
                "traceID": trace_id,
                "spanID": f"{i:016x}",
                "operationName": f"op-{i}",
                "references": [],
                "startTime": 1_694_430_060_100_000 + i * 1000,
                "duration": 1000 + i,
                "tags": [],
                "logs": [],
                "processID": "p1",
            }
        )
    return {
        "data": [
            {
                "traceID": trace_id,
                "spans": spans,
                "processes": {"p1": {"serviceName": "svc-a", "tags": []}},
            }
        ]
    }


def _build_tar_zst(members: dict[str, bytes], dest: Path) -> Path:
    import zstandard as zstd

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    tar_buf.seek(0)
    cctx = zstd.ZstdCompressor(level=3)
    dest.write_bytes(cctx.compress(tar_buf.read()))
    return dest


class TestTaleStreamExtract(unittest.TestCase):
    def test_caps_stop_early_and_output_loadable(self):
        from corpus.ingest.adapters.tale_of_errors import TaleOfErrorsAdapter
        from corpus.ingest.tale_stream_extract import extract_from_input

        members = {}
        for i in range(8):
            tid = f"{i:032x}"
            doc = _jaeger_doc(tid, n_spans=3)
            members[f"traces/trace_{i:02d}.json"] = (
                json.dumps(doc).encode("utf-8") + b"\n"
            )
        # Non-jaeger / skip noise
        members["package.json"] = b'{"name":"nope"}\n'
        members["readme.txt"] = b"not json\n"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _build_tar_zst(members, tmp_path / "trace1-sanitized.tar.zst")
            out_dir = tmp_path / "out"

            stats = extract_from_input(
                str(archive),
                str(out_dir),
                max_files=3,
                max_spans=0,
                max_bytes=0,
            )
            self.assertEqual(stats.files_written, 3)
            self.assertEqual(stats.stopped_reason, "max-files")
            self.assertLess(stats.members_seen, 8 + 2)  # stopped before exhausting
            self.assertTrue((out_dir / "traces").is_dir())
            written = list((out_dir / "traces").rglob("*.json"))
            self.assertEqual(len(written), 3)
            self.assertTrue((out_dir / "extract_provenance.json").is_file())

            # Adapter can load the capped tree (same discovery as CRISP).
            bundles = list(TaleOfErrorsAdapter().load(str(out_dir), max_spans=100))
            self.assertEqual(len(bundles), 1)
            self.assertEqual(bundles[0].source_id, "uber-tale-of-errors")
            self.assertGreater(len(bundles[0].spans), 0)
            # 3 files × 3 spans
            self.assertEqual(len(bundles[0].spans), 9)

            # Span cap also stops early.
            out2 = tmp_path / "out2"
            stats2 = extract_from_input(
                str(archive),
                str(out2),
                max_spans=5,
            )
            self.assertEqual(stats2.stopped_reason, "max-spans")
            self.assertGreaterEqual(stats2.spans_written, 5)
            self.assertLessEqual(stats2.files_written, 3)

    def test_concat_pieces_without_full_decompress(self):
        from corpus.ingest.tale_stream_extract import (
            discover_trace_pieces,
            extract_from_input,
        )

        members = {}
        for i in range(4):
            tid = f"aa{i:030x}"
            members[f"traces/t{i}.json"] = json.dumps(_jaeger_doc(tid, 2)).encode()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _build_tar_zst(members, tmp_path / "full.tar.zst")
            raw = archive.read_bytes()
            # Split into fake Zenodo pieces
            mid = len(raw) // 2
            pieces_dir = tmp_path / "pieces"
            pieces_dir.mkdir()
            (pieces_dir / "trace1_aa").write_bytes(raw[:mid])
            (pieces_dir / "trace1_ab").write_bytes(raw[mid:])

            found = discover_trace_pieces(str(pieces_dir), prefix="trace1_")
            self.assertEqual(len(found), 2)

            out_dir = tmp_path / "out"
            concat_path = tmp_path / "trace1-sanitized.tar.zst"
            stats = extract_from_input(
                str(pieces_dir),
                str(out_dir),
                prefix="trace1_",
                concat_out=str(concat_path),
                max_files=2,
            )
            self.assertTrue(concat_path.is_file())
            self.assertEqual(stats.files_written, 2)
            self.assertEqual(stats.stopped_reason, "max-files")
            # Streaming concat path (no concat_out) also works.
            out3 = tmp_path / "out3"
            stats3 = extract_from_input(
                str(pieces_dir),
                str(out3),
                prefix="trace1_",
                max_files=1,
            )
            self.assertEqual(stats3.files_written, 1)

    def test_refuses_uncapped_and_auroc(self):
        from corpus.ingest.tale_stream_extract import extract_from_input, main

        with self.assertRaises(SystemExit) as ctx:
            extract_from_input("/tmp", "/tmp/out")
        self.assertIn("uncapped", str(ctx.exception).lower())

        with self.assertRaises(SystemExit) as ctx2:
            main(["--auroc", "--input", "/tmp", "--out", "/tmp/x", "--max-files", "1"])
        msg = str(ctx2.exception).lower()
        self.assertIn("auroc", msg)
        self.assertNotIn("val_bpb=", msg)  # never invent a measured number

        with self.assertRaises(SystemExit) as ctx3:
            main(
                [
                    "--full-decompress",
                    "--input",
                    "/tmp",
                    "--out",
                    "/tmp/x",
                    "--max-files",
                    "1",
                ]
            )
        self.assertIn("full", str(ctx3.exception).lower())

    def test_no_auroc_in_module_surface(self):
        """Guard: extractor must not grow AUROC / invented-metric entry points."""
        src = (ROOT / "corpus" / "ingest" / "tale_stream_extract.py").read_text(
            encoding="utf-8"
        )
        # Refusals may mention AUROC; must not implement an auroc calculator.
        self.assertNotIn("def compute_auroc", src)
        self.assertNotIn("sklearn", src)
        self.assertNotIn("roc_auc", src)
        self.assertIn("OUT OF SCOPE", src)
        self.assertIn("val_bpb", src.lower())  # honesty wording present


if __name__ == "__main__":
    unittest.main()
