"""Integration: synthetic .tar.zst → tale_stream_extract → build_shards.

No Zenodo. No MPS. No invented AUROC / val_bpb. prepare.py untouched.
"""

from __future__ import annotations

import contextlib
import io
import json
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


class TestBuildShardsCLIRefusals(unittest.TestCase):
    def test_refuse_auroc_and_invent_flags(self):
        from corpus.ingest.build_shards import EXIT_REFUSED_FLAG, main

        for flag in (
            "--auroc",
            "--lab-auroc",
            "--invent-metrics",
            "--invent-val-bpb",
            "--ranking",
            "--publish",
        ):
            with self.assertRaises(SystemExit) as ctx:
                main(
                    [
                        flag,
                        "--adapter",
                        "tale_of_errors",
                        "--input",
                        "/tmp",
                        "--max-spans",
                        "1",
                    ]
                )
            self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG, flag)

    def test_refuse_uncap_flags(self):
        from corpus.ingest.build_shards import EXIT_REFUSED_FLAG, main

        for flag in ("--uncapped", "--full-decompress", "--download-all"):
            with self.assertRaises(SystemExit) as ctx:
                main(
                    [
                        flag,
                        "--adapter",
                        "tale_of_errors",
                        "--input",
                        "/tmp",
                        "--max-spans",
                        "1",
                    ]
                )
            self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG, flag)

    def test_refuse_uncapped_tale_without_max_spans(self):
        from corpus.ingest.build_shards import EXIT_USAGE, main

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            with self.assertRaises(SystemExit) as ctx:
                main(
                    [
                        "--adapter",
                        "tale_of_errors",
                        "--input",
                        str(ROOT / "corpus" / "fixtures" / "tale_of_errors_sample"),
                        "--num-train-shards",
                        "1",
                    ]
                )
        self.assertEqual(ctx.exception.code, EXIT_USAGE)
        err = buf.getvalue().lower()
        self.assertIn("max-spans", err)
        self.assertIn("uncapped", err)
        self.assertNotIn("val_bpb=", err)
        self.assertNotIn("auroc=", err)

    def test_lab_capture_still_allows_uncapped(self):
        """Other adapters may omit --max-spans (small fixtures)."""
        from corpus.ingest.build_shards import main

        fixture = ROOT / "corpus" / "fixtures" / "lab_sample"
        self.assertTrue(fixture.is_dir())
        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "--adapter",
                    "lab_capture",
                    "--input",
                    str(fixture),
                    "--num-train-shards",
                    "1",
                    "--data-dir",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)
            shards = list(Path(tmp).glob("shard_*.parquet"))
            self.assertGreaterEqual(len(shards), 1)

    def test_no_auroc_calculator_in_module(self):
        src = (ROOT / "corpus" / "ingest" / "build_shards.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("def compute_auroc", src)
        self.assertNotIn("sklearn", src)
        self.assertNotIn("roc_auc", src)
        self.assertIn("OUT OF SCOPE", src)
        self.assertIn("val_bpb", src.lower())
        self.assertNotRegex(src.lower(), r"val_bpb\s*=\s*[0-9]")


class TestStreamExtractToBuildShards(unittest.TestCase):
    def test_synthetic_tar_zst_extract_then_shards_early_stop(self):
        from corpus.ingest.adapters.tale_of_errors import TaleOfErrorsAdapter
        from corpus.ingest.build_shards import main
        from corpus.ingest.tale_stream_extract import extract_from_input

        members: dict[str, bytes] = {}
        for i in range(12):
            tid = f"{i:032x}"
            members[f"traces/trace_{i:02d}.json"] = (
                json.dumps(_jaeger_doc(tid, n_spans=4)).encode("utf-8") + b"\n"
            )
        members["package.json"] = b'{"name":"skip"}\n'
        members["readme.txt"] = b"not json\n"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _build_tar_zst(
                members, tmp_path / "trace1-sanitized.tar.zst"
            )
            extract_dir = tmp_path / "extracted"
            data_dir = tmp_path / "shards"

            stats = extract_from_input(
                str(archive),
                str(extract_dir),
                max_files=4,
                max_spans=0,
                max_bytes=0,
            )
            self.assertEqual(stats.files_written, 4)
            self.assertEqual(stats.stopped_reason, "max-files")
            self.assertLess(stats.members_seen, 12 + 2)

            written = list((extract_dir / "traces").rglob("*.json"))
            self.assertEqual(len(written), 4)

            bundles = list(
                TaleOfErrorsAdapter().load(str(extract_dir), max_spans=7)
            )
            self.assertEqual(len(bundles), 1)
            self.assertEqual(bundles[0].source_id, "uber-tale-of-errors")
            self.assertLessEqual(len(bundles[0].spans), 7)
            self.assertGreater(len(bundles[0].spans), 0)

            rc = main(
                [
                    "--adapter",
                    "tale_of_errors",
                    "--input",
                    str(extract_dir),
                    "--max-spans",
                    "7",
                    "--num-train-shards",
                    "2",
                    "--write-val-shard",
                    "--data-dir",
                    str(data_dir),
                ]
            )
            self.assertEqual(rc, 0)
            train0 = data_dir / "shard_00000.parquet"
            train1 = data_dir / "shard_00001.parquet"
            val = data_dir / "shard_06542.parquet"
            self.assertTrue(train0.is_file())
            self.assertTrue(train1.is_file())
            self.assertTrue(val.is_file())

            import pyarrow.parquet as pq

            n_rows = sum(
                pq.read_table(p).num_rows for p in (train0, train1, val)
            )
            self.assertGreater(n_rows, 0)
            self.assertLessEqual(n_rows, 8)

    def test_span_cap_stops_extract_before_shard_build(self):
        from corpus.ingest.build_shards import main
        from corpus.ingest.tale_stream_extract import extract_from_input

        members = {
            f"traces/t{i:02d}.json": json.dumps(
                _jaeger_doc(f"{i:032x}", n_spans=5)
            ).encode()
            for i in range(10)
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _build_tar_zst(members, tmp_path / "full.tar.zst")
            out = tmp_path / "out"
            stats = extract_from_input(str(archive), str(out), max_spans=6)
            self.assertEqual(stats.stopped_reason, "max-spans")
            self.assertGreaterEqual(stats.spans_written, 6)
            self.assertLessEqual(stats.files_written, 3)

            data_dir = tmp_path / "data"
            rc = main(
                [
                    "--adapter",
                    "tale_of_errors",
                    "--input",
                    str(out),
                    "--max-spans",
                    "6",
                    "--num-train-shards",
                    "1",
                    "--data-dir",
                    str(data_dir),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue((data_dir / "shard_00000.parquet").is_file())
            self.assertFalse((data_dir / "shard_00001.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
