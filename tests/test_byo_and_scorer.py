"""Tests for BYO ingest adapter + session scorer dry-run."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


class TestByoDetectFormat(unittest.TestCase):
    def test_detect_lab_sample_otlp(self):
        from corpus.ingest.adapters.byo import FORMAT_OTLP_JSONL, detect_format

        fmt = detect_format(str(ROOT / "corpus" / "fixtures" / "lab_sample"))
        self.assertEqual(fmt, FORMAT_OTLP_JSONL)

    def test_detect_crisp_sample_jaeger(self):
        from corpus.ingest.adapters.byo import FORMAT_JAEGER_JSON, detect_format

        fmt = detect_format(str(ROOT / "corpus" / "fixtures" / "crisp_sample"))
        self.assertEqual(fmt, FORMAT_JAEGER_JSON)


class TestByoLabSample(unittest.TestCase):
    def test_byo_loads_lab_sample_and_builds_shards(self):
        from corpus.ingest.adapters.byo import ByoAdapter
        from corpus.ingest.build_shards import build

        fixture = ROOT / "corpus" / "fixtures" / "lab_sample"
        sessions = list(ByoAdapter().iter_sessions(str(fixture)))
        self.assertGreaterEqual(len(sessions), 2)
        texts = [t for t, _w, _b in sessions]
        self.assertTrue(any("status=error" in t for t in texts))
        bundle = sessions[0][2]
        self.assertEqual(bundle.source_kind, "byo")
        self.assertTrue((bundle.extra_provenance or {}).get("byo"))

        with tempfile.TemporaryDirectory() as tmp:
            prov = build(
                "byo",
                str(fixture),
                num_train_shards=2,
                write_val_shard=True,
                data_dir=tmp,
            )
            self.assertEqual(prov["adapter"], "byo")
            self.assertGreater(prov["session_count"], 0)
            self.assertTrue(os.path.exists(os.path.join(tmp, "shard_00000.parquet")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "shard_06542.parquet")))
            table = pq.read_table(os.path.join(tmp, "shard_00000.parquet"))
            self.assertEqual(table.column_names, ["text"])


class TestByoCrispSample(unittest.TestCase):
    def test_byo_loads_crisp_jaeger_fixture(self):
        from corpus.ingest.adapters.byo import ByoAdapter
        from corpus.ingest.build_shards import build

        fixture = ROOT / "corpus" / "fixtures" / "crisp_sample"
        sessions = list(ByoAdapter().iter_sessions(str(fixture)))
        self.assertEqual(len(sessions), 2)
        joined = "\n".join(t for t, _w, _b in sessions)
        self.assertIn("svc=service-1", joined)
        self.assertIn("trace_id=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", joined)

        with tempfile.TemporaryDirectory() as tmp:
            prov = build(
                "byo",
                str(fixture),
                num_train_shards=1,
                write_val_shard=True,
                data_dir=tmp,
            )
            self.assertEqual(prov["adapter"], "byo")
            self.assertEqual(prov["sources"][0]["source_kind"], "byo")
            table = pq.read_table(os.path.join(tmp, "shard_00000.parquet"))
            self.assertEqual(table.column_names, ["text"])
            self.assertGreater(table.num_rows, 0)


class TestByoParquetPassthrough(unittest.TestCase):
    def test_parquet_text_column_roundtrip(self):
        from corpus.ingest.adapters.byo import (
            FORMAT_PARQUET_SESSIONS,
            ByoAdapter,
            detect_format,
        )
        from corpus.ingest.build_shards import build

        docs = [
            "# aomb_meta source=byo-test window=unknown\n"
            "[ts=2026-09-11T17:01:00.100Z] [src=OTel] trace_id=abc "
            "span_id=1 parent=n/a op=GET svc=api duration_ms=10 status=ok",
            "# aomb_meta source=byo-test window=unknown\n"
            "[ts=2026-09-11T17:02:00.100Z] [src=OTel] trace_id=def "
            "span_id=2 parent=n/a op=POST svc=api duration_ms=99 status=error",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            pq_path = os.path.join(tmp, "shard_00000.parquet")
            pq.write_table(
                pa.table({"text": pa.array(docs, type=pa.string())}),
                pq_path,
            )
            self.assertEqual(detect_format(tmp), FORMAT_PARQUET_SESSIONS)
            sessions = list(ByoAdapter().iter_sessions(tmp))
            self.assertEqual(len(sessions), 2)
            self.assertIn("status=error", sessions[1][0])

            out = os.path.join(tmp, "out")
            prov = build(
                "byo",
                tmp,
                num_train_shards=1,
                write_val_shard=False,
                data_dir=out,
            )
            self.assertEqual(prov["session_count"], 2)
            self.assertEqual(
                prov["sources"][0]["extra"].get("format"),
                FORMAT_PARQUET_SESSIONS,
            )


class TestScoreCliDryRun(unittest.TestCase):
    def test_dry_run_lab_sample(self):
        from eval.score_cli import load_session_texts, main

        fixture = str(ROOT / "corpus" / "fixtures" / "lab_sample")
        rows, meta = load_session_texts(fixture)
        self.assertGreaterEqual(len(rows), 2)
        self.assertIn(meta.get("loader"), {"lab_capture", "byo"})

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "score.json")
            rc = main(["--input", fixture, "--dry-run", "--out", out, "--json"])
            self.assertEqual(rc, 0)
            report = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertEqual(report["claim_status"], "not_published")
            self.assertEqual(report["train_meta"]["mode"], "dry_run")
            self.assertGreaterEqual(len(report["sessions"]), 2)
            self.assertTrue(all(s["bpb"] is None for s in report["sessions"]))

    def test_dry_run_crisp_sample(self):
        from eval.score_cli import main

        fixture = str(ROOT / "corpus" / "fixtures" / "crisp_sample")
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "score.json")
            rc = main(["--input", fixture, "--dry-run", "--out", out])
            self.assertEqual(rc, 0)
            report = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertEqual(report["claim_status"], "not_published")
            self.assertEqual(len(report["sessions"]), 2)

    def test_module_score_session_importable(self):
        import score_session

        self.assertTrue(callable(score_session.main))


class TestExistingAdaptersUnbroken(unittest.TestCase):
    def test_lab_capture_still_registered(self):
        from corpus.ingest.build_shards import ADAPTERS

        self.assertIn("lab_capture", ADAPTERS)
        self.assertIn("crisp_zenodo", ADAPTERS)
        self.assertIn("byo", ADAPTERS)


if __name__ == "__main__":
    unittest.main()
