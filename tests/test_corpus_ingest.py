"""Unit tests for Jaeger/CRISP → AOMB session conversion and lab adapter."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


class TestSessionFormat(unittest.TestCase):
    def test_span_line_shape(self):
        from corpus.ingest.session_format import span_event_line

        line = span_event_line(
            ts="2026-09-11T17:01:00.100Z",
            trace_id="abc",
            span_id="def",
            parent="n/a",
            op="GET_/api",
            svc="api",
            duration_ms=42,
            status="ok",
            http_status=200,
        )
        self.assertIn("[src=OTel]", line)
        self.assertIn("trace_id=abc", line)
        self.assertIn("duration_ms=42", line)


class TestLabFixtureIngest(unittest.TestCase):
    def test_lab_sample_to_sessions_and_shards(self):
        from corpus.ingest.adapters.lab_capture import LabCaptureAdapter
        from corpus.ingest.build_shards import build
        from corpus.ingest.otlp_to_sessions import bundle_to_sessions

        fixture = ROOT / "corpus" / "fixtures" / "lab_sample"
        bundles = list(LabCaptureAdapter().load(str(fixture)))
        self.assertEqual(len(bundles), 1)
        sessions = bundle_to_sessions(bundles[0])
        self.assertGreaterEqual(len(sessions), 2)
        labels = {w.label for _, w in sessions}
        self.assertIn("normal", labels)
        self.assertIn("incident", labels)
        incident_text = next(t for t, w in sessions if w.label == "incident")
        self.assertIn("status=error", incident_text)

        with tempfile.TemporaryDirectory() as tmp:
            prov = build(
                "lab_capture",
                str(fixture),
                num_train_shards=2,
                write_val_shard=True,
                data_dir=tmp,
            )
            self.assertGreater(prov["session_count"], 0)
            self.assertTrue(os.path.exists(os.path.join(tmp, "shard_00000.parquet")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "shard_06542.parquet")))
            table = pq.read_table(os.path.join(tmp, "shard_00000.parquet"))
            self.assertEqual(table.column_names, ["text"])
            self.assertGreater(table.num_rows, 0)


class TestCrispJaegerIngest(unittest.TestCase):
    def test_crisp_sample_sessions_by_trace_id(self):
        from corpus.ingest.adapters.crisp_zenodo import CrispZenodoAdapter
        from corpus.ingest.build_shards import build
        from corpus.ingest.otlp_to_sessions import bundle_to_sessions

        fixture = ROOT / "corpus" / "fixtures" / "crisp_sample"
        bundles = list(CrispZenodoAdapter().load(str(fixture)))
        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0].source_id, "uber-crisp-zenodo-13956078")
        self.assertEqual(bundles[0].license, "CC-BY-4.0")
        sessions = bundle_to_sessions(bundles[0])
        # Two traces → two sessions
        self.assertEqual(len(sessions), 2)
        texts = [t for t, _ in sessions]
        joined = "\n".join(texts)
        self.assertIn("svc=service-1", joined)
        self.assertIn("svc=service-2", joined)
        self.assertIn("svc=service-3", joined)
        self.assertIn("source=uber-crisp-zenodo-13956078", joined)
        # Multi-service session shares one trace_id
        self.assertTrue(any("trace_id=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in t for t in texts))

        with tempfile.TemporaryDirectory() as tmp:
            prov = build(
                "crisp_zenodo",
                str(fixture),
                num_train_shards=1,
                write_val_shard=True,
                data_dir=tmp,
            )
            self.assertEqual(prov["sources"][0]["license"], "CC-BY-4.0")
            table = pq.read_table(os.path.join(tmp, "shard_00000.parquet"))
            self.assertEqual(table.column_names, ["text"])


class TestRejectedSourcesListed(unittest.TestCase):
    def test_denylist_mentions_otel_demo(self):
        from corpus.ingest.rejected_sources import REJECTED_PUBLIC_SOURCES

        ids = " ".join(r["id"] for r in REJECTED_PUBLIC_SOURCES)
        self.assertIn("otel-demo", ids)
        self.assertIn("DeathStarBench", ids)


if __name__ == "__main__":
    unittest.main()
