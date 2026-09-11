"""Unit tests for OTLP → AOMB session conversion and lab adapter."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
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
        # Incident session should mention error status somewhere
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


class TestOtelDemoParquetAdapter(unittest.TestCase):
    def test_minimal_parquet_roundtrip(self):
        from corpus.ingest.adapters.otel_demo_hf import OtelDemoHfAdapter
        from corpus.ingest.otlp_to_sessions import sessions_only

        with tempfile.TemporaryDirectory() as tmp:
            traces_dir = os.path.join(tmp, "otlp_traces", "year=2026")
            os.makedirs(traces_dir)
            start = datetime(2026, 7, 1, 20, 9, 40, tzinfo=timezone.utc)
            table = pa.table(
                {
                    "start_time_unix_nano": pa.array([start]),
                    "duration_time_unix_nano": pa.array([42_000_000]),
                    "trace_id": pa.array(["0dc6f50a3525ccd0980f986ac70c4cdf"]),
                    "span_id": pa.array(["33231253437d02b5"]),
                    "parent_span_id": pa.array([None], type=pa.string()),
                    "service_name": pa.array(["payment"]),
                    "name": pa.array(["GET /status"]),
                    "status_code": pa.array([0], type=pa.int32()),
                    "status_status_message": pa.array([None], type=pa.string()),
                    "span_attributes": pa.array(
                        ['{"http.method":"GET","http.status_code":200}']
                    ),
                }
            )
            pq.write_table(table, os.path.join(traces_dir, "sample.parquet"))

            bundles = list(OtelDemoHfAdapter().load(tmp))
            docs = sessions_only(bundles[0])
            self.assertEqual(len(docs), 1)
            self.assertIn("[src=OTel]", docs[0])
            self.assertIn("svc=payment", docs[0])
            self.assertIn("source=smithclay/otel-demo-telemetry", docs[0])


if __name__ == "__main__":
    unittest.main()
