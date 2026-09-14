"""Tests for public accuracy eval metrics + lab provenance label parsing."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestBinaryLabels(unittest.TestCase):
    def test_binary_from_label(self):
        from eval.labels import binary_from_label

        self.assertEqual(binary_from_label("normal"), 0)
        self.assertEqual(binary_from_label("incident"), 1)
        self.assertEqual(binary_from_label("cascade"), 1)
        self.assertEqual(binary_from_label("anomalous"), 1)
        self.assertIsNone(binary_from_label("unknown"))
        self.assertIsNone(binary_from_label(""))


class TestLabSampleLabels(unittest.TestCase):
    def test_fixture_has_normal_and_incident(self):
        from eval.labels import filter_scorable, load_lab_sessions, parse_windows

        fixture = ROOT / "corpus" / "fixtures" / "lab_sample"
        sessions, meta = load_lab_sessions(fixture)
        self.assertGreaterEqual(len(sessions), 2)
        self.assertTrue(meta["content_sha256"])
        self.assertEqual(meta["capture_id"], "fixture-demo")

        windows = parse_windows(
            json.loads((fixture / "provenance.json").read_text(encoding="utf-8"))
        )
        labels = {w["label"] for w in windows}
        self.assertIn("normal", labels)
        self.assertIn("incident", labels)

        y_true, kept = filter_scorable(sessions)
        self.assertEqual(len(y_true), len(kept))
        self.assertIn(0, y_true)
        self.assertIn(1, y_true)
        self.assertEqual(set(s.label for s in kept), {"normal", "incident"})


class TestOTelProtoJsonTimestampLabels(unittest.TestCase):
    """Regression: real lab captures use OTel ProtoJSON string nanos, not RFC3339."""

    def test_parse_telemetry_timestamp_units_and_strings(self):
        from datetime import datetime, timezone

        from corpus.ingest.timestamps import parse_telemetry_timestamp

        # ProtoJSON fixed64 string — must NOT become year 1726 via fromisoformat.
        dt = parse_telemetry_timestamp("1726060860100000000")
        self.assertIsNotNone(dt)
        assert dt is not None
        self.assertEqual(dt.year, 2024)
        self.assertEqual(dt, datetime(2024, 9, 11, 13, 21, 0, 100000, tzinfo=timezone.utc))

        self.assertEqual(
            parse_telemetry_timestamp(1726060860100000000),
            parse_telemetry_timestamp("1726060860100000000"),
        )
        # ms / s magnitudes (previously misclassified as µs / ms)
        self.assertEqual(parse_telemetry_timestamp(1726060860100).year, 2024)
        self.assertEqual(parse_telemetry_timestamp(1726060860).year, 2024)
        self.assertEqual(parse_telemetry_timestamp("1726060860").year, 2024)
        self.assertEqual(
            parse_telemetry_timestamp("2026-09-11T17:01:00.100Z").year, 2026
        )

    def test_synthetic_otel_jsonl_labels_normal_and_incident(self):
        from eval.labels import filter_scorable, load_lab_sessions
        from eval.run_eval import main

        # Windows: 2024-09-11 13:20–13:22 normal, 13:25–13:27 incident (UTC)
        normal_ns = "1726060860100000000"  # 13:21:00.100Z
        incident_ns = "1726061160000000000"  # 13:26:00.000Z
        normal_end_ns = "1726060860500000000"
        incident_end_ns = "1726061160900000000"

        with tempfile.TemporaryDirectory() as tmp:
            cap = Path(tmp) / "otel_protojson_capture"
            cap.mkdir()
            (cap / "provenance.json").write_text(
                json.dumps(
                    {
                        "corpus_version": "v1",
                        "source_id": "lab-aomb-stack",
                        "source_kind": "lab_capture",
                        "license": "Apache-2.0",
                        "capture_id": "synthetic-otel-nanos",
                        "windows": [
                            {
                                "label": "normal",
                                "start": "2024-09-11T13:20:00Z",
                                "end": "2024-09-11T13:22:00Z",
                                "fault": "",
                            },
                            {
                                "label": "incident",
                                "start": "2024-09-11T13:25:00Z",
                                "end": "2024-09-11T13:27:00Z",
                                "fault": "api_latency",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            # Mimic otel-collector file exporter ProtoJSON (resourceSpans / string nanos)
            traces = [
                {
                    "resourceSpans": [
                        {
                            "resource": {
                                "attributes": [
                                    {
                                        "key": "service.name",
                                        "value": {"stringValue": "api"},
                                    }
                                ]
                            },
                            "scopeSpans": [
                                {
                                    "spans": [
                                        {
                                            "traceId": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                                            "spanId": "1111111111111111",
                                            "name": "GET /health",
                                            "startTimeUnixNano": normal_ns,
                                            "endTimeUnixNano": normal_end_ns,
                                            "status": {"code": 1},
                                        }
                                    ]
                                }
                            ],
                        }
                    ]
                },
                {
                    "resourceSpans": [
                        {
                            "resource": {
                                "attributes": [
                                    {
                                        "key": "service.name",
                                        "value": {"stringValue": "api"},
                                    }
                                ]
                            },
                            "scopeSpans": [
                                {
                                    "spans": [
                                        {
                                            "traceId": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                                            "spanId": "2222222222222222",
                                            "name": "GET /checkout",
                                            "startTimeUnixNano": incident_ns,
                                            "endTimeUnixNano": incident_end_ns,
                                            "status": {"code": 2, "message": "error"},
                                        }
                                    ]
                                }
                            ],
                        }
                    ]
                },
            ]
            logs = [
                {
                    "resourceLogs": [
                        {
                            "resource": {
                                "attributes": [
                                    {
                                        "key": "service.name",
                                        "value": {"stringValue": "api"},
                                    }
                                ]
                            },
                            "scopeLogs": [
                                {
                                    "logRecords": [
                                        {
                                            "timeUnixNano": normal_ns,
                                            "severityText": "INFO",
                                            "body": {"stringValue": "ok"},
                                            "traceId": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                                        },
                                        {
                                            "timeUnixNano": incident_ns,
                                            "severityText": "ERROR",
                                            "body": {"stringValue": "checkout_failed"},
                                            "traceId": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                                        },
                                    ]
                                }
                            ],
                        }
                    ]
                },
            ]
            (cap / "traces.jsonl").write_text(
                "\n".join(json.dumps(t) for t in traces) + "\n", encoding="utf-8"
            )
            (cap / "logs.jsonl").write_text(
                "\n".join(json.dumps(t) for t in logs) + "\n", encoding="utf-8"
            )

            sessions, meta = load_lab_sessions(cap)
            self.assertEqual(meta["capture_id"], "synthetic-otel-nanos")
            self.assertEqual(meta.get("n_events_missing_timestamp"), 0)
            labels = {s.label for s in sessions}
            self.assertIn("normal", labels)
            self.assertIn("incident", labels)
            self.assertNotIn("unknown", labels)

            y_true, kept = filter_scorable(sessions)
            self.assertIn(0, y_true)
            self.assertIn(1, y_true)

            out = Path(tmp) / "out"
            rc = main(
                [
                    "--capture",
                    str(cap),
                    "--scores-from",
                    "length",
                    "--seed",
                    "0",
                    "--out-dir",
                    str(out),
                    "--random-draws",
                    "8",
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["claim_status"], "not_published")

    def test_missing_timestamp_does_not_invent_now_in_session_text(self):
        from datetime import datetime, timezone

        from corpus.ingest.adapters.base import SpanRecord, SourceBundle, TimeWindow
        from corpus.ingest.otlp_to_sessions import bundle_to_sessions

        bundle = SourceBundle(
            source_id="t",
            source_kind="lab_capture",
            license="Apache-2.0",
            spans=[
                SpanRecord(
                    trace_id="c" * 32,
                    span_id="d" * 16,
                    name="x",
                    service_name="api",
                    start_time=None,
                )
            ],
            windows=[
                TimeWindow(
                    label="normal",
                    start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    end=datetime(2024, 1, 2, tzinfo=timezone.utc),
                )
            ],
            capture_id="x",
        )
        sessions = bundle_to_sessions(bundle)
        self.assertEqual(len(sessions), 1)
        text, window = sessions[0]
        self.assertEqual(window.label, "unknown")
        self.assertIn("ts=n/a", text)
        self.assertNotRegex(text, r"ts=20\d{2}-")


class TestMetrics(unittest.TestCase):
    def test_perfect_separation_auroc(self):
        from eval.metrics import auroc, pr_auc, precision_at_k

        y = [0, 0, 0, 1, 1, 1]
        s = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        self.assertAlmostEqual(auroc(y, s), 1.0)
        self.assertAlmostEqual(pr_auc(y, s), 1.0)
        self.assertAlmostEqual(precision_at_k(y, s, 3), 1.0)

    def test_inverted_scores_auroc_zero(self):
        from eval.metrics import auroc

        y = [0, 0, 1, 1]
        s = [0.9, 0.8, 0.2, 0.1]
        self.assertAlmostEqual(auroc(y, s), 0.0)

    def test_single_class_nan(self):
        from eval.metrics import auroc, pr_auc

        self.assertTrue(math.isnan(auroc([0, 0, 0], [0.1, 0.2, 0.3])))
        self.assertTrue(math.isnan(pr_auc([0, 0, 0], [0.1, 0.2, 0.3])))

    def test_random_baseline_near_half(self):
        from eval.metrics import random_baseline_metrics

        y = [0] * 20 + [1] * 20
        rb = random_baseline_metrics(y, n_draws=200, seed=0)
        self.assertGreater(rb["auroc"]["n"], 0)
        self.assertAlmostEqual(rb["auroc"]["mean"], 0.5, delta=0.08)

    def test_summarize_includes_baseline(self):
        from eval.metrics import summarize_ranking

        y = [0, 0, 1, 1]
        s = [0.1, 0.2, 0.8, 0.9]
        out = summarize_ranking(y, s, random_draws=8, random_seed=1)
        self.assertIn("auroc", out)
        self.assertIn("pr_auc", out)
        self.assertIn("precision_at_k", out)
        self.assertIn("random_baseline", out)
        self.assertGreater(out["auroc"], 0.9)


class TestReportAndHarnessSmoke(unittest.TestCase):
    def test_run_eval_length_baseline_on_fixture(self):
        from eval.run_eval import main

        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "--capture",
                    str(ROOT / "corpus" / "fixtures" / "lab_sample"),
                    "--scores-from",
                    "length",
                    "--seed",
                    "0",
                    "--out-dir",
                    tmp,
                    "--random-draws",
                    "16",
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(Path(tmp, "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["claim_status"], "not_published")
            self.assertIn("0.458756", report["disclaimer"])
            self.assertIn("auroc", report["metrics"])
            self.assertIn("random_baseline", report["metrics"])
            self.assertTrue((Path(tmp) / "report.md").is_file())

    def test_aggregate_mean_std(self):
        from eval.report import aggregate_seed_reports, render_aggregate_markdown

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            paths = []
            for seed, auroc in [(0, 0.6), (1, 0.7), (2, 0.8)]:
                d = tmp_path / f"seed-{seed}"
                d.mkdir()
                report = {
                    "seed": seed,
                    "score_method": "length_baseline",
                    "git": {"head": "abc"},
                    "metrics": {
                        "auroc": auroc,
                        "pr_auc": auroc - 0.05,
                        "precision_at_k": {"1": 1.0},
                    },
                }
                p = d / "report.json"
                p.write_text(json.dumps(report), encoding="utf-8")
                paths.append(p)
            agg = aggregate_seed_reports(paths)
            self.assertEqual(agg["n_seeds"], 3)
            self.assertAlmostEqual(agg["metrics_mean_std"]["auroc"]["mean"], 0.7)
            self.assertEqual(agg["claim_status"], "not_published")
            md = render_aggregate_markdown(agg)
            self.assertIn("mean ± std", md)


class TestScoreResolve(unittest.TestCase):
    def test_length_and_precomputed(self):
        from eval.labels import LabeledSession
        from eval.score import resolve_scores

        sessions = [
            LabeledSession("a", "xx", "normal", 0, n_chars=2),
            LabeledSession("b", "yyyy", "incident", 1, n_chars=4),
        ]
        scores, method = resolve_scores(sessions, scores_from="length")
        self.assertEqual(method, "length_baseline")
        self.assertEqual(scores, [2.0, 4.0])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.json"
            path.write_text(json.dumps({"scores": {"a": 0.1, "b": 0.9}}), encoding="utf-8")
            scores2, method2 = resolve_scores(
                sessions, scores_from="precomputed", precomputed_path=path
            )
            self.assertEqual(method2, "precomputed")
            self.assertEqual(scores2, [0.1, 0.9])


if __name__ == "__main__":
    unittest.main()
