"""Tests for public ranking card v1 fixture + reproduce entry."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "corpus" / "fixtures" / "public_ranking_card_v1"


class TestPublicRankingCardFixture(unittest.TestCase):
    def test_fixture_has_normal_and_positives(self):
        from eval.labels import filter_scorable, load_lab_sessions

        sessions, meta = load_lab_sessions(FIXTURE)
        self.assertEqual(meta["capture_id"], "public_ranking_card_v1")
        self.assertTrue(meta["content_sha256"])
        y_true, kept = filter_scorable(sessions)
        self.assertGreaterEqual(len(kept), 8)
        self.assertIn(0, y_true)
        self.assertIn(1, y_true)
        labels = {s.label for s in kept}
        self.assertIn("normal", labels)
        self.assertTrue(labels & {"incident", "cascade"})

    def test_length_baseline_smoke(self):
        from eval.run_eval import main

        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "--capture",
                    str(FIXTURE),
                    "--scores-from",
                    "length",
                    "--seed",
                    "0",
                    "--out-dir",
                    tmp,
                    "--random-draws",
                    "8",
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(Path(tmp, "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["claim_status"], "not_published")
            self.assertIn("auroc", report["metrics"])
            # Must not be a single-class toy that collapses ranking
            self.assertGreaterEqual(report["metrics"]["n"], 8)
            self.assertGreaterEqual(report["metrics"]["n_positive"], 2)
            self.assertGreaterEqual(report["metrics"]["n_negative"], 2)


class TestPublicRankingCardReproduce(unittest.TestCase):
    def test_run_baselines_writes_card(self):
        from eval.run_public_ranking_card import main

        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "--out-dir",
                    tmp,
                    "--seeds",
                    "0..2",
                    "--random-draws",
                    "8",
                ]
            )
            self.assertEqual(rc, 0)
            card = Path(tmp) / "CARD.md"
            self.assertTrue(card.is_file())
            text = card.read_text(encoding="utf-8")
            self.assertIn("not_published", text)
            self.assertIn("public_ranking_card_v1", text)
            self.assertIn("Fixture baselines", text)
            self.assertIn("out of scope", text)
            for name in ("baselines-length", "baselines-events"):
                agg = json.loads(
                    (Path(tmp) / name / "aggregate.json").read_text(encoding="utf-8")
                )
                self.assertEqual(agg["claim_status"], "not_published")
                self.assertEqual(agg["card_id"], "public_ranking_card_v1")
                self.assertEqual(agg["n_seeds"], 3)


if __name__ == "__main__":
    unittest.main()
