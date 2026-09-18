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
        self.assertGreaterEqual(len(kept), 30)
        self.assertIn(0, y_true)
        self.assertIn(1, y_true)
        labels = {s.label for s in kept}
        self.assertIn("normal", labels)
        self.assertTrue(labels & {"incident", "cascade"})

    def test_frozen_split_both_classes_in_eval(self):
        from eval.fixture_train import load_split, partition_by_split
        from eval.labels import filter_scorable, load_lab_sessions

        split = load_split(FIXTURE)
        self.assertGreaterEqual(split["n_eval"], 24)
        self.assertGreaterEqual(split["n_train"], 20)
        sessions, _ = load_lab_sessions(FIXTURE)
        _, kept = filter_scorable(sessions)
        train_s, eval_s = partition_by_split(kept, split)
        self.assertEqual(len(train_s), split["n_train"])
        self.assertEqual(len(eval_s), split["n_eval"])
        eval_bins = {int(s.binary) for s in eval_s}  # type: ignore[arg-type]
        self.assertEqual(eval_bins, {0, 1})
        self.assertFalse({s.session_id for s in train_s} & {s.session_id for s in eval_s})

    def test_length_baseline_eval_split_smoke(self):
        from eval.run_eval import main

        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "--capture",
                    str(FIXTURE),
                    "--scores-from",
                    "length",
                    "--session-split",
                    "eval",
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
            self.assertIn("auroc", report["metrics"])
            self.assertGreaterEqual(report["metrics"]["n"], 24)
            self.assertGreaterEqual(report["metrics"]["n_positive"], 2)
            self.assertGreaterEqual(report["metrics"]["n_negative"], 2)
            self.assertEqual(report["corpus"]["split"]["split_role"], "eval")
            # Hygiene: no machine-absolute capture_dir in committed-style reports
            self.assertFalse(str(report["corpus"]["capture_dir"]).startswith("/workspace"))
            self.assertFalse(str(report["corpus"]["capture_dir"]).startswith("/Users/"))


class TestPublicRankingCardReproduce(unittest.TestCase):
    def test_run_baselines_writes_card(self):
        from eval.run_public_ranking_card import main

        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "--baselines-only",
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
            self.assertIn("public_ranking_card_v1", text)
            self.assertIn("Limitations", text)
            self.assertIn("synthetic", text.lower())
            self.assertIn("lab-pool", text.lower())
            self.assertIn("production", text.lower())
            for name in ("baselines-length", "baselines-events"):
                agg = json.loads(
                    (Path(tmp) / name / "aggregate.json").read_text(encoding="utf-8")
                )
                self.assertEqual(agg["card_id"], "public_ranking_card_v1")
                self.assertEqual(agg["session_split"], "eval")
                self.assertEqual(agg["n_seeds"], 3)
                self.assertEqual(agg["claim_status"], "not_published")
                for row in agg.get("per_seed") or []:
                    self.assertFalse(str(row.get("path", "")).startswith("/workspace"))


if __name__ == "__main__":
    unittest.main()
