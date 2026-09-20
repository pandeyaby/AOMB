"""Tests for public ranking card v1 — honest fixture runner.

Covers committed fixtures, baseline/session-score emission, flag refusals,
path validation, lab_public_pack stays not_published. No Zenodo / no MPS /
no invented AUROC heroes.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "corpus" / "fixtures" / "public_ranking_card_v1"
LAB_PACK = ROOT / "corpus" / "fixtures" / "lab_public_pack_v0"
SHELL = ROOT / "scripts" / "run_public_ranking_card_v1.sh"


class TestPublicRankingCardFixture(unittest.TestCase):
    def test_fixture_has_normal_and_positives(self):
        from eval.labels import filter_scorable, load_lab_sessions

        sessions, meta = load_lab_sessions(FIXTURE)
        self.assertEqual(meta["capture_id"], "public_ranking_card_v1")
        self.assertTrue(meta["content_sha256"])
        y_true, kept = filter_scorable(sessions)
        self.assertGreaterEqual(len(kept), 60)
        self.assertIn(0, y_true)
        self.assertIn(1, y_true)
        labels = {s.label for s in kept}
        self.assertIn("normal", labels)
        self.assertTrue(labels & {"incident", "cascade"})

    def test_frozen_split_both_classes_in_eval(self):
        from eval.fixture_train import load_split, partition_by_split
        from eval.labels import filter_scorable, load_lab_sessions

        split = load_split(FIXTURE)
        self.assertGreaterEqual(split["n_eval"], 30)
        self.assertGreaterEqual(split["n_train"], 30)
        sessions, _ = load_lab_sessions(FIXTURE)
        _, kept = filter_scorable(sessions)
        train_s, eval_s = partition_by_split(kept, split)
        self.assertEqual(len(train_s), split["n_train"])
        self.assertEqual(len(eval_s), split["n_eval"])
        eval_bins = {int(s.binary) for s in eval_s}  # type: ignore[arg-type]
        self.assertEqual(eval_bins, {0, 1})
        # Balanced labels on eval (equal pos/neg for this card)
        n_pos = sum(1 for s in eval_s if s.binary == 1)
        n_neg = sum(1 for s in eval_s if s.binary == 0)
        self.assertEqual(n_pos, n_neg)
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
            self.assertGreaterEqual(report["metrics"]["n"], 30)
            self.assertGreaterEqual(report["metrics"]["n_positive"], 10)
            self.assertGreaterEqual(report["metrics"]["n_negative"], 10)
            self.assertEqual(report["corpus"]["split"]["split_role"], "eval")
            # Hygiene: no machine-absolute capture_dir in committed-style reports
            self.assertFalse(str(report["corpus"]["capture_dir"]).startswith("/workspace"))
            self.assertFalse(str(report["corpus"]["capture_dir"]).startswith("/Users/"))


class TestFlagRefusals(unittest.TestCase):
    def test_refuse_auroc_before_argparse(self):
        from eval.run_public_ranking_card import main

        with self.assertRaises(SystemExit) as ctx:
            main(["--auroc", "--baselines-only"])
        msg = str(ctx.exception)
        self.assertIn("auroc", msg.lower())
        self.assertIn("not_published", msg.lower())

    def test_refuse_invent_publish_readme_hero(self):
        from eval.run_public_ranking_card import main

        for flag in (
            "--publish",
            "--claim",
            "--invent-auroc",
            "--invent-metrics",
            "--readme-hero",
            "--publish-readme",
            "--hero-auroc",
            "--lab-auroc",
            "--val-bpb",
            "--invent-val-bpb",
            "--accuracy",
            "--claim-auroc",
        ):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit) as ctx:
                    main([flag, "--baselines-only"])
                self.assertIn(flag, str(ctx.exception))

    def test_no_auroc_invent_calculator(self):
        src = (ROOT / "eval" / "run_public_ranking_card.py").read_text(encoding="utf-8")
        self.assertNotIn("def compute_auroc", src)
        self.assertIn("_REFUSED_METRIC_FLAGS", src)
        self.assertIn("readme-hero", src)
        self.assertIn("not_published", src)


class TestFixturePathValidation(unittest.TestCase):
    def test_missing_fixture_exit_2(self):
        from eval.run_public_ranking_card import main

        rc = main(["--fixture", "/no/such/aomb/fixture", "--session-scores-only"])
        self.assertEqual(rc, 2)

    def test_unknown_fixture_name_exit_2(self):
        from eval.run_public_ranking_card import main

        rc = main(["--fixture", "not_a_real_fixture", "--session-scores-only"])
        self.assertEqual(rc, 2)

    def test_validate_fixture_ok(self):
        from eval.run_public_ranking_card import validate_fixture

        self.assertIsNone(validate_fixture(FIXTURE))
        self.assertIsNone(validate_fixture(LAB_PACK))

    def test_validate_fixture_missing_provenance(self):
        from eval.run_public_ranking_card import validate_fixture

        with tempfile.TemporaryDirectory() as tmp:
            err = validate_fixture(Path(tmp))
            self.assertIsNotNone(err)
            assert err is not None
            self.assertIn("provenance.json", err)


class TestSessionBaselineScores(unittest.TestCase):
    def test_session_scores_only_public_card(self):
        from eval.run_public_ranking_card import main

        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "--fixture",
                    "public_ranking_card_v1",
                    "--session-scores-only",
                    "--out-dir",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)
            path = Path(tmp) / "session_baseline_scores.json"
            self.assertTrue(path.is_file())
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["claim_status"], "not_published")
            self.assertGreaterEqual(payload["n_sessions"], 30)
            self.assertEqual(payload["session_split"]["split_role"], "eval")
            row = payload["sessions"][0]
            self.assertIn("score_length", row)
            self.assertIn("score_events", row)
            self.assertIn("session_id", row)
            # No invented metric keys on the session sidecar
            self.assertNotIn("auroc", payload)
            self.assertNotIn("metrics", payload)
            self.assertNotIn("val_bpb", payload)
            self.assertEqual(payload["claim_status"], "not_published")
            blob = json.dumps({k: v for k, v in payload.items() if k != "disclaimer"})
            self.assertNotIn("auroc", blob.lower())
            self.assertNotIn("val_bpb", blob.lower())
            for s in payload["sessions"]:
                self.assertNotIn("auroc", s)
                self.assertNotIn("bpb", s)  # baselines only — no invented session BPB
                self.assertIn("score_length", s)
                self.assertIn("score_events", s)

    def test_session_scores_lab_public_pack(self):
        from eval.run_public_ranking_card import main

        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "--fixture",
                    "lab_public_pack_v0",
                    "--session-scores-only",
                    "--out-dir",
                    tmp,
                ]
            )
            self.assertEqual(rc, 0)
            payload = json.loads(
                (Path(tmp) / "session_baseline_scores.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["claim_status"], "not_published")
            self.assertGreaterEqual(payload["n_sessions"], 40)
            self.assertEqual(payload["session_split"]["split_role"], "all")


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
            self.assertIn("README", text)
            self.assertIn("lab-pool", text.lower())
            self.assertIn("production", text.lower())
            scores = Path(tmp) / "session_baseline_scores.json"
            self.assertTrue(scores.is_file())
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

    def test_lab_public_pack_baselines_not_published(self):
        from eval.run_public_ranking_card import main

        with tempfile.TemporaryDirectory() as tmp:
            rc = main(
                [
                    "--fixture",
                    "lab_public_pack_v0",
                    "--baselines-only",
                    "--out-dir",
                    tmp,
                    "--seeds",
                    "0..1",
                    "--random-draws",
                    "8",
                ]
            )
            self.assertEqual(rc, 0)
            for name in ("baselines-length", "baselines-events"):
                agg = json.loads(
                    (Path(tmp) / name / "aggregate.json").read_text(encoding="utf-8")
                )
                self.assertEqual(agg["claim_status"], "not_published")
                self.assertEqual(agg["fixture_name"], "lab_public_pack_v0")
                self.assertEqual(agg["session_split"], "all")
            # Must not write frozen-card REFERENCE heroes for lab pack
            self.assertFalse((Path(tmp) / "REFERENCE_baselines-length.json").is_file())


class TestShellWrapper(unittest.TestCase):
    def test_shell_refuses_auroc(self):
        if not SHELL.is_file():
            self.skipTest("run_public_ranking_card_v1.sh missing")
        proc = subprocess.run(
            ["bash", str(SHELL), "--auroc", "--baselines-only"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("auroc", (proc.stderr + proc.stdout).lower())

    def test_shell_refuses_readme_hero(self):
        if not SHELL.is_file():
            self.skipTest("run_public_ranking_card_v1.sh missing")
        proc = subprocess.run(
            ["bash", str(SHELL), "--readme-hero"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("readme-hero", (proc.stderr + proc.stdout).lower())

    def test_help_mentions_refusals(self):
        from eval.run_public_ranking_card import build_parser

        help_text = build_parser().format_help()
        self.assertIn("not_published", help_text)
        self.assertIn("auroc", help_text.lower())
        self.assertIn("lab_public_pack_v0", help_text)


if __name__ == "__main__":
    unittest.main()
