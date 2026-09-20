"""
Runtime hardening tests for the session scorer CLI + BYO wrapper.

Covers missing dump / missing checkpoint, flag refusals, fixture dry-run,
BYO shell edge cases (shared exit codes with score_cli), and a tiny synthetic
checkpoint happy path (mocked model score — no torch, no Zenodo, no MPS,
no invented AUROC).
"""

from __future__ import annotations

import io
import re
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
LAB_SAMPLE = ROOT / "corpus" / "fixtures" / "lab_sample"
CRISP_SAMPLE = ROOT / "corpus" / "fixtures" / "crisp_sample"
BYO_SCRIPT = ROOT / "scripts" / "byo_score.sh"


class TestValidateRuntimePaths(unittest.TestCase):
    def test_missing_dump(self):
        from eval.score_cli import validate_runtime_paths

        err = validate_runtime_paths(
            input_path="/no/such/aomb/dump",
            dry_run=True,
        )
        self.assertIsNotNone(err)
        assert err is not None
        self.assertIn("session dump not found", err)
        self.assertIn("/no/such/aomb/dump", err)

    def test_missing_checkpoint(self):
        from eval.score_cli import validate_runtime_paths

        err = validate_runtime_paths(
            input_path=str(LAB_SAMPLE),
            checkpoint="/no/such/scorer.pt",
            dry_run=False,
        )
        self.assertIsNotNone(err)
        assert err is not None
        self.assertIn("checkpoint not found", err)
        self.assertIn("/no/such/scorer.pt", err)

    def test_requires_mode_when_not_dry_run(self):
        from eval.score_cli import validate_runtime_paths

        err = validate_runtime_paths(
            input_path=str(LAB_SAMPLE),
            dry_run=False,
            train_seconds=0.0,
            checkpoint=None,
        )
        self.assertIsNotNone(err)
        assert err is not None
        self.assertIn("--checkpoint", err)
        self.assertIn("--train-seconds", err)
        self.assertIn("--dry-run", err)

    def test_ok_dry_run_and_checkpoint_file(self):
        from eval.score_cli import validate_runtime_paths

        self.assertIsNone(
            validate_runtime_paths(input_path=str(LAB_SAMPLE), dry_run=True)
        )
        with tempfile.NamedTemporaryFile(suffix=".pt") as fh:
            self.assertIsNone(
                validate_runtime_paths(
                    input_path=str(LAB_SAMPLE),
                    checkpoint=fh.name,
                    dry_run=False,
                )
            )


class TestFlagRefusals(unittest.TestCase):
    def test_refuse_auroc_before_argparse(self):
        from eval.score_cli import EXIT_REFUSED_FLAG, main

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            with self.assertRaises(SystemExit) as ctx:
                main(["--auroc", "--input", str(LAB_SAMPLE), "--dry-run"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)
        msg = buf.getvalue().lower()
        self.assertIn("auroc", msg)
        self.assertIn("not_published", msg)

    def test_refuse_publish_and_ranking(self):
        from eval.score_cli import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS, main

        for flag in sorted(REFUSED_METRIC_FLAGS):
            with self.subTest(flag=flag):
                buf = io.StringIO()
                with mock.patch("sys.stderr", buf):
                    with self.assertRaises(SystemExit) as ctx:
                        main([flag, "--input", str(LAB_SAMPLE), "--dry-run"])
                self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)
                self.assertIn(flag, buf.getvalue())

    def test_refuse_equals_form_on_dry_run_and_real_score_paths(self):
        """Refusals apply equally on --dry-run and --checkpoint / --train-seconds."""
        from eval.score_cli import EXIT_REFUSED_FLAG, main

        cases = (
            ["--input", str(LAB_SAMPLE), "--dry-run", "--publish=1"],
            [
                "--input",
                str(LAB_SAMPLE),
                "--checkpoint",
                "/tmp/aomb-unused.pt",
                "--invent-auroc",
            ],
            ["--auroc=true", "--input", str(LAB_SAMPLE), "--train-seconds", "1"],
        )
        for argv in cases:
            with self.subTest(argv=argv):
                buf = io.StringIO()
                with mock.patch("sys.stderr", buf):
                    with self.assertRaises(SystemExit) as ctx:
                        main(argv)
                self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)
                self.assertIn("Refusing", buf.getvalue())

    def test_no_auroc_calculator_in_module(self):
        src = (ROOT / "eval" / "score_cli.py").read_text(encoding="utf-8")
        self.assertNotIn("def compute_auroc", src)
        self.assertIn("not_published", src)
        self.assertIn("REFUSED_METRIC_FLAGS", src)
        self.assertIn("EXIT_PATH_ERROR", src)


class TestMissingPathsViaMain(unittest.TestCase):
    def test_main_missing_dump_exit_2(self):
        from eval.score_cli import main

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            rc = main(["--input", "/no/such/dump", "--dry-run"])
        self.assertEqual(rc, 2)
        self.assertIn("session dump not found", buf.getvalue())

    def test_main_missing_checkpoint_exit_2(self):
        from eval.score_cli import main

        buf = io.StringIO()
        with mock.patch("sys.stderr", buf):
            rc = main(
                [
                    "--input",
                    str(LAB_SAMPLE),
                    "--checkpoint",
                    "/tmp/aomb-missing-scorer.pt",
                ]
            )
        self.assertEqual(rc, 2)
        self.assertIn("checkpoint not found", buf.getvalue())


class TestHappyPathFixture(unittest.TestCase):
    def test_dry_run_lab_sample_deterministic(self):
        from eval.score_cli import main

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "score.json")
            rc = main(
                [
                    "--input",
                    str(LAB_SAMPLE),
                    "--dry-run",
                    "--out",
                    out,
                    "--json",
                    "--max-sessions",
                    "2",
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertEqual(report["claim_status"], "not_published")
            self.assertEqual(report["train_meta"]["mode"], "dry_run")
            self.assertEqual(len(report["sessions"]), 2)
            self.assertTrue(all(s["bpb"] is None for s in report["sessions"]))
            # No AUROC / ranking keys invented
            blob = json.dumps(report)
            self.assertNotIn("auroc", blob.lower())
            self.assertNotIn('"ranking"', blob.lower())

    def test_dry_run_crisp_sample(self):
        from eval.score_cli import main

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "score.json")
            rc = main(["--input", str(CRISP_SAMPLE), "--dry-run", "--out", out])
            self.assertEqual(rc, 0)
            report = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertEqual(report["claim_status"], "not_published")
            self.assertEqual(len(report["sessions"]), 2)

    def test_synthetic_checkpoint_happy_path_mocked_bpb(self):
        """Checkpoint file exists; model score mocked → deterministic BPB only."""
        from eval.score_cli import main

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "tiny_scorer.pt"
            ckpt.write_bytes(b"aomb_session_scorer_v1_stub")
            out = os.path.join(tmp, "score.json")

            def _fake_score(rows, **kwargs):
                self.assertEqual(kwargs.get("checkpoint"), str(ckpt))
                scores = [1.25 + 0.1 * i for i in range(len(rows))]
                return scores, {
                    "mode": "checkpoint",
                    "checkpoint": str(ckpt),
                    "device": "cpu",
                    "note": "Session BPB via model forward; prepare.evaluate_bpb untouched.",
                }

            with mock.patch("eval.score_cli.score_with_model", side_effect=_fake_score):
                rc = main(
                    [
                        "--input",
                        str(LAB_SAMPLE),
                        "--checkpoint",
                        str(ckpt),
                        "--out",
                        out,
                        "--json",
                        "--max-sessions",
                        "2",
                    ]
                )
            self.assertEqual(rc, 0)
            report = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertEqual(report["claim_status"], "not_published")
            self.assertEqual(report["train_meta"]["mode"], "checkpoint")
            self.assertEqual(len(report["sessions"]), 2)
            bpbs = [s["bpb"] for s in report["sessions"]]
            self.assertEqual(bpbs, [1.25, 1.35])
            blob = json.dumps(report).lower()
            self.assertNotIn("auroc", blob)
            self.assertNotIn("published_fixture_card", blob)


class TestScoreSessionEntrypoint(unittest.TestCase):
    def test_module_delegates_to_score_cli(self):
        import score_session
        from eval import score_cli

        self.assertIs(score_session.main, score_cli.main)

    def test_help_mentions_refusals(self):
        from eval.score_cli import build_parser

        help_text = build_parser().format_help()
        self.assertIn("not_published", help_text)
        self.assertIn("auroc", help_text.lower())


class TestByoShellWrapper(unittest.TestCase):
    """BYO wrapper edge cases — shared refusals/exit codes with score_cli."""

    def _run_byo(self, *args: str) -> subprocess.CompletedProcess[str]:
        if not BYO_SCRIPT.is_file():
            self.skipTest("byo_score.sh missing")
        return subprocess.run(
            ["bash", str(BYO_SCRIPT), *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHON": sys.executable},
        )

    def test_shell_refuses_auroc_exit_1(self):
        from eval.score_cli import EXIT_REFUSED_FLAG

        proc = self._run_byo(str(LAB_SAMPLE), "--auroc")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertIn("auroc", blob)
        self.assertIn("not_published", blob)

    def test_shell_refuses_all_metric_flags(self):
        from eval.score_cli import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS

        for flag in sorted(REFUSED_METRIC_FLAGS):
            with self.subTest(flag=flag):
                proc = self._run_byo(str(LAB_SAMPLE), "--dry-run", flag)
                self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG, proc.stderr)
                self.assertIn("Refusing", proc.stderr + proc.stdout)

    def test_shell_refuses_equals_form_on_default_dry_run(self):
        from eval.score_cli import EXIT_REFUSED_FLAG

        proc = self._run_byo(str(LAB_SAMPLE), "--publish=yes")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("--publish", proc.stderr + proc.stdout)

    def test_shell_missing_dump_exit_2(self):
        from eval.score_cli import EXIT_PATH_ERROR

        proc = self._run_byo("/no/such/aomb/byo-dump", "--dry-run")
        self.assertEqual(proc.returncode, EXIT_PATH_ERROR)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertIn("dump path not found", blob)
        self.assertIn("/no/such/aomb/byo-dump", blob)

    def test_shell_missing_dump_arg_exit_2(self):
        from eval.score_cli import EXIT_PATH_ERROR

        proc = self._run_byo()
        self.assertEqual(proc.returncode, EXIT_PATH_ERROR)
        self.assertIn("missing dump path", (proc.stderr + proc.stdout).lower())

    def test_shell_flag_as_first_arg_exit_2(self):
        from eval.score_cli import EXIT_PATH_ERROR

        proc = self._run_byo("--dry-run")
        self.assertEqual(proc.returncode, EXIT_PATH_ERROR)
        self.assertIn("dump path", (proc.stderr + proc.stdout).lower())

    def test_shell_dry_run_smoke_lab_sample(self):
        proc = self._run_byo(
            str(LAB_SAMPLE), "--dry-run", "--max-sessions", "1", "--json"
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        blob = (proc.stderr + proc.stdout).lower()
        self.assertIn("not_published", blob)
        # stdout JSON/score lines must not invent AUROC keys (banner stays on stderr)
        for line in proc.stdout.splitlines():
            if line.strip().startswith("{"):
                self.assertNotIn("auroc", line.lower())

    def test_shell_refused_flags_match_score_cli(self):
        from eval.score_cli import REFUSED_METRIC_FLAGS

        src = BYO_SCRIPT.read_text(encoding="utf-8")
        m = re.search(
            r'case "\$key" in\n\s+(--[^\n]+)\n\s+die_refuse',
            src,
        )
        self.assertIsNotNone(m, "byo_score.sh refused-flag case arm missing")
        assert m is not None
        listed = {
            f.rstrip(")")
            for f in m.group(1).split("|")
            if f.startswith("--")
        }
        self.assertEqual(listed, set(REFUSED_METRIC_FLAGS))

    def test_shell_exit_constants_match_score_cli(self):
        from eval.score_cli import EXIT_PATH_ERROR, EXIT_REFUSED_FLAG

        src = BYO_SCRIPT.read_text(encoding="utf-8")
        self.assertIn(f"EXIT_REFUSED_FLAG={EXIT_REFUSED_FLAG}", src)
        self.assertIn(f"EXIT_PATH_ERROR={EXIT_PATH_ERROR}", src)


if __name__ == "__main__":
    unittest.main()
