"""Tale overnight launch helper — dry-run / refuse (no agent_loop, no APIs).

No MPS train / no paid APIs. prepare.py untouched. Never invents AUROC / floors.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.tale_overnight_launch import (  # noqa: E402
    EXIT_OK,
    EXIT_PLATFORM,
    EXIT_REFUSED_FLAG,
    assess_cache_lane,
    card_ok,
    format_dry_run,
    main,
    planned_env,
    resolve_dry_run_best_val,
)
from best_val_bpb import (  # noqa: E402
    SOURCE_ENV,
    SOURCE_GIT,
    SOURCE_MEASURED_CARD,
    SOURCE_MISSING,
)


def _write_card(path: Path, **fields) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields), encoding="utf-8")
    return path


GOOD = dict(
    claim_status="measured_not_published",
    val_bpb=1.37952,
    max_spans=200000,
)


class TestPlannedEnv(unittest.TestCase):
    def test_defaults(self):
        env = planned_env()
        self.assertEqual(env["AOMB_CORPUS"], "tale_of_errors")
        self.assertEqual(env["AOMB_SOURCE_ID"], "tale_capped_200k")
        self.assertEqual(env["AOMB_BEST_VAL_FROM_CARD"], "1")


class TestCardOk(unittest.TestCase):
    def test_good(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_card(Path(tmp) / "c.json", **GOOD)
            ok, msg = card_ok(path)
        self.assertTrue(ok)
        self.assertIn("1.379520", msg)

    def test_missing(self):
        ok, msg = card_ok(Path("/no/such/card.json"))
        self.assertFalse(ok)
        self.assertIn("missing", msg)


class TestCli(unittest.TestCase):
    def test_dry_run_default_exit_0(self):
        code = main([])
        self.assertEqual(code, EXIT_OK)
        code = main(["--dry-run"])
        self.assertEqual(code, EXIT_OK)

    def test_dry_run_with_missing_card_still_0(self):
        code = main(["--dry-run", "--card", "/no/such/measured.json"])
        self.assertEqual(code, EXIT_OK)

    def test_invent_refuse(self):
        for flag in ("--auroc", "--publish", "--cuda"):
            self.assertEqual(main([flag]), EXIT_REFUSED_FLAG)
            self.assertEqual(main(["--dry-run", flag]), EXIT_REFUSED_FLAG)

    def test_run_missing_card_exit_2(self):
        code = main(["--run", "--card", "/no/such/measured.json"])
        self.assertEqual(code, EXIT_PLATFORM)

    def test_run_starts_agent_when_platform_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_card(Path(tmp) / "c.json", **GOOD)
            cache = Path(tmp) / "cache"
            cache.mkdir()
            (cache / "data").mkdir()
            (cache / "tokenizer").mkdir()
            with mock.patch(
                "eval.tale_overnight_launch.darwin_mps_ok",
                return_value=(True, "Darwin + MPS available"),
            ):
                with mock.patch(
                    "eval.tale_overnight_launch.start_agent_loop",
                    return_value=0,
                ) as start:
                    code = main(
                        ["--run", "--card", str(path), "--cache-root", str(cache)]
                    )
        self.assertEqual(code, EXIT_OK)
        start.assert_called_once_with()

    def test_run_refused_when_not_darwin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_card(Path(tmp) / "c.json", **GOOD)
            cache = Path(tmp) / "cache"
            cache.mkdir()
            (cache / "data").mkdir()
            (cache / "tokenizer").mkdir()
            with mock.patch(
                "eval.tale_overnight_launch.darwin_mps_ok",
                return_value=(False, "not Darwin (detected: Linux)"),
            ):
                with mock.patch("eval.tale_overnight_launch.start_agent_loop") as start:
                    code = main(
                        ["--run", "--card", str(path), "--cache-root", str(cache)]
                    )
        self.assertEqual(code, EXIT_PLATFORM)
        start.assert_not_called()

    def test_shell_dry_run(self):
        script = ROOT / "scripts" / "tale_overnight_launch.sh"
        proc = subprocess.run(
            ["bash", str(script)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHON": sys.executable, "PYTHONPATH": str(ROOT)},
            check=False,
        )
        self.assertEqual(proc.returncode, EXIT_OK)
        self.assertIn("AOMB_CORPUS=tale_of_errors", proc.stdout)
        self.assertIn("AOMB_BEST_VAL_FROM_CARD=1", proc.stdout)
        self.assertIn("dry-run", proc.stdout)
        self.assertNotIn("starting agent_loop", proc.stdout)

    def test_shell_invent_refuse(self):
        script = ROOT / "scripts" / "tale_overnight_launch.sh"
        proc = subprocess.run(
            ["bash", str(script), "--auroc"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHON": sys.executable},
            check=False,
        )
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)



class TestCacheLaneGate(unittest.TestCase):
    """cache_lane status gate: dry-run warns; --run refuses without active lane."""

    def _home_cache(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        home = Path(td.name)
        cache = home / ".cache" / "autoresearch"
        cache.mkdir(parents=True)
        return home, cache

    def _good_card(self) -> Path:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return _write_card(Path(td.name) / "measured.json", **GOOD)

    def test_assess_ready_when_active(self):
        home, cache = self._home_cache()
        (cache / "data").mkdir()
        (cache / "tokenizer").mkdir()
        ready, summary, lines = assess_cache_lane(cache)
        self.assertTrue(ready)
        self.assertIn("ready", summary.lower())
        blob = "\n".join(lines).lower()
        self.assertIn("active: data=yes", blob)
        self.assertIn("tokenizer=yes", blob)

    def test_assess_missing_and_partial(self):
        home, cache = self._home_cache()
        ready, summary, lines = assess_cache_lane(cache)
        self.assertFalse(ready)
        self.assertIn("no active data", summary.lower())
        (cache / "data").mkdir()
        ready2, summary2, _ = assess_cache_lane(cache)
        self.assertFalse(ready2)
        self.assertIn("partial", summary2.lower())

    def test_dry_run_prints_status_without_active(self):
        home, cache = self._home_cache()
        card = self._good_card()
        # quarantine-only fixture
        q = cache / "data_tale_20260920_010203"
        q.mkdir()
        (cache / "tokenizer_tale_20260920_010203").mkdir()
        buf = __import__("io").StringIO()
        with mock.patch("sys.stdout", buf):
            code = main(
                ["--dry-run", "--card", str(card), "--home", str(home)]
            )
        self.assertEqual(code, EXIT_OK)
        out = buf.getvalue().lower()
        self.assertIn("cache_lane:", out)
        self.assertIn("quarantined:", out)
        self.assertIn("data_tale_20260920_010203", out)
        self.assertIn("warning", out)
        self.assertIn("no agent_loop", out)
        self.assertNotIn("starting agent_loop", out)

    def test_dry_run_active_lane_ok(self):
        home, cache = self._home_cache()
        (cache / "data").mkdir()
        (cache / "tokenizer").mkdir()
        card = self._good_card()
        buf = __import__("io").StringIO()
        with mock.patch("sys.stdout", buf):
            code = main(
                ["--dry-run", "--card", str(card), "--cache-root", str(cache)]
            )
        self.assertEqual(code, EXIT_OK)
        out = buf.getvalue().lower()
        self.assertIn("active: data=yes", out)
        self.assertIn("ready for --run", out)  # verdict wording

    def test_run_refuses_missing_lane_exit_2(self):
        home, cache = self._home_cache()
        card = self._good_card()
        with mock.patch("eval.tale_overnight_launch.start_agent_loop") as start:
            with mock.patch(
                "eval.tale_overnight_launch.darwin_mps_ok",
                return_value=(True, "Darwin + MPS available"),
            ):
                err = __import__("io").StringIO()
                with mock.patch("sys.stderr", err):
                    code = main(
                        ["--run", "--card", str(card), "--home", str(home)]
                    )
        self.assertEqual(code, EXIT_PLATFORM)
        self.assertIn("cache_lane", err.getvalue().lower())
        self.assertIn("not ready", err.getvalue().lower())
        start.assert_not_called()

    def test_run_proceeds_when_lane_ready(self):
        home, cache = self._home_cache()
        (cache / "data").mkdir()
        (cache / "tokenizer").mkdir()
        card = self._good_card()
        with mock.patch(
            "eval.tale_overnight_launch.start_agent_loop", return_value=0
        ) as start:
            with mock.patch(
                "eval.tale_overnight_launch.darwin_mps_ok",
                return_value=(True, "Darwin + MPS available"),
            ):
                code = main(
                    ["--run", "--card", str(card), "--cache-root", str(cache)]
                )
        self.assertEqual(code, EXIT_OK)
        start.assert_called_once_with()

    def test_invent_still_refused(self):
        home, _ = self._home_cache()
        for flag in ("--auroc", "--publish", "--cuda"):
            with self.subTest(flag=flag):
                with mock.patch("eval.tale_overnight_launch.start_agent_loop") as start:
                    self.assertEqual(
                        main(["--dry-run", flag, "--home", str(home)]),
                        EXIT_REFUSED_FLAG,
                    )
                    start.assert_not_called()


class TestDryRunBestValSource(unittest.TestCase):
    """Dry-run prints resolved best_val + source tag (never invents)."""

    def _home_cache(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        home = Path(td.name)
        cache = home / ".cache" / "autoresearch"
        cache.mkdir(parents=True)
        return home, cache

    def _good_card(self) -> Path:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return _write_card(Path(td.name) / "measured.json", **GOOD)

    def test_helper_source_env(self):
        home, cache = self._home_cache()
        card = self._good_card()
        env = planned_env(card_path=card)
        env["AOMB_BEST_VAL_BPB"] = "1.2000"
        value, source = resolve_dry_run_best_val(
            env, card_path=card, cache_root=cache, subjects=[]
        )
        self.assertAlmostEqual(value, 1.2000, places=4)
        self.assertEqual(source, SOURCE_ENV)

    def test_helper_source_measured_card(self):
        home, cache = self._home_cache()
        (cache / "data").mkdir()
        (cache / "tokenizer").mkdir()
        card = self._good_card()
        env = planned_env(card_path=card)
        value, source = resolve_dry_run_best_val(
            env, card_path=card, cache_root=cache, subjects=[]
        )
        self.assertAlmostEqual(value, 1.37952, places=5)
        self.assertEqual(source, SOURCE_MEASURED_CARD)

    def test_helper_source_git(self):
        home, cache = self._home_cache()
        card = Path("/no/such/measured.json")
        env = planned_env(card_path=card)
        subjects = [
            "[val_bpb=0.5120] [corpus=tale] [source_id=uber-tale-of-errors]"
        ]
        value, source = resolve_dry_run_best_val(
            env, card_path=card, cache_root=cache, subjects=subjects
        )
        self.assertAlmostEqual(value, 0.5120, places=4)
        self.assertEqual(source, SOURCE_GIT)

    def test_helper_source_missing(self):
        home, cache = self._home_cache()
        card = Path("/no/such/measured.json")
        env = planned_env(card_path=card)
        value, source = resolve_dry_run_best_val(
            env, card_path=card, cache_root=cache, subjects=[]
        )
        self.assertTrue(__import__("math").isinf(value))
        self.assertEqual(source, SOURCE_MISSING)

    def test_card_inactive_lane_not_measured_card(self):
        """PR #78: measured_card only when cache_lane active."""
        home, cache = self._home_cache()
        card = self._good_card()
        env = planned_env(card_path=card)
        value, source = resolve_dry_run_best_val(
            env, card_path=card, cache_root=cache, subjects=[]
        )
        self.assertTrue(__import__("math").isinf(value))
        self.assertEqual(source, SOURCE_MISSING)

    def test_dry_run_prints_missing_inf(self):
        home, cache = self._home_cache()
        buf = __import__("io").StringIO()
        with mock.patch("sys.stdout", buf):
            with mock.patch(
                "eval.tale_overnight_launch.load_git_commit_subjects",
                return_value=[],
            ):
                code = main(
                    [
                        "--dry-run",
                        "--card",
                        "/no/such/measured.json",
                        "--cache-root",
                        str(cache),
                    ]
                )
        self.assertEqual(code, EXIT_OK)
        out = buf.getvalue()
        self.assertIn("best_val: inf", out)
        self.assertIn("best_val_source: missing", out)
        self.assertNotIn("starting agent_loop", out.lower())

    def test_dry_run_prints_measured_card_when_lane_active(self):
        home, cache = self._home_cache()
        (cache / "data").mkdir()
        (cache / "tokenizer").mkdir()
        card = self._good_card()
        buf = __import__("io").StringIO()
        with mock.patch("sys.stdout", buf):
            with mock.patch(
                "eval.tale_overnight_launch.load_git_commit_subjects",
                return_value=[],
            ):
                code = main(
                    [
                        "--dry-run",
                        "--card",
                        str(card),
                        "--cache-root",
                        str(cache),
                    ]
                )
        self.assertEqual(code, EXIT_OK)
        out = buf.getvalue()
        self.assertIn("best_val: 1.37952", out)
        self.assertIn("best_val_source: measured_card", out)

    def test_format_dry_run_includes_source(self):
        env = planned_env()
        text = format_dry_run(
            env,
            card_path=Path("/x.json"),
            card_status="ok",
            best_val=float("inf"),
            best_val_source=SOURCE_MISSING,
        )
        self.assertIn("best_val: inf", text)
        self.assertIn("best_val_source: missing", text)


if __name__ == "__main__":
    unittest.main()
