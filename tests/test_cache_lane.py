"""Unit tests: cache_lane quarantine / restore (temp HOME; no MPS / Zenodo).

prepare.py untouched. No invented AUROC / val_bpb.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from corpus.ingest.cache_lane import (  # noqa: E402
    EXIT_OK,
    EXIT_REFUSED_FLAG,
    EXIT_STATE,
    REFUSED_METRIC_FLAGS,
    main,
    quarantine,
    restore,
    status,
)


def _touch_tree(path: Path, marker: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "marker.txt").write_text(marker, encoding="utf-8")


class TestCacheLane(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.home = Path(self._td.name)
        self.cache = self.home / ".cache" / "autoresearch"
        self.cache.mkdir(parents=True)
        _touch_tree(self.cache / "data", "active-data")
        _touch_tree(self.cache / "tokenizer", "active-tok")

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_quarantine_and_status(self) -> None:
        result = quarantine(
            cache_root=self.cache, lane="tale", stamp="20260920_010203"
        )
        self.assertEqual(result.data_dst.name, "data_tale_20260920_010203")
        self.assertFalse((self.cache / "data").exists())
        self.assertTrue(result.data_dst.is_dir())
        self.assertEqual(
            (result.data_dst / "marker.txt").read_text(encoding="utf-8"),
            "active-data",
        )
        rep = status(self.cache)
        self.assertFalse(rep.active_data)
        self.assertFalse(rep.active_tokenizer)
        self.assertEqual(
            rep.quarantines,
            (("data_tale_20260920_010203", "tokenizer_tale_20260920_010203"),),
        )

    def test_restore_lane_when_active_absent(self) -> None:
        quarantine(
            cache_root=self.cache, lane="crisp", stamp="20260920_010203"
        )
        restore(cache_root=self.cache, lane="crisp")
        self.assertTrue((self.cache / "data").is_dir())
        self.assertEqual(
            (self.cache / "data" / "marker.txt").read_text(encoding="utf-8"),
            "active-data",
        )
        self.assertFalse((self.cache / "data_crisp_20260920_010203").exists())

    def test_restore_refuses_clobber_without_force(self) -> None:
        quarantine(
            cache_root=self.cache, lane="tale", stamp="20260920_010203"
        )
        _touch_tree(self.cache / "data", "new-active")
        _touch_tree(self.cache / "tokenizer", "new-tok")
        with self.assertRaises(FileExistsError):
            restore(cache_root=self.cache, lane="tale", force=False)

    def test_restore_force_overwrites(self) -> None:
        quarantine(
            cache_root=self.cache, lane="tale", stamp="20260920_010203"
        )
        _touch_tree(self.cache / "data", "new-active")
        _touch_tree(self.cache / "tokenizer", "new-tok")
        restore(cache_root=self.cache, lane="tale", force=True)
        self.assertEqual(
            (self.cache / "data" / "marker.txt").read_text(encoding="utf-8"),
            "active-data",
        )

    def test_restore_from_dir(self) -> None:
        quarantine(
            cache_root=self.cache, lane="tale", stamp="20260920_111111"
        )
        src = self.cache / "data_tale_20260920_111111"
        restore(cache_root=self.cache, from_dir=src)
        self.assertTrue((self.cache / "data").is_dir())

    def test_restore_lane_picks_latest(self) -> None:
        quarantine(
            cache_root=self.cache, lane="tale", stamp="20260919_000000"
        )
        _touch_tree(self.cache / "data", "second")
        _touch_tree(self.cache / "tokenizer", "second-tok")
        quarantine(
            cache_root=self.cache, lane="tale", stamp="20260920_999999"
        )
        restore(cache_root=self.cache, lane="tale")
        self.assertEqual(
            (self.cache / "data" / "marker.txt").read_text(encoding="utf-8"),
            "second",
        )
        # older quarantine still present
        self.assertTrue((self.cache / "data_tale_20260919_000000").is_dir())

    def test_cli_status_and_refuse(self) -> None:
        code = main(["--home", str(self.home), "status"])
        self.assertEqual(code, EXIT_OK)
        for flag in ("--auroc", "--publish", "--cuda", "--val-bpb"):
            self.assertIn(flag, REFUSED_METRIC_FLAGS)
            self.assertEqual(
                main([flag, "--home", str(self.home), "status"]),
                EXIT_REFUSED_FLAG,
            )

    def test_cli_quarantine_restore(self) -> None:
        self.assertEqual(
            main(
                [
                    "--home",
                    str(self.home),
                    "quarantine",
                    "--lane",
                    "crisp",
                    "--stamp",
                    "20260920_010203",
                ]
            ),
            EXIT_OK,
        )
        self.assertEqual(
            main(["--home", str(self.home), "restore", "--lane", "crisp"]),
            EXIT_OK,
        )

    def test_cli_restore_without_force_exits_state(self) -> None:
        main(
            [
                "--home",
                str(self.home),
                "quarantine",
                "--lane",
                "tale",
                "--stamp",
                "20260920_010203",
            ]
        )
        _touch_tree(self.cache / "data", "x")
        _touch_tree(self.cache / "tokenizer", "y")
        self.assertEqual(
            main(["--home", str(self.home), "restore", "--lane", "tale"]),
            EXIT_STATE,
        )

    def test_shell_wrapper_refuse(self) -> None:
        script = ROOT / "scripts" / "cache_lane.sh"
        proc = subprocess.run(
            ["bash", str(script), "--auroc", "status"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHON": sys.executable},
            check=False,
        )
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("Refusing", proc.stderr)


if __name__ == "__main__":
    unittest.main()
