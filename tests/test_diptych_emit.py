"""DIPTYCH adapter emit CLI — deterministic probe JSON + flag refusals.

Covers fixture emit, dry-run, stub-pass refusal, AUROC invent refusal,
shell wrapper sync. No invented AUROC / val_bpb. CUDA gate stays skipped.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHELL = ROOT / "scripts" / "run_diptych_full8.sh"
PROBES = ROOT / "diptych-probes"


class TestDiptychEmit(unittest.TestCase):
    def test_emit_all_operators_deterministic(self):
        from eval.diptych import OPERATORS
        from eval.diptych.emit import emit_probe_json, emit_probes, load_and_validate_probe

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "emit"
            manifest = emit_probes(out_dir=out, dry_run=False)
            self.assertEqual(manifest["n_probes"], 16)
            self.assertEqual(manifest["diptych_schema"], "0.2")
            self.assertEqual(manifest["source"], "aomb")
            self.assertIn("Never invents AUROC", manifest["honesty"])
            self.assertTrue((out / "emit_manifest.json").is_file())

            for op in OPERATORS:
                for role in ("conforming", "violating"):
                    dest = out / op / role / "probe.json"
                    self.assertTrue(dest.is_file(), dest)
                    doc = json.loads(dest.read_text(encoding="utf-8"))
                    src = load_and_validate_probe(PROBES / op / role / "probe.json")
                    # Deterministic sort_keys rewrite matches emit_probe_json
                    self.assertEqual(dest.read_text(encoding="utf-8"), emit_probe_json(src))
                    self.assertEqual(doc["operator"], op)
                    self.assertEqual(doc["control_role"], role)
                    self.assertNotIn("auroc", json.dumps(doc).lower())

            # Second emit is byte-identical (deterministic regenerate)
            out2 = Path(tmp) / "emit2"
            emit_probes(out_dir=out2, dry_run=False)
            for op in OPERATORS:
                for role in ("conforming", "violating"):
                    a = (out / op / role / "probe.json").read_bytes()
                    b = (out2 / op / role / "probe.json").read_bytes()
                    self.assertEqual(a, b, f"{op}/{role}")

    def test_emit_single_operator(self):
        from eval.diptych.emit import emit_probes

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = emit_probes(
                out_dir=out, operators=("SIGNFLIP",), dry_run=False
            )
            self.assertEqual(manifest["n_probes"], 2)
            self.assertTrue((out / "SIGNFLIP" / "conforming" / "probe.json").is_file())
            self.assertFalse((out / "RESEED").exists())

    def test_dry_run_writes_nothing(self):
        from eval.diptych.emit import emit_probes, main

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "should_stay_empty"
            manifest = emit_probes(out_dir=out, dry_run=True)
            self.assertEqual(manifest["mode"], "dry-run")
            self.assertEqual(manifest["n_probes"], 16)
            self.assertFalse(out.exists())
            rc = main(["--dry-run", "--json"])
            self.assertEqual(rc, 0)

    def test_main_requires_out_unless_dry_run(self):
        from eval.diptych.emit import EXIT_PATH_ERROR, main

        rc = main([])
        self.assertEqual(rc, EXIT_PATH_ERROR)

    def test_main_emit_ok(self):
        from eval.diptych.emit import main

        with tempfile.TemporaryDirectory() as tmp:
            rc = main(["--out", tmp, "--operator", "RESEED"])
            self.assertEqual(rc, 0)
            self.assertTrue(
                (Path(tmp) / "RESEED" / "violating" / "probe.json").is_file()
            )

    def test_unknown_operator_exit_2(self):
        from eval.diptych.emit import EXIT_PATH_ERROR, main

        with tempfile.TemporaryDirectory() as tmp:
            rc = main(["--out", tmp, "--operator", "NOTREAL"])
            self.assertEqual(rc, EXIT_PATH_ERROR)

    def test_missing_probes_root_exit_2(self):
        from eval.diptych.emit import EXIT_PATH_ERROR, main

        with tempfile.TemporaryDirectory() as tmp:
            rc = main(["--out", tmp, "--probes-root", str(Path(tmp) / "nope")])
            self.assertEqual(rc, EXIT_PATH_ERROR)

    def test_refuse_stub_pass_content(self):
        from eval.diptych.contract import ContractError
        from eval.diptych.emit import EXIT_EMIT_ERROR, emit_probes, main

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            probes = tmp_path / "probes"
            # Copy one operator pair then corrupt violating twin into stub-pass
            import shutil

            shutil.copytree(PROBES / "SCHEMAX", probes / "SCHEMAX")
            viol = probes / "SCHEMAX" / "violating" / "probe.json"
            doc = json.loads(viol.read_text(encoding="utf-8"))
            doc["expected_verdict"] = "pass"  # stub-pass shape
            viol.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")

            with self.assertRaises(ContractError) as ctx:
                emit_probes(
                    out_dir=tmp_path / "out",
                    probes_root=probes,
                    operators=("SCHEMAX",),
                )
            self.assertIn("stub-pass", str(ctx.exception).lower())

            rc = main(
                [
                    "--out",
                    str(tmp_path / "out2"),
                    "--probes-root",
                    str(probes),
                    "--operator",
                    "SCHEMAX",
                ]
            )
            self.assertEqual(rc, EXIT_EMIT_ERROR)


class TestDiptychEmitFlagRefusals(unittest.TestCase):
    def test_refuse_auroc_before_argparse(self):
        from eval.diptych.emit import EXIT_REFUSED_FLAG, main

        with self.assertRaises(SystemExit) as ctx:
            main(["--auroc", "--dry-run"])
        self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)

    def test_refuse_stub_pass_and_invent_flags(self):
        from eval.diptych.emit import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS, main

        for flag in sorted(REFUSED_METRIC_FLAGS):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit) as ctx:
                    main([flag, "--dry-run"])
                self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)

    def test_refuse_equals_form(self):
        from eval.diptych.emit import EXIT_REFUSED_FLAG, main

        for flag in ("--auroc=true", "--stub-pass=1", "--invent-val-bpb=yes"):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit) as ctx:
                    main([flag, "--dry-run"])
                self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)

    def test_run_full8_refuses_same_flags(self):
        from eval.diptych.emit import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS
        from eval.diptych.run_full8 import main as gate_main

        for flag in ("--auroc", "--stub-pass", "--val-bpb", "--cuda"):
            with self.subTest(flag=flag):
                with self.assertRaises(SystemExit) as ctx:
                    gate_main([flag])
                self.assertEqual(ctx.exception.code, EXIT_REFUSED_FLAG)
        # Shared constant surface
        from eval.diptych import run_full8 as R

        self.assertEqual(R.REFUSED_METRIC_FLAGS, REFUSED_METRIC_FLAGS)

    def test_run_full8_emit_only_requires_dir(self):
        from eval.diptych.emit import EXIT_PATH_ERROR
        from eval.diptych.run_full8 import main as gate_main

        rc = gate_main(["--emit-only"])
        self.assertEqual(rc, EXIT_PATH_ERROR)

    def test_run_full8_emit_only_ok(self):
        from eval.diptych.run_full8 import main as gate_main

        with tempfile.TemporaryDirectory() as tmp:
            rc = gate_main(["--emit-only", "--emit-dir", tmp])
            self.assertEqual(rc, 0)
            self.assertTrue(
                (Path(tmp) / "FREEZEDRY" / "conforming" / "probe.json").is_file()
            )
            # emit-only must not require rewriting coverage/matrix
            self.assertTrue((Path(tmp) / "emit_manifest.json").is_file())


class TestDiptychEmitShell(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(SHELL), *args],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(ROOT)},
        )

    def test_shell_refuses_auroc(self):
        from eval.diptych.emit import EXIT_REFUSED_FLAG

        proc = self._run("--auroc")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("auroc", proc.stderr.lower())

    def test_shell_refuses_stub_pass(self):
        from eval.diptych.emit import EXIT_REFUSED_FLAG

        proc = self._run("--stub-pass")
        self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG)
        self.assertIn("stub", proc.stderr.lower())

    def test_shell_refuses_val_bpb_and_cuda(self):
        from eval.diptych.emit import EXIT_REFUSED_FLAG

        for flag in ("--val-bpb", "--cuda", "--hardcoded-pass", "--fake-green"):
            with self.subTest(flag=flag):
                proc = self._run(flag)
                self.assertEqual(proc.returncode, EXIT_REFUSED_FLAG, proc.stderr)

    def test_shell_emit_only_without_dir_exit_2(self):
        from eval.diptych.emit import EXIT_PATH_ERROR

        proc = self._run("--emit-only")
        self.assertEqual(proc.returncode, EXIT_PATH_ERROR)

    def test_shell_emit_only_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._run("--emit-only", "--emit-dir", tmp)
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            self.assertTrue(
                (Path(tmp) / "TRAJSWAP" / "violating" / "probe.json").is_file()
            )

    def test_shell_refused_flags_match_emit_cli(self):
        from eval.diptych.emit import EXIT_REFUSED_FLAG, REFUSED_METRIC_FLAGS

        src = SHELL.read_text(encoding="utf-8")
        m = re.search(
            r'case "\$key" in\n\s+(--[^\n]+)\n\s+die_refuse',
            src,
        )
        self.assertIsNotNone(m, "run_diptych_full8.sh refused-flag case arm missing")
        assert m is not None
        listed = {
            f.rstrip(")")
            for f in m.group(1).split("|")
            if f.startswith("--")
        }
        self.assertEqual(listed, set(REFUSED_METRIC_FLAGS))
        self.assertIn(f"EXIT_REFUSED_FLAG={EXIT_REFUSED_FLAG}", src)

    def test_no_auroc_invent_in_emit_module(self):
        src = (ROOT / "eval" / "diptych" / "emit.py").read_text(encoding="utf-8")
        self.assertNotIn("def compute_auroc", src)
        self.assertIn("REFUSED_METRIC_FLAGS", src)
        self.assertIn("stub-pass", src)
        self.assertIn("val_bpb", src)


if __name__ == "__main__":
    unittest.main()
