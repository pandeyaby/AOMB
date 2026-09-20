"""Tests for stream-capped Tale train/score pipeline (fixture + synthetic)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "corpus" / "fixtures" / "tale_of_errors_sample"
SCRIPT = ROOT / "scripts" / "tale_capped_train_score.sh"


def _jaeger_doc(trace_id: str, n_spans: int = 2) -> dict:
    spans = []
    for i in range(n_spans):
        spans.append(
            {
                "traceID": trace_id,
                "spanID": f"{i:016x}",
                "operationName": f"op-{i}",
                "references": [],
                "startTime": 1_694_430_060_100_000 + i * 1000,
                "duration": 1000 + i,
                "tags": [],
                "logs": [],
                "processID": "p1",
            }
        )
    return {
        "data": [
            {
                "traceID": trace_id,
                "spans": spans,
                "processes": {"p1": {"serviceName": "svc-a", "tags": []}},
            }
        ]
    }


def _build_tar_zst(members: dict[str, bytes], dest: Path) -> Path:
    import zstandard as zstd

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    tar_buf.seek(0)
    cctx = zstd.ZstdCompressor(level=3)
    dest.write_bytes(cctx.compress(tar_buf.read()))
    return dest


class TestTaleCappedPipeline(unittest.TestCase):
    def test_fixture_shards_and_score_dry_run(self):
        from corpus.ingest.tale_capped_pipeline import run_pipeline

        self.assertTrue(FIXTURE.is_dir(), "fixture must exist for CI")
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            score_out = os.path.join(tmp, "score.json")
            result = run_pipeline(
                max_spans=50,
                fixture=True,
                num_train_shards=1,
                data_dir=data_dir,
                score_dry_run=True,
                score_out=score_out,
            )
            self.assertEqual(result.claim_status, "not_published")
            self.assertIn("fixture", result.stages)
            self.assertIn("build_shards", result.stages)
            self.assertIn("score_dry_run", result.stages)
            shards = list(Path(data_dir).glob("shard_*.parquet"))
            self.assertGreaterEqual(len(shards), 1)
            report = json.loads(Path(score_out).read_text(encoding="utf-8"))
            self.assertEqual(report["claim_status"], "not_published")
            self.assertEqual(report["train_meta"]["mode"], "dry_run")
            # Must not invent a measured val_bpb in the orchestration result.
            blob = json.dumps(result.to_dict())
            self.assertNotIn("val_bpb=", blob)
            self.assertNotRegex(blob.lower(), r"val_bpb\"?\s*[:=]\s*[0-9]")

    def test_synthetic_extract_stops_at_caps(self):
        from corpus.ingest.tale_capped_pipeline import run_pipeline

        members = {}
        for i in range(10):
            tid = f"{i:032x}"
            members[f"traces/trace_{i:02d}.json"] = (
                json.dumps(_jaeger_doc(tid, n_spans=4)).encode("utf-8") + b"\n"
            )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive = _build_tar_zst(members, tmp_path / "trace1-sanitized.tar.zst")
            data_dir = tmp_path / "data"
            extract_out = tmp_path / "extracted"
            result = run_pipeline(
                max_spans=12,
                extract_input=str(archive),
                extract_out=str(extract_out),
                extract_max_files=3,
                num_train_shards=1,
                data_dir=str(data_dir),
            )
            self.assertIn("extract", result.stages)
            self.assertIn("build_shards", result.stages)
            self.assertIsNotNone(result.extract)
            # Caps must stop early (not full archive materialization).
            self.assertIn(
                result.extract["stopped_reason"],
                {"max-spans", "max-files"},
            )
            self.assertLessEqual(result.extract["files_written"], 3)
            written = list((extract_out / "traces").rglob("*.json"))
            self.assertLessEqual(len(written), 3)
            self.assertGreaterEqual(len(list(data_dir.glob("shard_*.parquet"))), 1)
            self.assertEqual(result.claim_status, "not_published")

    def test_refuses_auroc_full_decompress_uncapped(self):
        from corpus.ingest.tale_capped_pipeline import main

        with self.assertRaises(SystemExit) as ctx:
            main(["--auroc", "--fixture", "--max-spans", "10"])
        self.assertIn("auroc", str(ctx.exception).lower())

        with self.assertRaises(SystemExit) as ctx2:
            main(["--full-decompress", "--fixture", "--max-spans", "10"])
        msg2 = str(ctx2.exception).lower()
        self.assertTrue("full" in msg2 or "decompress" in msg2)

        with self.assertRaises(SystemExit) as ctx3:
            main(["--uncapped", "--fixture", "--max-spans", "10"])
        self.assertIn("uncapped", str(ctx3.exception).lower())

        with self.assertRaises(SystemExit) as ctx4:
            main(["--invent-metrics", "--fixture", "--max-spans", "10"])
        self.assertIn("invent", str(ctx4.exception).lower())

    def test_missing_input_exits_nonzero(self):
        from corpus.ingest.tale_capped_pipeline import main, run_pipeline

        with self.assertRaises(SystemExit) as ctx:
            run_pipeline(max_spans=10)  # no fixture / tree / extract
        self.assertIn("specify one of", str(ctx.exception).lower())

        with self.assertRaises(SystemExit) as ctx2:
            run_pipeline(
                max_spans=10,
                jaeger_tree="/tmp/aomb-definitely-missing-jaeger-tree-xyz",
            )
        self.assertIn("not found", str(ctx2.exception).lower())

        with self.assertRaises(SystemExit) as ctx3:
            run_pipeline(
                max_spans=10,
                extract_input="/tmp/aomb-missing-archive.tar.zst",
            )
        self.assertIn("not found", str(ctx3.exception).lower())

        # CLI: missing --max-spans / missing source → non-zero SystemExit
        with self.assertRaises(SystemExit) as ctx_cli:
            main([])
        self.assertNotEqual(ctx_cli.exception.code, 0)

        with self.assertRaises(SystemExit) as ctx_cli2:
            main(["--fixture"])  # argparse requires --max-spans
        self.assertNotEqual(ctx_cli2.exception.code, 0)

        rc3 = main(["--max-spans", "10"])  # no source
        self.assertEqual(rc3, 2)

    def test_prepare_refused_on_non_darwin(self):
        from corpus.ingest import tale_capped_pipeline as pipe

        if sys.platform == "darwin":
            self.skipTest("prepare gate is for non-Darwin CI hosts")
        with self.assertRaises(SystemExit) as ctx:
            pipe.run_prepare(1, str(pipe.DEFAULT_DATA_DIR))
        self.assertIn("macos", str(ctx.exception).lower())

    def test_no_auroc_calculator_in_module(self):
        src = (ROOT / "corpus" / "ingest" / "tale_capped_pipeline.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("def compute_auroc", src)
        self.assertNotIn("roc_auc", src)
        self.assertNotIn("sklearn", src)
        self.assertIn("not_published", src)
        self.assertIn("sacred", src.lower())

    def test_shell_wrapper_refuses_and_runs_fixture(self):
        self.assertTrue(SCRIPT.is_file())
        # Loud refusal via wrapper
        bad = subprocess.run(
            ["bash", str(SCRIPT), "--auroc", "--fixture", "--max-spans", "5"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("auroc", (bad.stderr + bad.stdout).lower())

        bad2 = subprocess.run(
            ["bash", str(SCRIPT), "--uncapped", "--fixture", "--max-spans", "5"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(bad2.returncode, 0)

        # Missing input
        missing = subprocess.run(
            ["bash", str(SCRIPT), "--jaeger-tree", "/tmp/nope-aomb", "--max-spans", "5"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(missing.returncode, 0)

        with tempfile.TemporaryDirectory() as tmp:
            ok = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--fixture",
                    "--max-spans",
                    "40",
                    "--data-dir",
                    tmp,
                    "--score-dry-run",
                    "--score-out",
                    os.path.join(tmp, "s.json"),
                ],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                ok.returncode,
                0,
                msg=f"stdout={ok.stdout}\nstderr={ok.stderr}",
            )
            self.assertTrue(Path(tmp, "s.json").is_file())
            report = json.loads(Path(tmp, "s.json").read_text(encoding="utf-8"))
            self.assertEqual(report["claim_status"], "not_published")


if __name__ == "__main__":
    unittest.main()
