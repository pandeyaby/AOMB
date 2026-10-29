"""The agent can't forge the score its changes are kept for (validation audit, 2026-10-29)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import agent_loop
from eval import objective

SESSIONS = [
    "op=GET_/a svc=api duration_ms=5 status=ok region=us\nop=GET_/b svc=db duration_ms=7 status=ok region=us\nop=GET_/c svc=api duration_ms=3 status=ok region=eu",
    "op=GET_/a svc=api duration_ms=6 status=ok region=eu\nop=GET_/b svc=db duration_ms=8 status=ok region=eu\nop=GET_/c svc=api duration_ms=4 status=ok region=us",
]


def _write_jsonl(path: Path, split: str) -> None:
    path.write_text(
        "".join(json.dumps({"session_id": f"{split}{i}", "label": "normal", "split": split, "text": t}) + "\n"
                for i, t in enumerate(SESSIONS)),
        encoding="utf-8",
    )


class TestRunTrainingTrustsOnlyCleanStdout(unittest.TestCase):
    def _run(self, stdout="", stderr="", returncode=0, timeout=False):
        def fake_run(*_a, **_k):
            if timeout:
                raise subprocess.TimeoutExpired(cmd="x", timeout=1, output=stdout.encode(), stderr=stderr.encode())
            return subprocess.CompletedProcess(args="x", returncode=returncode, stdout=stdout, stderr=stderr)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(agent_loop, "LOGS_DIR", Path(tmp)), \
                mock.patch.object(agent_loop.subprocess, "run", side_effect=fake_run):
            return agent_loop.run_training(1)

    def test_clean_run_is_scored(self):
        out, bpb = self._run(stdout="val_bpb: 0.5\ninject_auroc: 0.7\n")
        self.assertEqual(bpb, 0.5)
        self.assertEqual(agent_loop.parse_inject_auroc(out), 0.7)

    def test_crash_after_printing_a_score_is_not_scored(self):
        _out, bpb = self._run(stdout="val_bpb: 0.5\ninject_auroc: 0.99\n", returncode=1)
        self.assertIsNone(bpb)

    def test_timeout_is_not_scored(self):
        _out, bpb = self._run(stdout="val_bpb: 0.5\n", timeout=True)
        self.assertIsNone(bpb)

    def test_score_on_stderr_is_ignored(self):
        out, bpb = self._run(stdout="val_bpb: 0.5\ninject_auroc: 0.6\n", stderr="inject_auroc: 0.999\n")
        self.assertEqual(bpb, 0.5)
        self.assertEqual(agent_loop.parse_inject_auroc(out), 0.6)


class TestObjectiveRunsAgentCodeInAChild(unittest.TestCase):
    def _main(self, child_stdout="val_bpb: 0.5\n", returncode=0):
        calls = []

        def fake_run(cmd, **kw):
            rows = [json.loads(line) for line in Path(cmd[-2]).read_text(encoding="utf-8").splitlines()]
            calls.append((cmd, kw, rows))
            Path(cmd[-1]).write_text("".join("[1.0]\n" for _ in rows), encoding="utf-8")
            return subprocess.CompletedProcess(cmd, returncode, stdout=child_stdout, stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            val, train = Path(tmp) / "val.jsonl", Path(tmp) / "train.jsonl"
            _write_jsonl(val, "val")
            _write_jsonl(train, "train")
            env = {"AOMB_OBJECTIVE_VAL": str(val), "AOMB_OBJECTIVE_TRAIN": str(train)}
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(objective.subprocess, "run", side_effect=fake_run), \
                    mock.patch.object(objective, "score_texts", side_effect=lambda texts, nats: [0.0] * len(texts)), \
                    mock.patch.object(objective.sys, "argv", ["objective"]), \
                    redirect_stdout(out), redirect_stderr(err):
                rc = objective.main()
        return rc, out.getvalue(), calls

    def test_agent_code_runs_in_a_separate_process(self):
        _rc, _out, calls = self._main()
        cmd, _kw, _rows = calls[0]
        self.assertIn("--child", cmd)

    def test_child_is_not_given_the_data_paths(self):
        _rc, _out, calls = self._main()
        env = calls[0][1]["env"]
        for key in ("AOMB_OBJECTIVE_VAL", "AOMB_OBJECTIVE_TRAIN"):
            self.assertNotIn(key, env)

    def test_child_texts_carry_no_labels(self):
        _rc, _out, calls = self._main()
        rows = calls[0][2]
        self.assertTrue(rows)
        for row in rows:
            self.assertEqual(set(row), {"text"})

    def test_score_lines_printed_by_the_child_are_dropped(self):
        rc, out, _calls = self._main(child_stdout="val_bpb: 0.5\ninject_auroc: 0.999\n")
        self.assertEqual(rc, 0)
        self.assertNotIn("0.999", out)
        self.assertEqual(agent_loop.parse_inject_auroc(out), 0.5)  # equal scores → chance

    def test_child_failure_fails_the_objective(self):
        rc, out, _calls = self._main(child_stdout="inject_auroc: 0.999\n", returncode=1)
        self.assertEqual(rc, 1)
        self.assertIsNone(agent_loop.parse_inject_auroc(out))


class TestScoreTextsChecksTheChild(unittest.TestCase):
    class _Tok:
        def get_bos_token_id(self):
            return 0

        def encode(self, text, prepend=None):
            return [prepend] + [1] * len(text)

        def decode(self, ids):
            return "a"

    class _Bytes:
        def tolist(self):
            return [0, 1]

    def _score(self, texts, nats):
        import prepare

        with mock.patch.object(prepare.Tokenizer, "from_directory", return_value=self._Tok()), \
                mock.patch.object(prepare, "get_token_bytes", return_value=self._Bytes()):
            return objective.score_texts(texts, nats)

    def test_accepts_one_finite_value_per_token(self):
        self.assertEqual(len(self._score(["abc"], [[1.0, 2.0, 3.0]])), 1)

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            self._score(["abc"], [[1.0, 2.0]])

    def test_rejects_non_finite(self):
        with self.assertRaises(ValueError):
            self._score(["abc"], [[1.0, float("nan"), 3.0]])

    def test_rejects_missing_rows(self):
        with self.assertRaises(ValueError):
            self._score(["abc", "de"], [[1.0, 2.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
