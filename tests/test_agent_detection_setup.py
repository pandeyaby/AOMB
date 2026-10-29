"""Safety rails for the agent-loop ↔ detection experiment."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import agent_loop
from corpus.ingest.loghub_hdfs import select_val
from corpus.ingest.sessions_to_shards import write


class TestNoPush(unittest.TestCase):
    def test_no_push_env_skips_git_push(self):
        with mock.patch.object(agent_loop, "git") as git, mock.patch.dict(os.environ, {"AOMB_NO_PUSH": "1"}):
            agent_loop.push_to_remote(10)
        git.assert_not_called()

    def test_no_push_covers_the_final_push(self):
        import inspect

        src = inspect.getsource(agent_loop.main)
        tail = src[src.index("LOOP COMPLETE"):]
        self.assertIn("push_disabled()", tail.split('git("push"')[0])

    def test_push_still_happens_without_flag(self):
        env = {k: v for k, v in os.environ.items() if k != "AOMB_NO_PUSH"}
        with mock.patch.object(agent_loop, "git") as git, mock.patch.dict(os.environ, env, clear=True):
            agent_loop.push_to_remote(10)
        git.assert_called_once_with("push", "origin", "main", check=False)


class _Block:
    def __init__(self, type_, text=""):
        self.type, self.text = type_, text


class _Msg:
    def __init__(self, content, stop_reason="end_turn"):
        self.content, self.stop_reason, self.model, self.stop_details = content, stop_reason, "m", None


class _FakeAnthropic:
    """Minimal stand-in for anthropic.Anthropic used by _call_via_sdk."""

    last_kwargs: dict = {}
    reply = _Msg([])

    def __init__(self, api_key):
        outer = self

        class _Stream:
            def __enter__(self_):
                return self_

            def __exit__(self_, *a):
                return False

            def get_final_message(self_):
                return _FakeAnthropic.reply

        class _BetaMessages:
            def stream(self_, **kwargs):
                _FakeAnthropic.last_kwargs = kwargs
                return _Stream()

        self.beta = type("B", (), {"messages": _BetaMessages()})()


class _FakeModule:
    Anthropic = _FakeAnthropic
    RateLimitError = type("RateLimitError", (Exception,), {})
    APIStatusError = type("APIStatusError", (Exception,), {})


class TestSdkCall(unittest.TestCase):
    def _call(self, model="sonnet"):
        with mock.patch.object(agent_loop, "_anthropic_module", _FakeModule, create=True):
            return agent_loop._call_via_sdk("p", model, "key-123456", 1, 1)

    def test_skips_thinking_blocks_and_uses_current_model(self):
        _FakeAnthropic.reply = _Msg([_Block("thinking"), _Block("text", "```python\nx=1\n```")])
        self.assertEqual(self._call(), "```python\nx=1\n```")
        kw = _FakeAnthropic.last_kwargs
        self.assertEqual(kw["model"], "claude-sonnet-5-5")
        self.assertEqual(kw["max_tokens"], agent_loop.SDK_MAX_TOKENS)
        self.assertEqual(kw["fallbacks"], "default")

    def test_refusal_and_truncation_return_none(self):
        for stop in ("refusal", "max_tokens"):
            _FakeAnthropic.reply = _Msg([_Block("text", "partial")], stop_reason=stop)
            self.assertIsNone(self._call())

    def test_haiku_gets_no_fallback(self):
        _FakeAnthropic.reply = _Msg([_Block("text", "ok")])
        self._call("haiku")
        self.assertNotIn("fallbacks", _FakeAnthropic.last_kwargs)


class TestInjectObjective(unittest.TestCase):
    def test_parse_takes_the_final_inject_auroc_line(self):
        out = "inject_truncate: 0.9\ninject_auroc:     0.612345\nnoise\ninject_auroc: 0.700000\n"
        self.assertEqual(agent_loop.parse_inject_auroc(out), 0.7)
        self.assertIsNone(agent_loop.parse_inject_auroc("val_bpb: 0.3\n"))

    def test_improvement_direction_and_margin(self):
        with mock.patch.object(agent_loop, "OBJECTIVE", "inject"), mock.patch.object(agent_loop, "MIN_DELTA", 0.01):
            self.assertTrue(agent_loop.is_improvement(0.72, 0.70))
            self.assertFalse(agent_loop.is_improvement(0.705, 0.70))  # inside the noise margin
            self.assertFalse(agent_loop.is_improvement(0.60, 0.70))
        with mock.patch.object(agent_loop, "OBJECTIVE", "val_bpb"), mock.patch.object(agent_loop, "MIN_DELTA", 0.0):
            self.assertTrue(agent_loop.is_improvement(0.30, 0.31))
            self.assertFalse(agent_loop.is_improvement(0.32, 0.31))

    def test_best_inject_from_env_then_git_subjects(self):
        with mock.patch.dict(os.environ, {"AOMB_BEST_INJECT_AUROC": "0.61"}):
            self.assertEqual(agent_loop.get_best_inject_auroc(), 0.61)
        env = {k: v for k, v in os.environ.items() if k != "AOMB_BEST_INJECT_AUROC"}
        log = "[inject_auroc=0.6400] [Δ=+0.0200] [val_bpb=0.31] [change: x]\n[val_bpb=0.30] old\n[inject_auroc=0.6200] y"
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(agent_loop, "git", return_value=log):
            self.assertEqual(agent_loop.get_best_inject_auroc(), 0.64)

    def test_commit_subject_leads_with_the_kept_metric(self):
        with mock.patch.object(agent_loop, "git") as git:
            agent_loop.commit_experiment(3, 0.31, 0.60, "wider model", inject_auroc=0.65)
        msg = git.call_args_list[-1].args[2]
        self.assertTrue(msg.startswith("[inject_auroc=0.6500] [Δ=+0.0500] [val_bpb=0.3100] [change: wider model]"))

    def test_prompt_states_the_objective(self):
        with mock.patch.object(agent_loop, "OBJECTIVE", "inject"), \
             mock.patch.object(agent_loop, "get_best_inject_auroc", return_value=0.61), \
             mock.patch.object(agent_loop, "get_best_val_bpb", return_value=0.3), \
             mock.patch.object(agent_loop, "get_recent_experiments", return_value="none"):
            prompt = agent_loop._build_prompt(1)
        self.assertIn("inject_auroc (HIGHER is better)", prompt)
        self.assertIn("Best inject_auroc so far: 0.6100", prompt)
        self.assertNotIn("Best val_bpb so far", prompt)


class TestShards(unittest.TestCase):
    def _jsonl(self, d: Path, name: str, rows: list[dict]) -> Path:
        p = d / name
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return p

    def test_writes_train_and_val_shards_from_normals(self):
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            train = self._jsonl(d, "t.jsonl", [
                {"session_id": "a", "label": "normal", "split": "train", "text": "x"},
                {"session_id": "b", "label": "incident", "split": "eval", "text": "y"},  # eval rows ignored
                {"session_id": "c", "label": "normal", "split": "train", "text": "z"},
            ])
            val = self._jsonl(d, "v.jsonl", [{"session_id": "v", "label": "normal", "split": "val", "text": "w"}])
            out = write(train, val, d / "data", num_train_shards=2)
            self.assertEqual(out["train_docs"], 2)
            self.assertIn("shard_06542.parquet", out["shards"])
            self.assertEqual(pq.read_table(d / "data" / "shard_06542.parquet").column("text").to_pylist(), ["w"])
            with self.assertRaises(SystemExit):  # refuses to overwrite an existing cache
                write(train, val, d / "data", num_train_shards=2)

    def test_refuses_incident_in_train_or_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            train = self._jsonl(d, "t.jsonl", [{"session_id": "a", "label": "normal", "split": "train", "text": "x"}])
            val = self._jsonl(d, "v.jsonl", [{"session_id": "v", "label": "incident", "split": "val", "text": "w"}])
            with self.assertRaises(ValueError):
                write(train, val, d / "data", num_train_shards=1)


class TestAgentCommitsSummary(unittest.TestCase):
    def test_spearman_sign(self):
        from eval.agent_commits import spearman

        val_bpb = [0.30, 0.28, 0.26, 0.25]
        self.assertAlmostEqual(spearman(val_bpb, [0.70, 0.72, 0.75, 0.80]), -1.0)  # lower bpb, higher AUROC
        self.assertAlmostEqual(spearman(val_bpb, [0.80, 0.75, 0.72, 0.70]), 1.0)
        self.assertNotEqual(spearman([1, 2], [1, 2]), spearman([1, 2], [1, 2]))  # NaN below 3 points

    def test_spearman_averages_ties(self):
        from eval.agent_commits import spearman

        # scipy.stats.spearmanr([1, 1, 2, 3], [1, 2, 3, 4]) = 0.9486832980505138
        self.assertAlmostEqual(spearman([1, 1, 2, 3], [1, 2, 3, 4]), 0.9486832980505138)


class TestValSelection(unittest.TestCase):
    def test_val_is_normal_and_disjoint(self):
        labels = [(f"b{i}", "Anomaly" if i % 5 == 0 else "Normal") for i in range(50)]
        exclude = {"b1", "b2", "b3"}
        val = select_val(labels, exclude, 10, seed=0)
        self.assertEqual(len(val), 10)
        self.assertFalse(set(val) & exclude)
        self.assertTrue(all(dict(labels)[b] == "Normal" for b in val))
        self.assertEqual(select_val(labels, exclude, 0, seed=0), [])


if __name__ == "__main__":
    unittest.main()
