"""Tests for the in-domain lab eval (eval/in_domain.py) — pure logic, no torch."""

from __future__ import annotations

import math
import unittest

from eval.in_domain import (
    ValueStats,
    chunk_lines,
    load_session_file,
    _noise_mask,
    baseline_scores,
    fit_duration_stats,
    fit_novelty,
    log_template,
    model_scores_for,
    temporal_split,
)
from eval.labels import LabeledSession


def _sess(sid, label, text, fault=""):
    return LabeledSession(
        session_id=sid,
        text=text,
        label=label,
        binary={"normal": 0, "incident": 1}[label],
        fault=fault,
        n_chars=len(text),
    )


def _span(ts, op, dur, status="ok"):
    return (
        f"[ts=2026-09-18T05:23:{ts:02d}.000Z] [src=OTel] trace_id=ab12 span_id=cd34 "
        f"parent=n/a op={op} svc=api duration_ms={dur} status={status}"
    )


class TestTemporalSplit(unittest.TestCase):
    def test_earlier_normals_train_later_normals_and_incidents_eval(self):
        sessions = [_sess(f"n{i}", "normal", _span(i, "GET", 5)) for i in range(4)]
        sessions.append(_sess("i0", "incident", _span(30, "GET", 800)))
        cap = {s.session_id: "c1" for s in sessions}
        train, ev = temporal_split(sessions, cap)
        self.assertEqual([s.session_id for s in train], ["n0", "n1"])
        self.assertEqual({s.session_id for s in ev}, {"n2", "n3", "i0"})
        self.assertTrue(all(s.binary == 0 for s in train))


class TestBaselines(unittest.TestCase):
    def test_duration_z_and_error_rule(self):
        train = [_sess(f"n{i}", "normal", _span(i, "GET", 4 + i % 3)) for i in range(6)]
        stats = fit_duration_stats(train)
        ev = [
            _sess("ok", "normal", _span(10, "GET", 5)),
            _sess("slow", "incident", _span(11, "GET", 800)),
            _sess("err", "incident", _span(12, "GET", 5, status="error")),
        ]
        sc = baseline_scores(ev, stats)
        self.assertGreater(sc["duration_z"][1], 5 * max(sc["duration_z"][0], 0.1))
        self.assertEqual(sc["error_lines"], [0.0, 0.0, 1.0])
        # error outranks everything under the rule; slow outranks ok
        self.assertEqual(sorted(range(3), key=lambda i: sc["rule"][i]), [0, 1, 2])


class TestNoveltyBaseline(unittest.TestCase):
    def _log(self, msg, level="INFO"):
        return f"[ts=2026-09-18T05:23:01.000Z] [src=OTelLog] level={level} svc=api msg={msg}"

    def test_template_masks_values_but_not_new_messages(self):
        self.assertEqual(
            log_template("INFO", "api", "checkout_ok_db=ok_hits=7"),
            log_template("INFO", "api", "checkout_ok_db=replica_hits=123"),
        )
        self.assertNotEqual(
            log_template("INFO", "api", "checkout_ok_db=ok_hits=7"),
            log_template("WARN", "api", "pricing_fallback_source=static"),
        )

    def test_novelty_flags_new_template_op_and_shape_only(self):
        normal = "\n".join([_span(1, "GET", 5), self._log("checkout_ok_db=ok_hits=7")])
        train = [_sess("n0", "normal", normal)]
        stats = fit_duration_stats(train)
        seen = fit_novelty(train)
        ev = [
            _sess("same", "normal", normal.replace("hits=7", "hits=9")),
            _sess("drift", "incident", normal.replace("db=ok", "db=replica")),
            _sess("newlog", "incident", normal + "\n" + self._log("pricing_fallback_x=1", "WARN")),
            _sess("retry", "incident", normal + "\n" + _span(2, "GET", 5)),
        ]
        nov = baseline_scores(ev, stats, seen)["novelty"]
        self.assertEqual(nov[0], 0.0)
        self.assertEqual(nov[1], 0.0)  # value drift is invisible to templates
        self.assertEqual(nov[2], 1.0)  # new template
        self.assertEqual(nov[3], 1.0)  # new trace shape (extra span)


class TestValueBaselines(unittest.TestCase):
    def _checkout(self, region, currency, pricing="v1", hits=7):
        return (
            "[ts=2026-09-26T05:23:01.000Z] [src=OTelLog] level=INFO svc=api "
            f"msg=checkout_ok_db=ok_hits={hits}_region={region}_currency={currency}_pricing={pricing}"
        )

    def test_value_novelty_rarity_and_pairs(self):
        train = [
            _sess(f"n{i}", "normal", self._checkout(r, c, "v2" if i == 0 else "v1", hits=i))
            for i, (r, c) in enumerate([("us-east", "USD"), ("eu-west", "EUR")] * 5)
        ]
        vs = ValueStats(train)
        self.assertNotIn("hits", vs.counts)  # numeric → never categorical
        normal = vs.score(self._checkout("us-east", "USD", hits=99))
        new_region = vs.score(self._checkout("ap-east", "USD"))
        swapped = vs.score(self._checkout("us-east", "EUR"))
        rare = vs.score(self._checkout("us-east", "USD", "v2"))
        self.assertEqual(normal[0], 0.0)
        self.assertEqual(new_region[0], 1.0)  # unseen value
        self.assertEqual(swapped[0], 0.0)  # every value familiar...
        self.assertGreater(swapped[2], normal[2])  # ...but the pairing is new
        self.assertGreater(rare[1], normal[1])  # v2 is rarer than v1


class TestShortnessBaseline(unittest.TestCase):
    def test_too_short_flags_only_sessions_under_half_the_training_minimum(self):
        from eval.in_domain import min_lines

        line = _span(1, "GET", 5)
        train = [_sess(f"n{i}", "normal", "\n".join([line] * k)) for i, k in enumerate((10, 12, 14))]
        self.assertEqual(min_lines(train), 10)
        ev = [
            _sess("full", "normal", "\n".join([line] * 11)),
            _sess("half", "normal", "\n".join([line] * 5)),   # exactly half: not flagged
            _sess("cut", "incident", "\n".join([line] * 4)),
        ]
        sc = baseline_scores(ev, fit_duration_stats(train), fit_novelty(train), ValueStats(train), min_lines(train))
        self.assertEqual(sc["too_short"], [0.0, 0.0, 1.0])
        self.assertGreater(sc["rarity_or_short"][2], max(sc["rarity_or_short"][:2]))


class TestSessionFiles(unittest.TestCase):
    def test_load_session_file_split_and_sequence_novelty(self):
        import json
        import tempfile
        from pathlib import Path

        log = "[ts=081109T203518] [src=OTelLog] level=INFO svc=dfs.X msg={}"
        rows = [
            {"session_id": "a", "label": "normal", "split": "train", "group": "g", "text": log.format("Receiving_blk")},
            {"session_id": "b", "label": "normal", "split": "eval", "group": "g", "text": log.format("Receiving_blk")},
            {"session_id": "c", "label": "incident", "split": "eval", "group": "g",
             "text": log.format("Receiving_blk") + "\n" + log.format("Receiving_blk")},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "s.jsonl"
            f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            train, ev, group_of, meta = load_session_file(f)
        self.assertEqual([s.session_id for s in train], ["a"])
        self.assertEqual([s.session_id for s in ev], ["b", "c"])
        self.assertEqual(group_of["c"], "g")
        self.assertEqual(len(meta["content_sha256"]), 64)
        sc = baseline_scores(ev, fit_duration_stats(train), fit_novelty(train))
        # same template twice: no new template, but a new event multiset
        self.assertEqual(sc["novelty"][1], 0.0)
        self.assertEqual(sc["sequence_novelty"], [0.0, 1.0])

    def test_training_session_must_be_normal(self):
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "s.jsonl"
            f.write_text(json.dumps({"session_id": "x", "label": "incident", "split": "train", "text": "t"}) + "\n")
            with self.assertRaises(ValueError):
                load_session_file(f)

    def test_end_marker_lands_only_on_the_last_chunk_and_is_scored(self):
        from eval.in_domain import END_MARKER

        text = "\n".join(f"l{i}" for i in range(5))
        chunks = chunk_lines([text], 4, END_MARKER)
        self.assertEqual(sum(END_MARKER in c for c in chunks), 1)
        self.assertTrue(chunks[-1].endswith(END_MARKER))
        self.assertEqual(chunk_lines([text], None, END_MARKER), [text + "\n" + END_MARKER])

        scored = _span(1, "GET", 5) + "\n" + END_MARKER
        toks = [(c, 2.0 if i >= len(scored) - len(END_MARKER) else 0.5, 1) for i, c in enumerate(scored)]
        out = model_scores_for(toks, scored)
        self.assertAlmostEqual(out["bits_end"], len(END_MARKER) * 2.0 / math.log(2))
        self.assertEqual(model_scores_for(toks[: -len(END_MARKER)], _span(1, "GET", 5) + "\n")["bits_end"], 0.0)

    def test_chunk_lines_keeps_every_line(self):
        text = "\n".join(f"l{i}" for i in range(10))
        chunks = chunk_lines([text], 4)
        self.assertEqual([c.count("\n") + 1 for c in chunks], [4, 4, 2])
        self.assertEqual("\n".join(chunks), text)
        self.assertEqual(chunk_lines([text], None), [text])


class TestModelScoring(unittest.TestCase):
    def test_noise_mask_covers_ids_and_timestamp_only(self):
        text = _span(1, "GET", 150)
        toks = [(c, 1.0, 1) for c in text]
        masked = "".join("_" if m else c for c, m in zip(text, _noise_mask(text, toks)))
        self.assertIn("[ts=________________________]", masked)
        self.assertIn("trace_id=____", masked)
        self.assertIn("parent=___", masked)
        self.assertIn("duration_ms=150", masked)

    def test_noise_mask_covers_monotonic_counter(self):
        text = "[src=OTelLog] level=INFO svc=api msg=checkout_ok_db=ok_hits=1234_region=us-east"
        toks = [(c, 1.0, 1) for c in text]
        masked = "".join("_" if m else c for c, m in zip(text, _noise_mask(text, toks)))
        self.assertIn("hits=____", masked)
        self.assertIn("region=us-east", masked)

    def test_max_field_picks_the_surprising_value_and_ignores_ids(self):
        text = _span(1, "GET", 5) + " region=us-east"
        toks = [(c, 1.0, 1) for c in text]
        mask = _noise_mask(text, toks)
        base = model_scores_for(toks, text)["bits_max_field"]
        start = text.index("us-east")
        spiked = [
            (c, 30.0 if start <= i < start + 7 else (90.0 if m else n), b)
            for i, ((c, n, b), m) in enumerate(zip(toks, mask))
        ]
        out = model_scores_for(spiked, text)["bits_max_field"]
        self.assertAlmostEqual(out, 7 * 30.0 / math.log(2))  # region value, not the IDs
        self.assertGreater(out, base)

    def test_content_score_ignores_surprise_in_ids(self):
        text = _span(1, "GET", 5)
        toks = [(c, 1.0, 1) for c in text]
        base = model_scores_for(toks, text)
        # make every ID/timestamp char hugely surprising — content score must not move
        mask = _noise_mask(text, toks)
        noisy = [(c, 50.0 if m else n, b) for (c, n, b), m in zip(toks, mask)]
        out = model_scores_for(noisy, text)
        self.assertAlmostEqual(out["bpb_content"], base["bpb_content"])
        self.assertGreater(out["bpb_mean"], base["bpb_mean"])
        self.assertAlmostEqual(base["bpb_content"], 1.0 / math.log(2))


if __name__ == "__main__":
    unittest.main()
