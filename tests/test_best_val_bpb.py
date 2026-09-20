"""Unit tests: agent_loop best-val override / CRISP-Tale isolation.

No MPS. No API keys. No invented AUROC / val_bpb floor. prepare.py untouched.
CUDA gate stays skipped.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from best_val_bpb import (  # noqa: E402
    best_val_from_commit_subjects,
    commit_matches_lane,
    format_lane_commit_tags,
    normalize_lane_key,
    parse_best_val_override,
    parse_commit_lane_tags,
    resolve_best_val_bpb,
)


# Synthetic smoke-era best that must NOT poison a CRISP overnight.
_SYNTHETIC = (
    "[val_bpb=0.3682] [Δ=-0.0003] [change: adam] [exp: 114]"
)
_CRISP_A = (
    "[val_bpb=0.4554] [Δ=+0.0000] [change: start] [exp: 1] "
    "[corpus=uber-crisp-zenodo-13956078]"
)
_CRISP_B = (
    "[val_bpb=0.4309] [Δ=-0.0087] [change: win] [exp: 20] "
    "[corpus=crisp] [source_id=uber-crisp-zenodo-13956078]"
)
_TALE = (
    "[val_bpb=0.5120] [Δ=+0.0000] [change: tale start] [exp: 1] "
    "[corpus=tale] [source_id=uber-tale-of-errors]"
)
_MALFORMED = "[val_bpb=pending] [change: bad] [exp: 9] [corpus=crisp]"
_UNTAGGED_CRISPISH = "[val_bpb=0.4309] [Δ=-0.01] [change: legacy] [exp: 20]"


class TestParseBestValOverride(unittest.TestCase):
    def test_override_sane(self):
        self.assertAlmostEqual(parse_best_val_override("0.4309"), 0.4309, places=4)

    def test_override_missing_refuses_invent(self):
        self.assertIsNone(parse_best_val_override(None))
        self.assertIsNone(parse_best_val_override(""))
        self.assertIsNone(parse_best_val_override("   "))

    def test_override_malformed_refuses_invent(self):
        for bad in ("pending", "nan", "inf", "n/a", "auto", "not-a-float", "-1", "999"):
            self.assertIsNone(parse_best_val_override(bad), msg=bad)


class TestLaneNormalization(unittest.TestCase):
    def test_crisp_aliases(self):
        self.assertEqual(
            normalize_lane_key("crisp"), "uber-crisp-zenodo-13956078"
        )
        self.assertEqual(
            normalize_lane_key("uber-crisp-zenodo-13956078"),
            "uber-crisp-zenodo-13956078",
        )

    def test_tale_aliases(self):
        self.assertEqual(normalize_lane_key("tale"), "uber-tale-of-errors")
        self.assertEqual(
            normalize_lane_key("uber-tale-of-errors"), "uber-tale-of-errors"
        )

    def test_synthetic_aliases(self):
        self.assertEqual(normalize_lane_key("smoke"), "synthetic-smoke")


class TestCommitLaneIsolation(unittest.TestCase):
    def test_untagged_matches_only_when_no_filter(self):
        self.assertTrue(commit_matches_lane(_SYNTHETIC))
        self.assertFalse(
            commit_matches_lane(_SYNTHETIC, corpus="crisp")
        )
        self.assertFalse(
            commit_matches_lane(
                _SYNTHETIC, source_id="uber-crisp-zenodo-13956078"
            )
        )

    def test_crisp_tag_matches_crisp_filter(self):
        self.assertTrue(commit_matches_lane(_CRISP_A, corpus="crisp"))
        self.assertTrue(
            commit_matches_lane(
                _CRISP_B, source_id="uber-crisp-zenodo-13956078"
            )
        )
        self.assertFalse(commit_matches_lane(_CRISP_A, corpus="tale"))

    def test_parse_tags(self):
        corpus, source = parse_commit_lane_tags(_CRISP_B)
        self.assertEqual(corpus, "uber-crisp-zenodo-13956078")
        self.assertEqual(source, "uber-crisp-zenodo-13956078")


class TestBestValFromSubjects(unittest.TestCase):
    def test_no_filter_takes_global_min_including_synthetic(self):
        subjects = [_SYNTHETIC, _CRISP_A, _CRISP_B, _TALE]
        best = best_val_from_commit_subjects(subjects)
        self.assertAlmostEqual(best, 0.3682, places=4)

    def test_crisp_filter_ignores_synthetic_and_tale(self):
        subjects = [_SYNTHETIC, _CRISP_A, _CRISP_B, _TALE, _MALFORMED]
        best = best_val_from_commit_subjects(subjects, corpus="crisp")
        self.assertAlmostEqual(best, 0.4309, places=4)

    def test_tale_filter_isolates(self):
        subjects = [_SYNTHETIC, _CRISP_B, _TALE]
        best = best_val_from_commit_subjects(subjects, corpus="tale")
        self.assertAlmostEqual(best, 0.5120, places=4)

    def test_missing_lane_returns_inf_never_invents_floor(self):
        subjects = [_SYNTHETIC, _UNTAGGED_CRISPISH, _MALFORMED]
        best = best_val_from_commit_subjects(subjects, corpus="crisp")
        self.assertTrue(math.isinf(best))
        # Must not invent documented CRISP floors
        self.assertNotAlmostEqual(best, 0.4309, places=4)
        self.assertNotAlmostEqual(best, 0.458756, places=6)
        self.assertNotAlmostEqual(best, 0.407753, places=6)

    def test_empty_subjects_inf(self):
        self.assertTrue(math.isinf(best_val_from_commit_subjects([])))
        self.assertTrue(
            math.isinf(best_val_from_commit_subjects([], corpus="crisp"))
        )


class TestResolveBestValBpb(unittest.TestCase):
    def test_env_override_wins(self):
        subjects = [_SYNTHETIC, _CRISP_B]
        env = {
            "AOMB_BEST_VAL_BPB": "0.4554",
            "AOMB_CORPUS": "crisp",
        }
        best = resolve_best_val_bpb(subjects, environ=env)
        self.assertAlmostEqual(best, 0.4554, places=4)

    def test_override_missing_falls_through_to_lane(self):
        subjects = [_SYNTHETIC, _CRISP_A, _CRISP_B]
        env = {"AOMB_CORPUS": "crisp"}  # no AOMB_BEST_VAL_BPB
        best = resolve_best_val_bpb(subjects, environ=env)
        self.assertAlmostEqual(best, 0.4309, places=4)

    def test_malformed_override_refuses_invent_falls_through(self):
        subjects = [_CRISP_B]
        env = {
            "AOMB_BEST_VAL_BPB": "pending",
            "AOMB_CORPUS": "crisp",
        }
        best = resolve_best_val_bpb(subjects, environ=env)
        self.assertAlmostEqual(best, 0.4309, places=4)

    def test_malformed_override_and_missing_lane_is_inf(self):
        env = {
            "AOMB_BEST_VAL_BPB": "nan",
            "AOMB_CORPUS": "crisp",
        }
        best = resolve_best_val_bpb([_SYNTHETIC], environ=env)
        self.assertTrue(math.isinf(best))

    def test_explicit_override_arg(self):
        best = resolve_best_val_bpb(
            [_SYNTHETIC],
            environ={},
            override="0.458756",
        )
        self.assertAlmostEqual(best, 0.458756, places=6)

    def test_source_id_env_isolation(self):
        subjects = [_SYNTHETIC, _TALE, _CRISP_B]
        env = {"AOMB_SOURCE_ID": "uber-tale-of-errors"}
        best = resolve_best_val_bpb(subjects, environ=env)
        self.assertAlmostEqual(best, 0.5120, places=4)


class TestFormatLaneCommitTags(unittest.TestCase):
    def test_corpus_only(self):
        self.assertEqual(
            format_lane_commit_tags(corpus="crisp"),
            "[corpus=uber-crisp-zenodo-13956078]",
        )

    def test_source_only(self):
        self.assertEqual(
            format_lane_commit_tags(source_id="uber-tale-of-errors"),
            "[source_id=uber-tale-of-errors]",
        )

    def test_both_dedup_when_same(self):
        tags = format_lane_commit_tags(
            corpus="crisp", source_id="uber-crisp-zenodo-13956078"
        )
        self.assertEqual(tags, "[corpus=uber-crisp-zenodo-13956078]")


class TestAgentLoopWiresHelper(unittest.TestCase):
    def test_get_best_val_bpb_uses_isolation(self):
        import agent_loop

        fake_log = "\n".join([_SYNTHETIC, _CRISP_A, _CRISP_B, _TALE])
        with mock.patch.dict(
            "os.environ", {"AOMB_CORPUS": "crisp"}, clear=False
        ):
            # Clear override if present in the real env
            with mock.patch.dict(
                "os.environ", {"AOMB_BEST_VAL_BPB": ""}, clear=False
            ):
                with mock.patch.object(
                    agent_loop, "git", return_value=fake_log
                ):
                    best = agent_loop.get_best_val_bpb()
        self.assertAlmostEqual(best, 0.4309, places=4)

    def test_get_best_val_bpb_override(self):
        import agent_loop

        with mock.patch.dict(
            "os.environ",
            {
                "AOMB_BEST_VAL_BPB": "0.5000",
                "AOMB_CORPUS": "crisp",
            },
            clear=False,
        ):
            with mock.patch.object(agent_loop, "git", return_value=_SYNTHETIC):
                best = agent_loop.get_best_val_bpb()
        self.assertAlmostEqual(best, 0.5000, places=4)

    def test_get_best_val_bpb_missing_lane_inf(self):
        import agent_loop

        with mock.patch.dict(
            "os.environ",
            {"AOMB_CORPUS": "crisp", "AOMB_BEST_VAL_BPB": ""},
            clear=False,
        ):
            with mock.patch.object(agent_loop, "git", return_value=_SYNTHETIC):
                best = agent_loop.get_best_val_bpb()
        self.assertTrue(math.isinf(best))

    def test_commit_experiment_tags_lane(self):
        import agent_loop

        calls: list[tuple] = []

        def fake_git(*args, **kwargs):
            calls.append(args)
            return ""

        with mock.patch.dict(
            "os.environ",
            {"AOMB_CORPUS": "crisp", "AOMB_SOURCE_ID": ""},
            clear=False,
        ):
            with mock.patch.object(agent_loop, "git", side_effect=fake_git):
                with mock.patch.object(agent_loop, "log"):
                    agent_loop.commit_experiment(3, 0.4400, 0.4500, "tweak lr")

        commit_msgs = [c for c in calls if c and c[0] == "commit"]
        self.assertEqual(len(commit_msgs), 1)
        msg = commit_msgs[0][2]
        self.assertIn("[val_bpb=0.4400]", msg)
        self.assertIn("[corpus=uber-crisp-zenodo-13956078]", msg)

    def test_parse_val_bpb_delegates_refuse_invent(self):
        import agent_loop

        self.assertIsNone(agent_loop.parse_val_bpb("no metric here"))
        self.assertIsNone(agent_loop.parse_val_bpb("val_bpb:          pending"))
        self.assertAlmostEqual(
            agent_loop.parse_val_bpb("val_bpb:          0.430912"),
            0.430912,
            places=6,
        )


class TestHonestyNoInventedFloorInModule(unittest.TestCase):
    def test_helper_source_refuses_invent_wording(self):
        src = (ROOT / "best_val_bpb.py").read_text(encoding="utf-8")
        self.assertIn("Never invent", src)
        self.assertNotRegex(src.lower(), r"auroc\s*=\s*0\.")
        # Must not hard-code a CRISP overnight floor as a default return
        self.assertNotRegex(src, r"return\s+0\.4309")
        self.assertNotRegex(src, r"return\s+0\.458756")
        self.assertNotRegex(src, r"return\s+0\.3682")


if __name__ == "__main__":
    unittest.main()
