"""Resolve agent_loop best-val floor without inventing numbers.

Product overnight previously needed ``AOMB_BEST_VAL_BPB`` so a synthetic
smoke-era best (e.g. 0.3682) did not poison CRISP / Tale breeding. This
module hardens that in code.

Precedence (first sane wins)::

  1. Env / explicit override ``AOMB_BEST_VAL_BPB``
  2. Tale measured card (when enabled + cache_lane active data/tokenizer) —
     factual ``val_bpb`` only; missing/ambiguous cache → treat card as unavailable
  3. Min of in-lane git/log commit subjects
  4. ``float("inf")`` — never invent a floor

Card enablement::

  - ``AOMB_CORPUS`` / ``AOMB_SOURCE_ID`` is a Tale lane alias
    (``tale``, ``tale_of_errors``, ``uber-tale-of-errors``, …), **or**
  - ``AOMB_BEST_VAL_FROM_CARD=1`` (or true/yes/on)

Card acceptance::

  - File present and JSON-parseable
  - ``claim_status == "measured_not_published"``
    (``pending`` / other / missing → treat as missing → no card floor)
  - ``val_bpb`` is a finite number in ``[0, MAX_SANE]``

Missing / malformed / null / NaN card → skip card (fall through). Never invent.

Does not touch ``prepare.py``. Does not invent AUROC.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from val_bpb_parse import parse_val_bpb_from_commit_message

ROOT = Path(__file__).resolve().parent
DEFAULT_TALE_MEASURED_CARD = (
    ROOT / "reports" / "tale-capped" / "measured_capped_200k.json"
)

# Short aliases operators may set via AOMB_CORPUS → canonical lane keys.
_CORPUS_ALIASES: dict[str, str] = {
    "crisp": "uber-crisp-zenodo-13956078",
    "uber-crisp": "uber-crisp-zenodo-13956078",
    "uber-crisp-zenodo-13956078": "uber-crisp-zenodo-13956078",
    "tale": "uber-tale-of-errors",
    "tale-of-errors": "uber-tale-of-errors",
    "tale_of_errors": "uber-tale-of-errors",
    "uber-tale-of-errors": "uber-tale-of-errors",
    "synthetic": "synthetic-smoke",
    "smoke": "synthetic-smoke",
    "synthetic-smoke": "synthetic-smoke",
}

_TALE_LANE = "uber-tale-of-errors"

_CORPUS_TAG_RE = re.compile(r"\[corpus=([^\]]+)\]", re.IGNORECASE)
_SOURCE_TAG_RE = re.compile(r"\[source_id=([^\]]+)\]", re.IGNORECASE)

# Reject nonsense override tokens the same way val_bpb_parse refuses invent.
_MAX_SANE_VAL_BPB = 50.0

ALLOWED_CARD_CLAIM_STATUS = frozenset({"measured_not_published"})

# Invent / publish / CUDA — refuse on CLI (mirrors stranger / product_mac).
REFUSED_METRIC_FLAGS = frozenset(
    {
        "--auroc",
        "--lab-auroc",
        "--accuracy",
        "--ranking",
        "--publish",
        "--claim",
        "--invent-metrics",
        "--invent-auroc",
        "--claim-auroc",
        "--val-bpb",
        "--invent-val-bpb",
        "--readme-hero",
        "--publish-readme",
        "--hero-auroc",
        "--cuda",
        "--gpu",
    }
)

EXIT_OK = 0
EXIT_REFUSED_FLAG = 1


def normalize_lane_key(raw: str | None) -> str | None:
    """Map operator short names / source ids to a canonical lane key."""
    if raw is None:
        return None
    key = raw.strip().lower()
    if not key:
        return None
    return _CORPUS_ALIASES.get(key, key)


def is_tale_lane(*, corpus: str | None = None, source_id: str | None = None) -> bool:
    """True when corpus/source_id normalizes to the Tale lane."""
    return (
        normalize_lane_key(corpus) == _TALE_LANE
        or normalize_lane_key(source_id) == _TALE_LANE
    )


def parse_best_val_override(raw: str | None) -> float | None:
    """Parse ``AOMB_BEST_VAL_BPB``. None = unset / refuse invent (no floor)."""
    if raw is None:
        return None
    token = raw.strip()
    if not token:
        return None
    if token.lower() in {
        "nan",
        "inf",
        "-inf",
        "+inf",
        "none",
        "null",
        "n/a",
        "na",
        "pending",
        "auto",
    }:
        return None
    try:
        val = float(token)
    except ValueError:
        return None
    if not math.isfinite(val):
        return None
    if val < 0.0 or val > _MAX_SANE_VAL_BPB:
        return None
    return val


def _truthy_env(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def resolve_autoresearch_cache_root(
    environ: Mapping[str, str] | None = None,
    *,
    cache_root: Path | str | None = None,
    home: Path | str | None = None,
) -> Path:
    """Resolve ``~/.cache/autoresearch`` (tests may pass cache_root / HOME)."""
    if cache_root is not None:
        return Path(cache_root).expanduser().resolve()
    env = os.environ if environ is None else environ
    raw = (env.get("AOMB_CACHE_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    home_path: Path | None
    if home is not None:
        home_path = Path(home).expanduser()
    else:
        home_raw = (env.get("AOMB_CACHE_HOME") or env.get("HOME") or "").strip()
        home_path = Path(home_raw).expanduser() if home_raw else None
    try:
        from corpus.ingest.cache_lane import default_cache_root

        return default_cache_root(home_path)
    except Exception:
        base = home_path if home_path is not None else Path.home()
        return (base / ".cache" / "autoresearch").resolve()


def cache_lane_ready_for_card_floor(
    environ: Mapping[str, str] | None = None,
    *,
    cache_root: Path | str | None = None,
    home: Path | str | None = None,
) -> tuple[bool, str]:
    """True when active autoresearch data+tokenizer dirs exist.

    Import-safe: uses ``corpus.ingest.cache_lane.status`` when available.
    Missing / partial / ambiguous lane → False (never invent a card floor).
    """
    root = resolve_autoresearch_cache_root(
        environ, cache_root=cache_root, home=home
    )
    try:
        from corpus.ingest.cache_lane import status
    except Exception as exc:  # noqa: BLE001 — import-safe refuse
        return False, f"cache_lane import unavailable ({exc}); refuse card floor"

    try:
        rep = status(root)
    except Exception as exc:  # noqa: BLE001
        return False, f"cache_lane status failed ({exc}); refuse card floor"

    if rep.active_data and rep.active_tokenizer:
        return True, f"active data+tokenizer under {root}"
    if rep.active_data ^ rep.active_tokenizer:
        return (
            False,
            "ambiguous/partial cache_lane "
            f"(data={'yes' if rep.active_data else 'no'}, "
            f"tokenizer={'yes' if rep.active_tokenizer else 'no'}) under {root}",
        )
    return False, f"no active data dir under {root}"



def card_floor_enabled(
    environ: Mapping[str, str] | None = None,
    *,
    corpus: str | None = None,
    source_id: str | None = None,
) -> bool:
    """Whether to attempt reading the Tale measured card as a floor."""
    env = os.environ if environ is None else environ
    if _truthy_env(env.get("AOMB_BEST_VAL_FROM_CARD")):
        return True
    c = corpus
    s = source_id
    if c is None and s is None:
        c, s = resolve_lane_from_environ(env)
    return is_tale_lane(corpus=c, source_id=s)


def parse_val_bpb_from_measured_card(
    path: Path | str | None,
) -> float | None:
    """Read factual ``val_bpb`` from a Tale measured card, or None.

    Never invents. Returns None when missing / malformed / wrong claim_status /
    null / NaN / non-finite / out of sane range.
    """
    if path is None:
        return None
    card_path = Path(path)
    if not card_path.is_file():
        return None
    try:
        raw = card_path.read_text(encoding="utf-8")
        data: Any = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None

    claim = data.get("claim_status")
    if claim in (None, "pending"):
        return None
    if not isinstance(claim, str) or claim not in ALLOWED_CARD_CLAIM_STATUS:
        return None

    val_raw = data.get("val_bpb")
    if val_raw is None:
        return None
    if isinstance(val_raw, bool):
        return None
    if isinstance(val_raw, str):
        return parse_best_val_override(val_raw)
    try:
        val = float(val_raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    if val < 0.0 or val > _MAX_SANE_VAL_BPB:
        return None
    return val


def resolve_measured_card_path(
    environ: Mapping[str, str] | None = None,
    *,
    card_path: Path | str | None = None,
) -> Path:
    """Explicit path / ``AOMB_BEST_VAL_CARD`` / default Tale measured card."""
    if card_path is not None:
        return Path(card_path)
    env = os.environ if environ is None else environ
    raw = (env.get("AOMB_BEST_VAL_CARD") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return DEFAULT_TALE_MEASURED_CARD


def parse_commit_lane_tags(msg: str) -> tuple[str | None, str | None]:
    """Extract optional ``[corpus=…]`` / ``[source_id=…]`` from a commit subject."""
    if not msg:
        return None, None
    corpus_m = _CORPUS_TAG_RE.search(msg)
    source_m = _SOURCE_TAG_RE.search(msg)
    corpus = normalize_lane_key(corpus_m.group(1) if corpus_m else None)
    source = normalize_lane_key(source_m.group(1) if source_m else None)
    return corpus, source


def commit_matches_lane(
    msg: str,
    *,
    corpus: str | None = None,
    source_id: str | None = None,
) -> bool:
    """True if commit belongs to the requested lane.

    - No filter → all commits with a factual ``val_bpb=`` match (legacy).
    - With filter → require a matching ``[corpus=…]`` or ``[source_id=…]`` tag.
      Untagged legacy subjects are excluded so synthetic history cannot poison
      a CRISP/Tale overnight.
    """
    want_corpus = normalize_lane_key(corpus)
    want_source = normalize_lane_key(source_id)
    if want_corpus is None and want_source is None:
        return True

    got_corpus, got_source = parse_commit_lane_tags(msg)
    if want_corpus is not None:
        if got_corpus == want_corpus or got_source == want_corpus:
            return True
    if want_source is not None:
        if got_source == want_source or got_corpus == want_source:
            return True
    return False


def best_val_from_commit_subjects(
    subjects: list[str] | tuple[str, ...],
    *,
    corpus: str | None = None,
    source_id: str | None = None,
) -> float:
    """Lowest factual ``val_bpb=`` among subjects in-lane; else ``inf``.

    Never invents a substitute floor when the lane is empty.
    """
    bpbs: list[float] = []
    for msg in subjects:
        if not commit_matches_lane(msg, corpus=corpus, source_id=source_id):
            continue
        val = parse_val_bpb_from_commit_message(msg)
        if val is not None:
            bpbs.append(val)
    return min(bpbs) if bpbs else float("inf")


def resolve_lane_from_environ(
    environ: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Read ``AOMB_CORPUS`` / ``AOMB_SOURCE_ID`` (empty → unset)."""
    env = os.environ if environ is None else environ
    corpus = normalize_lane_key(env.get("AOMB_CORPUS"))
    source_id = normalize_lane_key(env.get("AOMB_SOURCE_ID"))
    return corpus, source_id


# Source tags for resolve_best_val_bpb_with_source (never invent a label).
SOURCE_ENV = "env"
SOURCE_MEASURED_CARD = "measured_card"
SOURCE_GIT = "git"
SOURCE_MISSING = "missing"

ALLOWED_BEST_VAL_SOURCES = frozenset(
    {SOURCE_ENV, SOURCE_MEASURED_CARD, SOURCE_GIT, SOURCE_MISSING}
)


def resolve_best_val_bpb_with_source(
    subjects: list[str] | tuple[str, ...] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    corpus: str | None = None,
    source_id: str | None = None,
    override: str | float | None = ...,  # type: ignore[assignment]
    card_path: Path | str | None = None,
    cache_root: Path | str | None = None,
    home: Path | str | None = None,
) -> tuple[float, str]:
    """Resolve best-val and the factual source tag.

    Returns ``(value, source)`` where ``source`` is one of::

      - ``env`` — sane ``AOMB_BEST_VAL_BPB`` / explicit override
      - ``measured_card`` — factual card floor + active cache_lane
      - ``git`` — min in-lane commit subject ``val_bpb=``
      - ``missing`` — nothing factual → ``float("inf")``

    Never invents a number or a source tag. Measured card without an active
    cache_lane is treated as missing card (fall through to git / missing).
    """
    env = os.environ if environ is None else environ

    if override is ...:
        override_raw: str | float | None = env.get("AOMB_BEST_VAL_BPB")
    else:
        override_raw = override

    if isinstance(override_raw, (int, float)) and not isinstance(override_raw, bool):
        ov = float(override_raw)
        if math.isfinite(ov) and 0.0 <= ov <= _MAX_SANE_VAL_BPB:
            return ov, SOURCE_ENV
        # Nonsense numeric override → refuse invent (fall through)
    else:
        ov = parse_best_val_override(
            None if override_raw is None else str(override_raw)
        )
        if ov is not None:
            return ov, SOURCE_ENV

    if corpus is None and source_id is None:
        corpus, source_id = resolve_lane_from_environ(env)

    if card_floor_enabled(env, corpus=corpus, source_id=source_id):
        path = resolve_measured_card_path(env, card_path=card_path)
        card_val = parse_val_bpb_from_measured_card(path)
        if card_val is not None:
            ready, lane_msg = cache_lane_ready_for_card_floor(
                env, cache_root=cache_root, home=home
            )
            if ready:
                return card_val, SOURCE_MEASURED_CARD
            # Card present but cache lane missing/ambiguous → same as missing card.
            print(
                f"best_val_bpb: refusing measured-card floor ({lane_msg}); "
                "never invent val_bpb — falling through to git / inf",
                file=sys.stderr,
            )
        # missing/malformed card or refused lane → fall through (may still be inf)

    git_best = best_val_from_commit_subjects(
        list(subjects or ()),
        corpus=corpus,
        source_id=source_id,
    )
    if math.isfinite(git_best):
        return git_best, SOURCE_GIT
    return float("inf"), SOURCE_MISSING


def resolve_best_val_bpb(
    subjects: list[str] | tuple[str, ...] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    corpus: str | None = None,
    source_id: str | None = None,
    override: str | float | None = ...,  # type: ignore[assignment]
    card_path: Path | str | None = None,
    cache_root: Path | str | None = None,
    home: Path | str | None = None,
) -> float:
    """Resolve best-val for agent_loop.

    Priority:
      1. Explicit / env override ``AOMB_BEST_VAL_BPB`` (when sane)
      2. Tale measured card (when enabled + factual + cache_lane active)
      3. Min of in-lane commit subjects
      4. ``float("inf")`` — never invent a floor
    """
    value, _source = resolve_best_val_bpb_with_source(
        subjects,
        environ=environ,
        corpus=corpus,
        source_id=source_id,
        override=override,
        card_path=card_path,
        cache_root=cache_root,
        home=home,
    )
    return value


def format_best_val_display(value: float, source: str) -> str:
    """Two-line dry-run display: resolved value + source tag (never invent)."""
    if source not in ALLOWED_BEST_VAL_SOURCES:
        source = SOURCE_MISSING
    if math.isfinite(value):
        val_s = f"{value}"
    else:
        val_s = "inf"
        source = SOURCE_MISSING
    return f"  best_val: {val_s}\n  best_val_source: {source}"


def format_lane_commit_tags(
    *,
    corpus: str | None = None,
    source_id: str | None = None,
) -> str:
    """Build ``[corpus=…] [source_id=…]`` fragment for experiment commits."""
    parts: list[str] = []
    c = normalize_lane_key(corpus)
    s = normalize_lane_key(source_id)
    if c:
        parts.append(f"[corpus={c}]")
    if s and s != c:
        parts.append(f"[source_id={s}]")
    elif s and not c:
        parts.append(f"[source_id={s}]")
    return " ".join(parts)


def refuse_loud_flags(argv: list[str]) -> str | None:
    for arg in argv:
        key = arg.split("=", 1)[0]
        if key in REFUSED_METRIC_FLAGS:
            return key
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI: print resolved best-val (debug). No invent / publish / CUDA flags."""
    argv = list(sys.argv[1:] if argv is None else argv)
    bad = refuse_loud_flags(argv)
    if bad is not None:
        print(
            f"ERROR: refusing invent / publish / CUDA flag {bad}. "
            "best_val_bpb never invents AUROC / floors.",
            file=sys.stderr,
        )
        return EXIT_REFUSED_FLAG

    p = argparse.ArgumentParser(
        prog="best_val_bpb",
        description=(
            "Resolve AOMB best-val floor (override > Tale measured card > "
            "git subjects > inf). Never invents AUROC / val_bpb."
        ),
    )
    p.add_argument(
        "--card",
        type=Path,
        default=None,
        help="Override measured card path (default: reports/tale-capped/...)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print {best_val_bpb, card_enabled} as JSON",
    )
    args = p.parse_args(argv)

    env = dict(os.environ)
    best = resolve_best_val_bpb((), environ=env, card_path=args.card)
    enabled = card_floor_enabled(env)
    if args.json:
        payload = {
            "best_val_bpb": best if math.isfinite(best) else None,
            "best_val_bpb_inf": not math.isfinite(best),
            "card_enabled": enabled,
        }
        print(json.dumps(payload, indent=2))
    else:
        if math.isfinite(best):
            print(f"best_val_bpb={best}")
        else:
            print("best_val_bpb=inf")
        print(f"card_enabled={int(enabled)}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
