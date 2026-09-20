"""Resolve agent_loop best-val floor without inventing numbers.

Product overnight previously needed ``AOMB_BEST_VAL_BPB`` so a synthetic
smoke-era best (e.g. 0.3682) did not poison CRISP / Tale breeding. This
module hardens that in code:

1. Env override ``AOMB_BEST_VAL_BPB`` still wins when set and sane.
2. When ``AOMB_CORPUS`` / ``AOMB_SOURCE_ID`` (or explicit args) are present,
   only commit subjects tagged with a matching ``[corpus=…]`` /
   ``[source_id=…]`` count toward the lane best.
3. Missing / malformed / empty lane → ``float("inf")`` (start fresh).
   Never invent a CRISP/Tale/synthetic floor.

Does not touch ``prepare.py``. Does not invent AUROC.
"""

from __future__ import annotations

import math
import os
import re
from typing import Mapping

from val_bpb_parse import parse_val_bpb_from_commit_message

# Short aliases operators may set via AOMB_CORPUS → canonical lane keys.
_CORPUS_ALIASES: dict[str, str] = {
    "crisp": "uber-crisp-zenodo-13956078",
    "uber-crisp": "uber-crisp-zenodo-13956078",
    "uber-crisp-zenodo-13956078": "uber-crisp-zenodo-13956078",
    "tale": "uber-tale-of-errors",
    "tale-of-errors": "uber-tale-of-errors",
    "uber-tale-of-errors": "uber-tale-of-errors",
    "synthetic": "synthetic-smoke",
    "smoke": "synthetic-smoke",
    "synthetic-smoke": "synthetic-smoke",
}

_CORPUS_TAG_RE = re.compile(r"\[corpus=([^\]]+)\]", re.IGNORECASE)
_SOURCE_TAG_RE = re.compile(r"\[source_id=([^\]]+)\]", re.IGNORECASE)

# Reject nonsense override tokens the same way val_bpb_parse refuses invent.
_MAX_SANE_VAL_BPB = 50.0


def normalize_lane_key(raw: str | None) -> str | None:
    """Map operator short names / source ids to a canonical lane key."""
    if raw is None:
        return None
    key = raw.strip().lower()
    if not key:
        return None
    return _CORPUS_ALIASES.get(key, key)


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


def resolve_best_val_bpb(
    subjects: list[str] | tuple[str, ...] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    corpus: str | None = None,
    source_id: str | None = None,
    override: str | float | None = ...,  # type: ignore[assignment]
) -> float:
    """Resolve best-val for agent_loop.

    Priority:
      1. Explicit / env override ``AOMB_BEST_VAL_BPB`` (when sane)
      2. Min of in-lane commit subjects
      3. ``float("inf")`` — never invent a floor
    """
    env = os.environ if environ is None else environ

    if override is ...:
        override_raw: str | float | None = env.get("AOMB_BEST_VAL_BPB")
    else:
        override_raw = override

    if isinstance(override_raw, (int, float)) and not isinstance(override_raw, bool):
        ov = float(override_raw)
        if math.isfinite(ov) and 0.0 <= ov <= _MAX_SANE_VAL_BPB:
            return ov
        # Nonsense numeric override → refuse invent (fall through)
    else:
        ov = parse_best_val_override(
            None if override_raw is None else str(override_raw)
        )
        if ov is not None:
            return ov

    if corpus is None and source_id is None:
        corpus, source_id = resolve_lane_from_environ(env)

    return best_val_from_commit_subjects(
        list(subjects or ()),
        corpus=corpus,
        source_id=source_id,
    )


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
