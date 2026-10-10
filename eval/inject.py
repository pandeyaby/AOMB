"""
Label-free detection objective: can the model tell a corrupted session from a clean one?

Real incident labels are scarce, and val_bpb only says how well a model predicts
normal traffic. This builds a detection test from normal sessions alone: each
clean session gets corrupted copies, one per corruption type, and the model is
scored on ranking corrupted copies above clean sessions (AUROC).

Corruptions, each mimicking a class of real fault:

- ``value_swap``   a categorical field takes another session's value for that field
- ``value_cross``  a categorical field takes a value from a different field (never seen there)
- ``drop_line``    one event is removed (a missing call)
- ``dup_line``     one event is repeated (a retry)
- ``truncate``     the session stops early
- ``reorder``      two adjacent events swap
- ``latency``      one duration_ms is multiplied by 10–100

IDs, timestamps and counters are never touched (they are masked from scoring
anyway). Everything is deterministic given the seed. A corruption that cannot
apply to a session (e.g. ``truncate`` on a one-line session) is skipped for it.

This is an objective to *optimise*, not evidence of real detection: whether it
tracks AUROC on real faults is measured separately (docs/lab/injection-objective.md).
"""

from __future__ import annotations

import random
import re
from typing import Callable, Sequence

from eval.in_domain import _FIELD, _NUMERIC

CORRUPTIONS = ("value_swap", "value_cross", "drop_line", "dup_line", "truncate", "reorder", "latency")
_SKIP_KEYS = {"id", "parent", "hits", "ts"}  # identifiers / counters: masked from scoring
_DURATION = re.compile(r"duration_ms=(\d+)")


def _fields(line: str) -> list[tuple[str, str, int, int]]:
    """Categorical (key, value, value_start, value_end) on a line."""
    out = []
    for m in _FIELD.finditer(line):
        k, v = m.group(1), m.group(2)
        if k in _SKIP_KEYS or _NUMERIC.match(v):
            continue
        out.append((k, v, m.start(2), m.end(2)))
    return out


class Injector:
    """Fits a value pool on reference sessions, then corrupts sessions deterministically."""

    def __init__(self, reference: Sequence[str], seed: int = 0):
        self.seed = seed
        pool: dict[str, set[str]] = {}
        for text in reference:
            for line in text.split("\n"):
                for k, v, _a, _b in _fields(line):
                    pool.setdefault(k, set()).add(v)
        # keys with many distinct values behave like IDs; leave them alone
        self.pool = {k: sorted(vs) for k, vs in pool.items() if 1 < len(vs) <= 20}
        self.all_values = sorted({v for vs in self.pool.values() for v in vs})

    # --- individual corruptions: return the new text, or None if not applicable ---

    def _replace_value(self, text: str, rng: random.Random, choose: Callable) -> str | None:
        lines = text.split("\n")
        spots = [
            (i, k, v, a, b)
            for i, line in enumerate(lines)
            for (k, v, a, b) in _fields(line)
            if k in self.pool
        ]
        rng.shuffle(spots)
        for i, k, v, a, b in spots:
            new = choose(k, v, rng)
            if new is not None:
                lines[i] = lines[i][:a] + new + lines[i][b:]
                return "\n".join(lines)
        return None

    def value_swap(self, text: str, rng: random.Random) -> str | None:
        def choose(k, v, r):
            options = [x for x in self.pool[k] if x != v]
            return r.choice(options) if options else None

        return self._replace_value(text, rng, choose)

    def value_cross(self, text: str, rng: random.Random) -> str | None:
        def choose(k, v, r):
            options = [x for x in self.all_values if x not in self.pool[k]]
            return r.choice(options) if options else None

        return self._replace_value(text, rng, choose)

    def drop_line(self, text: str, rng: random.Random) -> str | None:
        lines = text.split("\n")
        if len(lines) < 2:
            return None
        del lines[rng.randrange(len(lines))]
        return "\n".join(lines)

    def dup_line(self, text: str, rng: random.Random) -> str | None:
        lines = text.split("\n")
        i = rng.randrange(len(lines))
        return "\n".join(lines[: i + 1] + [lines[i]] + lines[i + 1 :])

    def truncate(self, text: str, rng: random.Random) -> str | None:
        lines = text.split("\n")
        if len(lines) < 2:
            return None
        return "\n".join(lines[: rng.randrange(1, len(lines))])

    def reorder(self, text: str, rng: random.Random) -> str | None:
        lines = text.split("\n")
        spots = [i for i in range(len(lines) - 1) if lines[i] != lines[i + 1]]
        if not spots:
            return None
        i = rng.choice(spots)
        lines[i], lines[i + 1] = lines[i + 1], lines[i]
        return "\n".join(lines)

    def latency(self, text: str, rng: random.Random) -> str | None:
        hits = list(_DURATION.finditer(text))
        if not hits:
            return None
        m = rng.choice(hits)
        new = max(int(m.group(1)), 1) * rng.randint(10, 100)
        return text[: m.start(1)] + str(new) + text[m.end(1) :]

    def corrupt(self, texts: Sequence[str]) -> dict[str, list[tuple[int, str]]]:
        """{corruption: [(index of the clean session, corrupted text), ...]}"""
        out: dict[str, list[tuple[int, str]]] = {c: [] for c in CORRUPTIONS}
        for ci, name in enumerate(CORRUPTIONS):
            fn = getattr(self, name)
            for i, text in enumerate(texts):
                rng = random.Random(f"{self.seed}:{ci}:{i}")
                new = fn(text, rng)
                if new is not None and new != text:
                    out[name].append((i, new))
        return out


def injection_auroc(clean_scores: Sequence[float], corrupted_scores: Sequence[float]) -> float:
    """AUROC for ranking corrupted copies above clean sessions."""
    from eval.metrics import auroc

    y = [0] * len(clean_scores) + [1] * len(corrupted_scores)
    return auroc(y, list(clean_scores) + list(corrupted_scores))
