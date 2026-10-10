"""
In-domain lab eval: learn *this* environment's normal, then rank held-out sessions.

This is the experiment that matches the product pitch ("a model trained on your
own telemetry"). The zero-shot run in docs/lab/ranking-validation.md trains on
Uber CRISP and scores a lab it has never seen; here the model sees only the
lab's own **normal** traffic.

Split (per source capture, temporal): the earlier half of each capture's normal
sessions trains the tokenizer + LM and fits baseline statistics; the later half
of normals plus every incident session is the eval set. No incident text is
ever used for training or fitting.

Methods compared on the identical eval set:

- ``length``          session characters
- ``error_lines``     count of lines with status=error / level=ERROR
- ``duration_z``      max per-(op, svc) z-score of log duration_ms, fit on train normals
- ``rule``            error_lines, tie-broken by duration_z (a sensible SRE alert rule)
- ``novelty``         unseen (op, svc) + unseen log templates (values masked, Drain-style)
                      + unseen trace shape (multiset of spans), vs train normals
- ``heuristic``       rule + novelty: the strongest detector an SRE could hand-build
- ``value_novelty``   categorical field values never seen in train normals
- ``value_rarity``    per-line naive-Bayes surprisal: sum of -log2 p(value | field)
                      over a line's categorical fields, max over lines
- ``value_pair``      field-value *pairs* on one line never seen together in training
                      (the check you'd only write if you anticipated the fault)
- ``sequence_novelty`` session's multiset of events (ops + log templates) never seen in
                      training (the classic event-count-vector check on log benchmarks)

Inputs: a lab capture dir (``--capture``, temporal split per capture) or a prepared
session file (``--sessions``, JSONL with session_id / label / split / group / text,
e.g. from corpus.ingest.loghub_hdfs), whose ``split`` field defines train vs eval.
- ``bpb_mean``        session bits-per-byte (the current AOMB score)
- ``bpb_content``     bits-per-byte excluding IDs, timestamps and monotonic counters
                      (random hex is incompressible noise; clock-like values are confounds)
- ``bpb_max_event``   highest per-event (per-line) bits-per-byte, IDs/timestamps excluded
- ``bpb_top10``       mean bits of the 10% most surprising tokens, IDs/timestamps excluded
- ``bits_max_field``  most surprising single field value: total bits of the tokens in
                      one ``key=value`` value (≈ -log2 p(value | context)), max over fields
- ``bits_end``        with ``--end-marker``: bits of the end-of-session marker, i.e. how
                      unexpected it is that the session stopped here (0 without the flag)

Usage:
    uv run python -m eval.in_domain --capture lab/captures/pooled-20260918 \\
        --seeds 0..4 --train-seconds 120 --out-dir reports/public-accuracy/<run>
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from eval.lab_breakdown import session_capture_ids
from eval.labels import LabeledSession, filter_scorable, load_lab_sessions
from eval.metrics import auroc, mean_std, pr_auc, precision_at_k
from eval.run_multiseed import parse_seeds

ROOT = Path(__file__).resolve().parents[1]

_TS = re.compile(r"\[ts=([0-9T:\-\.]+)Z\]")
_EVENT = re.compile(r"op=(\S+) svc=(\S+) duration_ms=(\d+(?:\.\d+)?)")
# Severity words a simple alert rule would key on (BGL uses FATAL / SEVERE / FAILURE).
_ERROR = re.compile(r"status=error|level=(?:ERROR|FATAL|SEVERE|FAILURE|CRITICAL)\b")
# key=value fields preceded by start/space/underscore (log msgs join words with "_")
_FIELD = re.compile(r"(?:^|(?<=[\s_]))([A-Za-z]+)=([^\s_\]]+)")
_NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")
MAX_CATEGORICAL = 20  # keys with more distinct train values are treated as IDs/counters
_LOG = re.compile(r"\[src=OTelLog\] level=(\S+) svc=(\S+) msg=(\S+)")
# Values that carry no information about system health: random hex IDs, the
# wall-clock timestamp, and monotonic counters (the lab's Redis ``hits``). Eval
# windows come later than training, so clock times and counter values in them
# are always "new" — scoring them would reward unfamiliar time, not anomalies.
_NOISE_VALUE = re.compile(
    r"(?:trace_id|span_id|parent)=(\S+)|\[ts=([^\]]+)\]|(?:^|(?<=[\s_]))hits=(\d+)"
)

MODEL_METHODS = ("bpb_mean", "bpb_content", "bpb_max_event", "bpb_top10", "bits_max_field", "bits_end")
# Appended to every session when --end-marker is on, so the model can be surprised
# by a session that stops early (surprise only scores what *did* happen otherwise).
END_MARKER = "[end_of_session]"
BASELINE_METHODS = (
    "length", "error_lines", "duration_z", "rule", "novelty", "heuristic",
    "value_novelty", "value_rarity", "value_pair", "sequence_novelty",
)


# ---------------------------------------------------------------- split


def temporal_split(
    sessions: Sequence[LabeledSession], capture_of: dict[str, str], train_frac: float = 0.5
) -> tuple[list[LabeledSession], list[LabeledSession]]:
    """Earlier ``train_frac`` of each capture's normals → train; rest + incidents → eval."""
    normals: dict[str, list[LabeledSession]] = defaultdict(list)
    eval_set: list[LabeledSession] = []
    for s in sessions:
        if s.binary == 0:
            normals[capture_of.get(s.session_id, "")].append(s)
        else:
            eval_set.append(s)

    def first_ts(s: LabeledSession) -> str:
        m = _TS.search(s.text)
        return m.group(1) if m else ""

    train: list[LabeledSession] = []
    for cid in sorted(normals):
        xs = sorted(normals[cid], key=first_ts)
        cut = int(len(xs) * train_frac)
        train.extend(xs[:cut])
        eval_set.extend(xs[cut:])
    return train, eval_set


# ---------------------------------------------------------------- session files


def load_session_file(path: str | Path) -> tuple[list[LabeledSession], list[LabeledSession], dict[str, str], dict]:
    """(train normals, eval sessions, group_of, meta) from a prepared JSONL session file."""
    from eval.labels import binary_from_label

    path = Path(path).expanduser()
    train, eval_set, group_of = [], [], {}
    in_subset: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        s = LabeledSession(
            session_id=r["session_id"],
            text=r["text"],
            label=r["label"],
            binary=binary_from_label(r["label"]),
            n_chars=len(r["text"]),
        )
        if s.binary is None:
            continue
        group_of[s.session_id] = r.get("group", "")
        if r.get("touches_root_cause"):
            in_subset.add(s.session_id)
        if r["split"] == "train":
            if s.binary != 0:
                raise ValueError(f"training session {s.session_id} is not normal")
            train.append(s)
        else:
            eval_set.append(s)
    meta = {
        "capture_id": path.stem,
        "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "subset_field": "touches_root_cause" if in_subset else None,
        "subset_ids": in_subset,
    }
    return train, eval_set, group_of, meta


def chunk_lines(texts: Sequence[str], max_lines: int | None, suffix: str | None = None) -> list[str]:
    """
    Split long texts into ≤ max_lines pieces at line boundaries (no text dropped).

    ``suffix`` (the end-of-session marker) is added as a final line to each text,
    so it lands only on a session's last chunk.
    """
    out = []
    for t in texts:
        lines = t.split("\n") + ([suffix] if suffix else [])
        step = max_lines or len(lines)
        out.extend("\n".join(lines[i : i + step]) for i in range(0, len(lines), step))
    return out


# ---------------------------------------------------------------- baselines


def _log_dur(d: str) -> float:
    return math.log1p(float(d))


def fit_duration_stats(train: Sequence[LabeledSession]) -> dict[tuple[str, str], tuple[float, float]]:
    vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for s in train:
        for op, svc, d in _EVENT.findall(s.text):
            vals[(op, svc)].append(_log_dur(d))
    stats = {}
    for k, xs in vals.items():
        mu = sum(xs) / len(xs)
        var = sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)
        stats[k] = (mu, max(math.sqrt(var), 0.05))  # floor: avoid div-by-~0 on constant ops
    return stats


def log_template(level: str, svc: str, msg: str) -> str:
    """Drain-style template: mask every ``=value`` and all digits."""
    masked = re.sub(r"=[^_\s]+", "=<*>", msg)
    return f"{level} {svc} {re.sub(r'[0-9]+', '#', masked)}"


def session_features(text: str) -> tuple[set, set, tuple]:
    """(ops, log templates, trace shape) for one session."""
    ops = [(op, svc) for op, svc, _d in _EVENT.findall(text)]
    templates = {log_template(*m) for m in _LOG.findall(text)}
    shape = tuple(sorted(ops))
    return set(ops), templates, shape


def line_fields(line: str) -> list[tuple[str, str]]:
    """Non-numeric key=value fields on one line (IDs/timestamps filtered later by cardinality)."""
    return [(k, v) for k, v in _FIELD.findall(line) if not _NUMERIC.match(v)]


class ValueStats:
    """Per-field value frequencies and co-occurring pairs from train normals."""

    def __init__(self, train: Sequence[LabeledSession]):
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        per_line: list[list[tuple[str, str]]] = []
        for s in train:
            for line in s.text.split("\n"):
                fs = line_fields(line)
                per_line.append(fs)
                for k, v in fs:
                    counts[k][v] += 1
        self.counts = {
            k: dict(vs) for k, vs in counts.items() if len(vs) <= MAX_CATEGORICAL
        }
        self.pairs = set()
        for fs in per_line:
            cat = [(k, v) for k, v in fs if k in self.counts]
            self.pairs |= {(a, b) for i, a in enumerate(cat) for b in cat[i + 1 :]}

    def score(self, text: str) -> tuple[float, float, float]:
        novel, rarity, pair_novel = 0.0, 0.0, 0.0
        for line in text.split("\n"):
            cat = [(k, v) for k, v in line_fields(line) if k in self.counts]
            line_bits = 0.0
            for k, v in cat:
                c = self.counts[k]
                n = c.get(v, 0)
                novel += n == 0
                line_bits += -math.log2((n + 0.5) / (sum(c.values()) + 0.5 * len(c)))
            rarity = max(rarity, line_bits)
            pair_novel += sum(
                (a, b) not in self.pairs for i, a in enumerate(cat) for b in cat[i + 1 :]
            )
        return novel, rarity, pair_novel


def event_multiset(text: str) -> tuple:
    """Sorted multiset of a session's events: (op, svc) spans and log templates."""
    ops = [f"{op} {svc}" for op, svc, _d in _EVENT.findall(text)]
    logs = [log_template(*m) for m in _LOG.findall(text)]
    return tuple(sorted(ops + logs))


def fit_novelty(train: Sequence[LabeledSession]) -> tuple[set, set, set, set]:
    ops, templates, shapes, sequences = set(), set(), set(), set()
    for s in train:
        o, t, sh = session_features(s.text)
        ops |= o
        templates |= t
        shapes.add(sh)
        sequences.add(event_multiset(s.text))
    return ops, templates, shapes, sequences


def baseline_scores(
    sessions: Sequence[LabeledSession],
    stats: dict[tuple[str, str], tuple[float, float]],
    seen: tuple[set, ...] = (set(), set(), set(), set()),
    values: "ValueStats | None" = None,
) -> dict[str, list[float]]:
    seen_ops, seen_templates, seen_shapes = seen[:3]
    seen_sequences = seen[3] if len(seen) > 3 else set()
    out: dict[str, list[float]] = {m: [] for m in BASELINE_METHODS}
    for s in sessions:
        errors = float(len(_ERROR.findall(s.text)))
        zs = [
            (_log_dur(d) - stats[(op, svc)][0]) / stats[(op, svc)][1]
            for op, svc, d in _EVENT.findall(s.text)
            if (op, svc) in stats
        ]
        z = max(zs) if zs else 0.0
        out["length"].append(float(s.n_chars))
        out["error_lines"].append(errors)
        out["duration_z"].append(z)
        out["rule"].append(errors * 1000.0 + z)
        o, t, sh = session_features(s.text)
        novelty = float(len(o - seen_ops) + len(t - seen_templates) + (sh not in seen_shapes))
        out["novelty"].append(novelty)
        out["heuristic"].append(errors * 1000.0 + novelty * 100.0 + z)
        vn, vr, vp = values.score(s.text) if values is not None else (0.0, 0.0, 0.0)
        out["value_novelty"].append(vn)
        out["value_rarity"].append(vr)
        out["value_pair"].append(vp)
        out["sequence_novelty"].append(float(event_multiset(s.text) not in seen_sequences))
    return out


# ---------------------------------------------------------------- model


def token_surprise(model, tokenizer, token_bytes, text: str, max_seq_len: int) -> list[tuple[str, float, int]]:
    """Per-token (string, nats, n_bytes) for ``text`` under ``model`` (BOS excluded)."""
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    ids = tokenizer.encode(text, prepend=tokenizer.get_bos_token_id())
    out: list[tuple[str, float, int]] = []
    pos = 0
    with torch.no_grad():
        while pos + 1 < len(ids):
            chunk = ids[pos : pos + max_seq_len + 1]
            if len(chunk) < 2:
                break
            x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
            y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)
            nats = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), y.view(-1), reduction="none"
            )
            nb = token_bytes[y.view(-1)]
            for tid, n, b in zip(chunk[1:], nats.tolist(), nb.tolist()):
                out.append((tokenizer.decode([tid]), float(n), int(b)))
            pos += max_seq_len
    return out


def _noise_mask(text: str, toks: Sequence[tuple[str, float, int]]) -> list[bool]:
    """True where a token starts inside an ID value or a timestamp."""
    spans = [
        m.span(next(g for g in (1, 2, 3) if m.group(g) is not None))
        for m in _NOISE_VALUE.finditer(text)
    ]
    mask, off, j = [], 0, 0
    for tok, _n, _b in toks:
        while j < len(spans) and spans[j][1] <= off:
            j += 1
        mask.append(j < len(spans) and spans[j][0] <= off < spans[j][1])
        off += len(tok)
    return mask


def model_scores_for(toks: Sequence[tuple[str, float, int]], text: str) -> dict[str, float]:
    ln2 = math.log(2)
    is_id = _noise_mask(text, toks)
    tot_n = sum(n for _t, n, b in toks if b > 0)
    tot_b = sum(b for _t, _n, b in toks if b > 0)
    kept = [(t, n, b) for (t, n, b), m in zip(toks, is_id) if b > 0 and not m]
    kn = sum(n for _t, n, _b in kept)
    kb = sum(b for _t, _n, b in kept)

    # per-line (event) bpb, IDs excluded
    line_n: dict[int, float] = defaultdict(float)
    line_b: dict[int, int] = defaultdict(int)
    line = 0
    for (tok, n, b), m in zip(toks, is_id):
        if b > 0 and not m:
            line_n[line] += n
            line_b[line] += b
        line += tok.count("\n")
    ev = [line_n[k] / (ln2 * line_b[k]) for k in line_n if line_b[k] >= 8]

    bits = sorted((n / ln2 for _t, n, _b in kept), reverse=True)
    k = max(1, len(bits) // 10)
    nan = float("nan")

    # per-field: bits of the tokens starting inside each key=value value span
    starts, off = [], 0
    for tok, _n, _b in toks:
        starts.append(off)
        off += len(tok)
    bits_end = 0.0
    if text.endswith(END_MARKER):
        end_at = len(text) - len(END_MARKER)
        bits_end = sum(n / ln2 for (_t, n, _b), st in zip(toks, starts) if st >= end_at)
    field_bits = []
    for m in _FIELD.finditer(text):
        a, b = m.span(2)
        fb = sum(
            n / ln2
            for (tok, n, nb), st, noise in zip(toks, starts, is_id)
            if a <= st < b and nb > 0 and not noise
        )
        if fb > 0:
            field_bits.append(fb)
    return {
        "bpb_mean": tot_n / (ln2 * tot_b) if tot_b else nan,
        "bpb_content": kn / (ln2 * kb) if kb else nan,
        "bpb_max_event": max(ev) if ev else nan,
        "bpb_top10": sum(bits[:k]) / k if bits else nan,
        "bits_max_field": max(field_bits) if field_bits else nan,
        "bits_end": bits_end,
    }


# ---------------------------------------------------------------- metrics


def ranking_metrics(y: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    pairs = [(a, b) for a, b in zip(y, scores) if b == b]
    ys = [a for a, _ in pairs]
    ss = [b for _, b in pairs]
    return {
        "auroc": auroc(ys, ss),
        "pr_auc": pr_auc(ys, ss),
        "precision_at_10": precision_at_k(ys, ss, 10),
        "n": len(ys),
    }


def per_capture_auroc(
    sessions: Sequence[LabeledSession], scores: Sequence[float], capture_of: dict[str, str]
) -> dict[str, float]:
    by: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    for s, sc in zip(sessions, scores):
        if sc != sc:
            continue
        ys, ss = by[capture_of.get(s.session_id, "")]
        ys.append(int(s.binary))
        ss.append(float(sc))
    return {cid: auroc(ys, ss) for cid, (ys, ss) in sorted(by.items())}


# ---------------------------------------------------------------- heatmap


def _heat_html(toks, text) -> str:
    is_id = _noise_mask(text, toks)
    parts = []
    for (tok, n, b), m in zip(toks, is_id):
        bits = n / math.log(2)
        a = min(1.0, bits / 12.0)
        style = f"background:rgba(220,38,38,{a:.2f})" + (";opacity:.45" if m else "")
        parts.append(f'<span title="{bits:.1f} bits" style="{style}">{html.escape(tok)}</span>')
    return "".join(parts)


def write_heatmap(path: Path, picks: list[tuple[str, LabeledSession, float, list, str]]) -> None:
    body = []
    for title, s, score, toks, text in picks:
        body.append(
            f"<h3>{html.escape(title)} · {html.escape(s.label)}"
            f"{' / ' + html.escape(s.fault) if s.fault else ''} · score {score:.3f}</h3>"
            f"<pre>{_heat_html(toks, text)}</pre>"
        )
    path.write_text(
        "<!doctype html><meta charset=utf-8><title>AOMB surprise heatmap</title>"
        "<style>body{font:13px ui-monospace,monospace;margin:16px;max-width:1200px}"
        "pre{white-space:pre-wrap;word-break:break-all;border:1px solid #ddd;padding:8px}"
        "h3{font:600 14px system-ui;margin:20px 0 6px}</style>"
        "<h1 style='font:600 18px system-ui'>Per-token surprise (red = more bits; faded = IDs, timestamps, counters: not scored)</h1>"
        + "".join(body),
        encoding="utf-8",
    )


# ---------------------------------------------------------------- main


def run(
    capture: str | None,
    seeds: list[int],
    train_seconds: float,
    out_dir: Path,
    sessions_file: str | None = None,
    train_chunk_lines: int | None = None,
    end_marker: bool = False,
    inject_n: int = 0,
) -> dict[str, Any]:
    from eval.fixture_train import train_fixture_lm
    from eval.report import git_sha

    if sessions_file:
        train, eval_set, capture_of, corpus_meta = load_session_file(sessions_file)
    else:
        sessions, corpus_meta = load_lab_sessions(capture)
        _y, kept = filter_scorable(sessions)
        capture_of = session_capture_ids(capture)
        train, eval_set = temporal_split(kept, capture_of)
    y = [int(s.binary) for s in eval_set]

    results: dict[str, Any] = {
        "protocol": "eval/in_domain.py (temporal split, train on normals only)",
        "capture_id": corpus_meta["capture_id"],
        "content_sha256": corpus_meta["content_sha256"],
        "git_head": git_sha(ROOT),
        "n_train_normal": len(train),
        "n_eval": len(eval_set),
        "n_eval_normal": y.count(0),
        "n_eval_incident": y.count(1),
        "train_seconds": train_seconds,
        "train_chunk_lines": train_chunk_lines,
        "end_marker": end_marker,
        "seeds": seeds,
        "methods": {},
    }

    subset_ids = corpus_meta.pop("subset_ids", set()) if sessions_file else set()
    sub_idx = [i for i, s in enumerate(eval_set) if s.session_id in subset_ids]
    y_sub = [y[i] for i in sub_idx]
    if sub_idx:
        results["subset"] = {
            "field": corpus_meta.get("subset_field"),
            "n": len(sub_idx),
            "n_incident": sum(y_sub),
            "auroc": {},
        }

    stats = fit_duration_stats(train)
    seen = fit_novelty(train)
    values = ValueStats(train)
    for name, sc in baseline_scores(eval_set, stats, seen, values).items():
        if sub_idx:
            results["subset"]["auroc"][name] = auroc(y_sub, [sc[i] for i in sub_idx])
        results["methods"][name] = {
            "kind": "baseline",
            "metrics": ranking_metrics(y, sc),
            "per_capture_auroc": per_capture_auroc(eval_set, sc, capture_of),
        }

    # Label-free objective: corrupted copies of some eval normals (eval.inject).
    inj_clean_idx: list[int] = []
    inj_corrupted: dict[str, list[tuple[int, str]]] = {}
    inj_per_seed: list[dict[str, dict[str, float]]] = []
    if inject_n:
        from eval.inject import Injector, injection_auroc

        normals = [i for i, s in enumerate(eval_set) if s.binary == 0]
        step = max(1, len(normals) // inject_n)
        inj_clean_idx = normals[::step][:inject_n]
        inj_corrupted = Injector([s.text for s in train], seed=0).corrupt(
            [eval_set[i].text for i in inj_clean_idx]
        )

    per_seed: dict[str, list[dict]] = {m: [] for m in MODEL_METHODS}
    per_seed_cap: dict[str, list[dict]] = {m: [] for m in MODEL_METHODS}
    session_rows: dict[int, dict[str, list[float]]] = {}
    train_info: list[dict] = []
    for seed in seeds:
        model, tokenizer, token_bytes, info = train_fixture_lm(
            chunk_lines([s.text for s in train], train_chunk_lines, END_MARKER if end_marker else None),
            seed=seed,
            train_seconds=train_seconds,
            max_seq_len=None,
        )
        if info["n_train_cropped"]:
            raise RuntimeError(f"{info['n_train_cropped']} training sessions cropped")
        train_info.append({"seed": seed, **info})
        scores: dict[str, list[float]] = {m: [] for m in MODEL_METHODS}
        all_toks = []
        scored_texts = [s.text + "\n" + END_MARKER if end_marker else s.text for s in eval_set]
        for text in scored_texts:
            toks = token_surprise(model, tokenizer, token_bytes, text, info["max_seq_len"])
            all_toks.append(toks)
            for m, v in model_scores_for(toks, text).items():
                scores[m].append(v)
        for m in MODEL_METHODS:
            per_seed[m].append(ranking_metrics(y, scores[m]))
            per_seed_cap[m].append(per_capture_auroc(eval_set, scores[m], capture_of))
        session_rows[seed] = scores
        if inject_n:
            seed_inj: dict[str, dict[str, float]] = {m: {} for m in MODEL_METHODS}
            for name, items in inj_corrupted.items():
                cor: dict[str, list[float]] = {m: [] for m in MODEL_METHODS}
                for _i, text in items:
                    text = text + "\n" + END_MARKER if end_marker else text
                    toks = token_surprise(model, tokenizer, token_bytes, text, info["max_seq_len"])
                    for m, v in model_scores_for(toks, text).items():
                        cor[m].append(v)
                for m in MODEL_METHODS:
                    clean = [scores[m][i] for i in inj_clean_idx]
                    pairs = [(c, 0) for c in clean if c == c] + [(c, 1) for c in cor[m] if c == c]
                    seed_inj[m][name] = (
                        injection_auroc([c for c, y_ in pairs if y_ == 0], [c for c, y_ in pairs if y_ == 1])
                        if cor[m] else float("nan")
                    )
            for m in MODEL_METHODS:
                vals = [v for v in seed_inj[m].values() if v == v]
                seed_inj[m]["mean"] = sum(vals) / len(vals) if vals else float("nan")
            inj_per_seed.append(seed_inj)
        print(
            f"seed={seed} steps={info['num_steps']} "
            + " ".join(f"{m}={per_seed[m][-1]['auroc']:.4f}" for m in MODEL_METHODS),
            flush=True,
        )
        if seed == seeds[0]:
            order = sorted(range(len(eval_set)), key=lambda i: scores["bpb_max_event"][i])
            inc = [i for i in order if eval_set[i].binary == 1]
            def pick(title, i):
                return (title, eval_set[i], scores["bpb_max_event"][i], all_toks[i], scored_texts[i])

            picks = (
                [pick("Most surprising", i) for i in order[::-1][:8]]
                + [pick("Least surprising", i) for i in order[:4]]
                + [pick("Missed incident", i) for i in inc[:6]]
            )
            write_heatmap(out_dir / "heatmap-seed0.html", picks)
        del model

    results["train_info"] = train_info
    if inject_n:
        keys = list(inj_per_seed[0][MODEL_METHODS[0]])
        results["injection"] = {
            "n_clean": len(inj_clean_idx),
            "n_corrupted": {k: len(v) for k, v in inj_corrupted.items()},
            "auroc": {
                m: {k: mean_std([sd[m][k] for sd in inj_per_seed]) for k in keys} for m in MODEL_METHODS
            },
            "per_seed": inj_per_seed,
        }
    if sub_idx:
        for m in MODEL_METHODS:
            results["subset"]["auroc"][m] = mean_std(
                [auroc(y_sub, [sc[m][i] for i in sub_idx]) for sc in session_rows.values()]
            )
    for m in MODEL_METHODS:
        caps = sorted(per_seed_cap[m][0])
        results["methods"][m] = {
            "kind": "model",
            "metrics": {
                k: mean_std([r[k] for r in per_seed[m]])
                for k in ("auroc", "pr_auc", "precision_at_10")
            },
            "per_capture_auroc": {c: mean_std([r[c] for r in per_seed_cap[m]]) for c in caps},
            "per_seed": per_seed[m],
        }
    (out_dir / "sessions.json").write_text(
        json.dumps(
            {
                "session_ids": [s.session_id for s in eval_set],
                "labels": y,
                "capture": [capture_of.get(s.session_id, "") for s in eval_set],
                "fault": [s.fault for s in eval_set],
                "model_scores_by_seed": {str(k): v for k, v in session_rows.items()},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return results


def subset_report(capture: str, out_dir: Path, marker: str) -> dict[str, Any]:
    """
    Re-rank only eval sessions whose text contains ``marker`` (e.g. the checkout
    endpoint a fault touches), from a finished run's sessions.json — no retraining.

    Window labels include sessions a fault can never reach (other endpoints,
    frontend-only traces). Restricting by *endpoint* — not by label — shows how
    each method does where the fault can actually be seen, fairly for every method.
    """
    sessions, _ = load_lab_sessions(capture)
    _y, kept = filter_scorable(sessions)
    capture_of = session_capture_ids(capture)
    train, eval_set = temporal_split(kept, capture_of)
    saved = json.loads((out_dir / "sessions.json").read_text(encoding="utf-8"))
    if [s.session_id for s in eval_set] != saved["session_ids"]:
        raise RuntimeError("eval set differs from the saved run; rerun eval.in_domain")
    idx = [i for i, s in enumerate(eval_set) if marker in s.text]
    sub_sessions = [eval_set[i] for i in idx]
    y = [int(s.binary) for s in sub_sessions]

    methods: dict[str, Any] = {}
    base = baseline_scores(
        eval_set, fit_duration_stats(train), fit_novelty(train), ValueStats(train)
    )
    for name, sc in base.items():
        sub_sc = [sc[i] for i in idx]
        methods[name] = {
            "kind": "baseline",
            "auroc": auroc(y, sub_sc),
            "per_capture_auroc": per_capture_auroc(sub_sessions, sub_sc, capture_of),
        }
    for m in MODEL_METHODS:
        per_seed = [[v[m][i] for i in idx] for v in saved["model_scores_by_seed"].values()]
        caps = [per_capture_auroc(sub_sessions, sc, capture_of) for sc in per_seed]
        methods[m] = {
            "kind": "model",
            "auroc": mean_std([auroc(y, sc) for sc in per_seed]),
            "per_capture_auroc": {c: mean_std([d[c] for d in caps]) for c in sorted(caps[0])},
        }
    return {"marker": marker, "n": len(y), "n_incident": sum(y), "methods": methods}


def _fmt(v: Any) -> str:
    if isinstance(v, dict):
        return f"{v['mean']:.3f} ± {v['std']:.3f}"
    return f"{v:.3f}"


def render_markdown(r: dict[str, Any]) -> str:
    caps = sorted(next(iter(r["methods"].values()))["per_capture_auroc"])
    lines = [
        f"# In-domain lab eval — `{r['capture_id']}`",
        "",
        f"Train: {r['n_train_normal']} earlier normal sessions (no incidents). "
        f"Eval: {r['n_eval_normal']} later normals + {r['n_eval_incident']} incidents. "
        f"Model: {r['train_seconds']:.0f}s × seeds {r['seeds']}. Git `{r['git_head'][:7]}`.",
        "",
        "| Method | AUROC | PR-AUC | P@10 | " + " | ".join(f"`{c[-18:]}`" for c in caps) + " |",
        "|---|---|---|---|" + "---|" * len(caps),
    ]
    for name, d in r["methods"].items():
        m = d["metrics"]
        pc = d["per_capture_auroc"]
        lines.append(
            f"| {name} ({d['kind']}) | {_fmt(m['auroc'])} | {_fmt(m['pr_auc'])} | "
            f"{_fmt(m['precision_at_10'])} | " + " | ".join(_fmt(pc[c]) for c in caps) + " |"
        )
    inj = r.get("injection")
    if inj:
        kinds = [k for k in next(iter(inj["auroc"].values())) if k != "mean"]
        lines += [
            "",
            f"Injected faults ({inj['n_clean']} clean eval normals, one corrupted copy per type). "
            "A label-free objective, not evidence of real detection.",
            "",
            "| Method | mean | " + " | ".join(kinds) + " |",
            "|---|---|" + "---|" * len(kinds),
        ]
        for m, d in inj["auroc"].items():
            lines.append(f"| {m} | {_fmt(d['mean'])} | " + " | ".join(_fmt(d[k]) for k in kinds) + " |")
    sub = r.get("subset")
    if sub:
        lines += [
            "",
            f"Subset `{sub['field']}`: {sub['n']} eval sessions ({sub['n_incident']} incidents). "
            "Chosen by service, not by label.",
            "",
            "| Method | AUROC |",
            "|---|---|",
        ]
        lines += [f"| {name} | {_fmt(a)} |" for name, a in sub["auroc"].items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--capture", help="lab capture dir (temporal split per capture)")
    src.add_argument("--sessions", help="prepared session JSONL with a split field")
    p.add_argument(
        "--inject",
        type=int,
        default=0,
        metavar="N",
        help="Also score corrupted copies of N eval normals (label-free objective, eval.inject)",
    )
    p.add_argument(
        "--end-marker",
        action="store_true",
        help="Append an end-of-session marker line to every session (train and eval)",
    )
    p.add_argument(
        "--train-chunk-lines",
        type=int,
        default=None,
        help="Split long training sessions into chunks of at most N lines (none dropped)",
    )
    p.add_argument("--seeds", default="0..4")
    p.add_argument("--train-seconds", type=float, default=120.0)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--refresh-baselines",
        action="store_true",
        help="Recompute baseline rows in an existing results.json (no retraining)",
    )
    p.add_argument(
        "--subset-marker",
        default=None,
        help="Only re-rank eval sessions containing this text, from an existing run",
    )
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.refresh_baselines:
        r = json.loads((out / "results.json").read_text(encoding="utf-8"))
        sessions, _ = load_lab_sessions(args.capture)
        _y, kept = filter_scorable(sessions)
        capture_of = session_capture_ids(args.capture)
        train, eval_set = temporal_split(kept, capture_of)
        y = [int(s.binary) for s in eval_set]
        base = baseline_scores(
            eval_set, fit_duration_stats(train), fit_novelty(train), ValueStats(train)
        )
        models = {k: v for k, v in r["methods"].items() if v["kind"] == "model"}
        r["methods"] = {
            name: {
                "kind": "baseline",
                "metrics": ranking_metrics(y, sc),
                "per_capture_auroc": per_capture_auroc(eval_set, sc, capture_of),
            }
            for name, sc in base.items()
        } | models
        (out / "results.json").write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        md = render_markdown(r)
        (out / "results.md").write_text(md, encoding="utf-8")
        print(md)
        return 0
    if args.subset_marker:
        r = subset_report(args.capture, out, args.subset_marker)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", args.subset_marker).strip("_")
        (out / f"subset-{slug}.json").write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
        for name, d in r["methods"].items():
            a = d["auroc"]
            a = f"{a['mean']:.3f} ± {a['std']:.3f}" if isinstance(a, dict) else f"{a:.3f}"
            caps = " ".join(f"{c[-14:]}={_fmt(v)}" for c, v in d["per_capture_auroc"].items())
            print(f"{name:16} {a:15} {caps}")
        print(f"subset n={r['n']} incidents={r['n_incident']}")
        return 0
    if args.sessions and (args.refresh_baselines or args.subset_marker):
        p.error("--refresh-baselines / --subset-marker need --capture")
    r = run(
        args.capture,
        parse_seeds(args.seeds),
        args.train_seconds,
        out,
        sessions_file=args.sessions,
        train_chunk_lines=args.train_chunk_lines,
        end_marker=args.end_marker,
        inject_n=args.inject,
    )
    (out / "results.json").write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    md = render_markdown(r)
    (out / "results.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
