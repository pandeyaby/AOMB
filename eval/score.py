"""
Session scoring for public accuracy eval.

Reuses the demo_anomaly session-BPB idea (per-session next-token CE → bits/byte).
Does NOT call or modify prepare.evaluate_bpb (shard-level sacred metric).

Modes:
  - length / event_count baselines (no model)
  - precomputed scores JSON
  - optional model scoring via session_bpb_texts (torch)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from eval.labels import LabeledSession


def score_length(sessions: Sequence[LabeledSession]) -> list[float]:
    """Length baseline: longer sessions ranked more anomalous."""
    return [float(s.n_chars if s.n_chars else len(s.text)) for s in sessions]


def score_event_count(sessions: Sequence[LabeledSession]) -> list[float]:
    return [float(s.n_events) for s in sessions]


def load_precomputed_scores(
    path: str | Path,
    sessions: Sequence[LabeledSession],
) -> list[float]:
    """
    Load scores keyed by session_id, or a bare list aligned to sessions order.

    JSON shapes:
      {"scores": {"id": 1.2, ...}}
      {"scores": [1.2, 3.4, ...]}
      [1.2, 3.4, ...]
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        scores = raw
    elif isinstance(raw, dict) and "scores" in raw:
        scores = raw["scores"]
    else:
        raise ValueError("precomputed scores JSON must be a list or {\"scores\": ...}")

    if isinstance(scores, dict):
        out: list[float] = []
        for s in sessions:
            if s.session_id not in scores:
                raise KeyError(f"missing score for session_id={s.session_id}")
            out.append(float(scores[s.session_id]))
        return out

    if not isinstance(scores, list):
        raise ValueError("scores must be list or dict")
    if len(scores) != len(sessions):
        raise ValueError(
            f"score list length {len(scores)} != sessions {len(sessions)}"
        )
    return [float(x) for x in scores]


def resolve_scores(
    sessions: Sequence[LabeledSession],
    *,
    scores_from: str = "length",
    precomputed_path: str | Path | None = None,
    score_fn: Optional[Callable[[Sequence[LabeledSession]], list[float]]] = None,
) -> tuple[list[float], str]:
    """
    Resolve session scores for ranking.

    scores_from: length | events | precomputed | model
    For model, pass score_fn from run_eval after loading torch.
    """
    method = scores_from.strip().lower()
    if score_fn is not None:
        return score_fn(sessions), method or "custom"
    if method == "length":
        return score_length(sessions), "length_baseline"
    if method in {"events", "event_count", "n_events"}:
        return score_event_count(sessions), "event_count_baseline"
    if method == "precomputed":
        if not precomputed_path:
            raise ValueError("--precomputed-scores required when scores_from=precomputed")
        return load_precomputed_scores(precomputed_path, sessions), "precomputed"
    if method == "model":
        raise ValueError(
            "scores_from=model requires the train-then-score path in run_eval "
            "(torch + prepared tokenizer/corpus)."
        )
    raise ValueError(f"unknown scores_from={scores_from!r}")


def session_bpb_texts(
    model: Any,
    tokenizer: Any,
    token_bytes: Any,
    texts: Sequence[str],
    max_seq_len: int,
) -> list[float]:
    """
    Bits-per-byte surprise per session text (chunked to max_seq_len).

    Same formula spirit as prepare.evaluate_bpb, but per held-out session —
    does not use or alter evaluate_bpb.
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    bos = tokenizer.get_bos_token_id()
    out: list[float] = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, prepend=bos)
            if len(ids) < 2:
                out.append(float("nan"))
                continue
            total_nats = 0.0
            total_bytes = 0
            pos = 0
            while pos + 1 < len(ids):
                chunk = ids[pos : pos + max_seq_len + 1]
                if len(chunk) < 2:
                    break
                x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)
                logits = model(x)
                loss_flat = F.cross_entropy(
                    logits.view(-1, logits.size(-1)).float(),
                    y.view(-1),
                    reduction="none",
                )
                y_flat = y.view(-1)
                nbytes = token_bytes[y_flat]
                mask = nbytes > 0
                total_nats += (loss_flat * mask).sum().item()
                total_bytes += int(nbytes.sum().item())
                pos += max_seq_len
            if total_bytes == 0:
                out.append(float("nan"))
            else:
                out.append(total_nats / (math.log(2) * total_bytes))
    return out
