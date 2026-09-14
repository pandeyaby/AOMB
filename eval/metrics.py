"""
Ranking metrics for public accuracy eval (pure Python — no sklearn).

Scores are higher = more anomalous (session surprise / BPB).
Labels: 0 = normal, 1 = incident/cascade.
"""

from __future__ import annotations

import math
import random
from typing import Sequence


def _validate(y_true: Sequence[int], y_score: Sequence[float]) -> tuple[list[int], list[float]]:
    if len(y_true) != len(y_score):
        raise ValueError(f"length mismatch: labels={len(y_true)} scores={len(y_score)}")
    yt = [int(y) for y in y_true]
    ys = [float(s) for s in y_score]
    if not yt:
        raise ValueError("empty label/score lists")
    if any(y not in (0, 1) for y in yt):
        raise ValueError("labels must be binary 0/1")
    if any(math.isnan(s) or math.isinf(s) for s in ys):
        raise ValueError("scores must be finite")
    return yt, ys


def auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """
    ROC AUC via Mann–Whitney U (ties averaged).

    Returns NaN if only one class is present.
    """
    yt, ys = _validate(y_true, y_score)
    pos = [s for s, y in zip(ys, yt) if y == 1]
    neg = [s for s, y in zip(ys, yt) if y == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Rank all scores ascending; average ties
    indexed = sorted(enumerate(ys), key=lambda t: t[1])
    ranks = [0.0] * len(ys)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + (j + 1))  # 1-based ranks
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1

    sum_pos_ranks = sum(ranks[i] for i, y in enumerate(yt) if y == 1)
    # U = R_pos - n_pos*(n_pos+1)/2 ; AUC = U / (n_pos * n_neg)
    u = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """
    Area under the precision-recall curve (average precision style).

    Returns NaN if there are no positives.
    """
    yt, ys = _validate(y_true, y_score)
    n_pos = sum(yt)
    if n_pos == 0:
        return float("nan")

    order = sorted(range(len(ys)), key=lambda i: ys[i], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    i = 0
    while i < len(order):
        # Handle score ties as one step
        j = i
        while j + 1 < len(order) and ys[order[j + 1]] == ys[order[i]]:
            j += 1
        for k in range(i, j + 1):
            if yt[order[k]] == 1:
                tp += 1
            else:
                fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        area += precision * (recall - prev_recall)
        prev_recall = recall
        i = j + 1
    return area


def precision_at_k(y_true: Sequence[int], y_score: Sequence[float], k: int) -> float:
    """Fraction of positives among the k highest scores. Ties broken by stable sort index."""
    yt, ys = _validate(y_true, y_score)
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, len(ys))
    order = sorted(range(len(ys)), key=lambda i: (-ys[i], i))
    top = order[:k]
    return sum(yt[i] for i in top) / k


def default_k_values(n: int, n_pos: int) -> list[int]:
    """Protocol defaults: k = min(10, n_pos) and k = max(1, n // 10), unique sorted."""
    ks: set[int] = set()
    if n_pos > 0:
        ks.add(max(1, min(10, n_pos)))
    if n > 0:
        ks.add(max(1, n // 10))
    return sorted(ks)


def random_baseline_metrics(
    y_true: Sequence[int],
    *,
    n_draws: int = 64,
    seed: int = 0,
    k_values: Sequence[int] | None = None,
) -> dict:
    """
    Random ranking baseline: draw Uniform(0,1) scores repeatedly; report mean±std.
    """
    yt = [int(y) for y in y_true]
    if not yt:
        raise ValueError("empty labels")
    n_pos = sum(yt)
    ks = list(k_values) if k_values is not None else default_k_values(len(yt), n_pos)
    rng = random.Random(seed)

    aurocs: list[float] = []
    pras: list[float] = []
    precs: dict[int, list[float]] = {k: [] for k in ks}

    for _ in range(n_draws):
        scores = [rng.random() for _ in yt]
        a = auroc(yt, scores)
        p = pr_auc(yt, scores)
        if a == a:
            aurocs.append(a)
        if p == p:
            pras.append(p)
        for k in ks:
            precs[k].append(precision_at_k(yt, scores, k))

    def _mean_std(xs: list[float]) -> dict:
        if not xs:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        mean = sum(xs) / len(xs)
        if len(xs) == 1:
            return {"mean": mean, "std": 0.0, "n": 1}
        var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
        return {"mean": mean, "std": math.sqrt(var), "n": len(xs)}

    return {
        "n_draws": n_draws,
        "seed": seed,
        "auroc": _mean_std(aurocs),
        "pr_auc": _mean_std(pras),
        "precision_at_k": {str(k): _mean_std(precs[k]) for k in ks},
    }


def summarize_ranking(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    k_values: Sequence[int] | None = None,
    random_draws: int = 64,
    random_seed: int = 0,
) -> dict:
    """Full metric bundle for one scored set + random baseline."""
    yt, ys = _validate(y_true, y_score)
    n_pos = sum(yt)
    n_neg = len(yt) - n_pos
    ks = list(k_values) if k_values is not None else default_k_values(len(yt), n_pos)
    return {
        "n": len(yt),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "auroc": auroc(yt, ys),
        "pr_auc": pr_auc(yt, ys),
        "precision_at_k": {str(k): precision_at_k(yt, ys, k) for k in ks},
        "score_mean_by_label": {
            "0": _mean([s for s, y in zip(ys, yt) if y == 0]),
            "1": _mean([s for s, y in zip(ys, yt) if y == 1]),
        },
        "random_baseline": random_baseline_metrics(
            yt, n_draws=random_draws, seed=random_seed, k_values=ks
        ),
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def mean_std(values: Sequence[float]) -> dict:
    xs = [float(v) for v in values if v == v]
    if not xs:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = sum(xs) / len(xs)
    if len(xs) == 1:
        return {"mean": mean, "std": 0.0, "n": 1}
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return {"mean": mean, "std": math.sqrt(var), "n": len(xs)}
