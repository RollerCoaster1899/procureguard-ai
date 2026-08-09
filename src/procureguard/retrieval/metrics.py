"""Retrieval ranking metrics and bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

import numpy as np

METRIC_KEYS = ["ndcg@10", "recall@10", "mrr@10", "precision@5"]


def ndcgAtK(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Compute nDCG at cutoff k with binary relevance."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for position, docId in enumerate(ranked[:k], start=1):
        if docId in relevant:
            dcg += 1.0 / math.log2(position + 1)
    idcg = 0.0
    for position in range(1, min(len(relevant), k) + 1):
        idcg += 1.0 / math.log2(position + 1)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recallAtK(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Compute recall at cutoff k."""
    if not relevant:
        return 0.0
    hits = sum(1 for docId in ranked[:k] if docId in relevant)
    return hits / len(relevant)


def mrrAtK(ranked: Sequence[str], relevant: set[str], k: int = 10) -> float:
    """Compute mean reciprocal rank at cutoff k."""
    if not relevant:
        return 0.0
    for position, docId in enumerate(ranked[:k], start=1):
        if docId in relevant:
            return 1.0 / position
    return 0.0


def precisionAtK(ranked: Sequence[str], relevant: set[str], k: int = 5) -> float:
    """Compute precision at cutoff k."""
    if k == 0:
        return 0.0
    hits = sum(1 for docId in ranked[:k] if docId in relevant)
    return hits / k


def computeQueryMetrics(
    ranked: Sequence[str], relevant: set[str], *, topK: int = 10, precisionK: int = 5
) -> dict[str, float]:
    """Compute all retrieval metrics for a single query."""
    return {
        "ndcg@10": ndcgAtK(ranked, relevant, topK),
        "recall@10": recallAtK(ranked, relevant, topK),
        "mrr@10": mrrAtK(ranked, relevant, topK),
        f"precision@{precisionK}": precisionAtK(ranked, relevant, precisionK),
    }


def bootstrapMeanCI(
    values: Sequence[float],
    rng: random.Random,
    *,
    nResamples: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (mean, ciLow, ciHigh) using a seeded bootstrap.

    If there are fewer than two values, the CI is returned as the point
    estimate on both sides.
    """
    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        mean = float(np.mean(arr))
        return (mean, mean, mean)
    resamples = np.empty(nResamples, dtype=float)
    indices = np.arange(len(arr))
    for i in range(nResamples):
        sample = arr[rng.choices(list(indices), k=len(arr))]
        resamples[i] = np.mean(sample)
    mean = float(np.mean(arr))
    ciLow, ciHigh = np.percentile(resamples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (mean, float(ciLow), float(ciHigh))


def aggregateRetrievalMetrics(
    perQuery: Mapping[str, Mapping[str, float]],
    rng: random.Random,
    *,
    nResamples: int = 1000,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Aggregate per-query metrics into a summary with bootstrap CIs.

    Args:
        perQuery: Mapping from query ID to per-metric values.
        rng: Deterministic RNG for bootstrap resampling.

    Returns:
        A summary dict with mean and CI for each metric.
    """
    keys = list(next(iter(perQuery.values()), {}).keys())
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [metrics[key] for metrics in perQuery.values()]
        mean, ciLow, ciHigh = bootstrapMeanCI(values, rng, nResamples=nResamples, alpha=alpha)
        summary[key] = {
            "mean": round(mean, 6),
            "ciLow": round(ciLow, 6),
            "ciHigh": round(ciHigh, 6),
        }
    return summary
