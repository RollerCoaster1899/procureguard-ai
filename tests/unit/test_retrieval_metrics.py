"""Unit tests for retrieval ranking metrics."""

from __future__ import annotations

import random

from procureguard.retrieval.metrics import (
    aggregateRetrievalMetrics,
    bootstrapMeanCI,
    mrrAtK,
    ndcgAtK,
    precisionAtK,
    recallAtK,
)


def testNdcgAtKKnownValue():
    # ranked positions 1..3 with relevance at 2 and 3
    value = ndcgAtK(["a", "b", "c"], {"b", "c"}, k=3)
    import math

    dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert abs(value - dcg / idcg) < 1e-9


def testNdcgAtKNoRelevant():
    assert ndcgAtK(["a", "b"], {"x"}, k=10) == 0.0
    assert ndcgAtK(["a"], set(), k=10) == 0.0


def testRecallAtK():
    assert recallAtK(["a", "b", "c"], {"b", "c"}, k=2) == 0.5
    assert recallAtK(["a", "b", "c"], {"b", "c"}, k=1) == 0.0
    assert recallAtK([], {"b"}, k=2) == 0.0


def testMrrAtK():
    assert mrrAtK(["a", "b", "c"], {"c"}, k=3) == 1.0 / 3.0
    assert mrrAtK(["a", "b"], {"c"}, k=2) == 0.0
    assert mrrAtK([], set(), k=5) == 0.0


def testPrecisionAtK():
    ranked = ["a", "b", "c", "d", "e"]
    assert precisionAtK(ranked, {"a", "d"}, k=5) == 0.4
    assert precisionAtK(ranked, {"a", "d"}, k=0) == 0.0


def testBootstrapMeanCiDeterministic():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    rngA = random.Random(7)
    rngB = random.Random(7)
    first = bootstrapMeanCI(values, rngA, nResamples=100)
    second = bootstrapMeanCI(values, rngB, nResamples=100)
    assert first == second
    mean, ciLow, ciHigh = first
    assert ciLow <= mean <= ciHigh


def testBootstrapEmptyAndSingle():
    assert bootstrapMeanCI([], random.Random(1)) == (0.0, 0.0, 0.0)
    mean, ciLow, ciHigh = bootstrapMeanCI([0.25], random.Random(1))
    assert (mean, ciLow, ciHigh) == (0.25, 0.25, 0.25)


def testAggregateRetrievalMetrics():
    perQuery = {
        "q1": {"ndcg@10": 0.8, "recall@10": 0.5},
        "q2": {"ndcg@10": 0.6, "recall@10": 0.75},
    }
    summary = aggregateRetrievalMetrics(perQuery, random.Random(7), nResamples=100)
    assert summary["ndcg@10"]["mean"] == 0.7
    assert "ciLow" in summary["recall@10"]
