"""Retrieval benchmark orchestration for the four retrieval methods."""

from __future__ import annotations

import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from procureguard.artifacts import appendJsonLines, atomicWriteJson, writeCsv
from procureguard.data.schema import Corpus, QueryRecord
from procureguard.retrieval.base import Retriever, RetrieverRegistry
from procureguard.retrieval.metrics import (
    METRIC_KEYS,
    aggregateRetrievalMetrics,
    computeQueryMetrics,
)

RETRIEVAL_METHODS = ["no_retrieval", "bm25", "lsa_vector", "hybrid_rrf"]


def _buildRetrievers(registry: RetrieverRegistry) -> dict[str, Retriever]:
    return {method: registry.build(method) for method in RETRIEVAL_METHODS}


def runRetrievalBenchmark(
    *,
    corpus: Corpus,
    queries: Iterable[QueryRecord],
    registry: RetrieverRegistry,
    bootstrapRng: random.Random,
    outputDir: Path,
    runId: str,
    topK: int = 10,
    precisionK: int = 5,
    nResamples: int = 1000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run the retrieval benchmark and persist artifacts.

    Returns:
        A summary dict keyed by retrieval method containing metrics and latency.
    """
    queryList = list(queries)
    retrievers = _buildRetrievers(registry)
    for retriever in retrievers.values():
        retriever.fit(corpus)

    perQueryMetrics: dict[str, dict[str, dict[str, float]]] = {}
    latencyByMethod: dict[str, list[float]] = {method: [] for method in RETRIEVAL_METHODS}
    tracesDir = outputDir / "traces"
    tablesDir = outputDir / "tables"
    metricsDir = outputDir / "metrics"
    tracesDir.mkdir(parents=True, exist_ok=True)
    tablesDir.mkdir(parents=True, exist_ok=True)
    metricsDir.mkdir(parents=True, exist_ok=True)
    (tracesDir / "retrieval_traces.jsonl").unlink(missing_ok=True)

    rankingRows: list[dict[str, Any]] = []
    metricRows: list[dict[str, Any]] = []

    for query in queryList:
        relevant = set(query.qrels)
        for method in RETRIEVAL_METHODS:
            retriever = retrievers[method]
            start = time.perf_counter()
            ranked = retriever.search(query.query, topK)
            latencyMs = (time.perf_counter() - start) * 1000.0
            latencyByMethod[method].append(latencyMs)
            perMethod = computeQueryMetrics(ranked, relevant, topK=topK, precisionK=precisionK)
            perQueryMetrics.setdefault(query.queryId, {})[method] = perMethod
            rankingRows.append(
                {
                    "queryId": query.queryId,
                    "method": method,
                    "rankedDocIds": ranked,
                    "latencyMs": round(latencyMs, 4),
                }
            )
            for metricKey, value in perMethod.items():
                metricRows.append(
                    {
                        "queryId": query.queryId,
                        "method": method,
                        "metric": metricKey,
                        "value": round(value, 6),
                    }
                )

    writeCsv(tablesDir / "retrieval_per_query_metrics.csv", metricRows)
    writeCsv(tablesDir / "retrieval_rankings.csv", rankingRows)
    for record in rankingRows:
        appendJsonLines(tracesDir / "retrieval_traces.jsonl", record)

    summary: dict[str, Any] = {}
    for method in RETRIEVAL_METHODS:
        perQuery = {
            queryId: methodMetrics[method] for queryId, methodMetrics in perQueryMetrics.items()
        }
        aggregated = aggregateRetrievalMetrics(
            perQuery,
            bootstrapRng,
            nResamples=nResamples,
            alpha=alpha,
        )
        latencies = latencyByMethod[method]
        summary[method] = {
            "metrics": aggregated,
            "meanLatencyMs": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        }

    output: dict[str, Any] = {
        "runId": runId,
        "numQueries": len(queryList),
        "topK": topK,
        "precisionK": precisionK,
        "methods": RETRIEVAL_METHODS,
        "results": summary,
    }
    atomicWriteJson(metricsDir / "retrieval_summary.json", output)
    return output


def renderRetrievalComparisonRow(summary: dict[str, Any], method: str) -> dict[str, Any]:
    """Render a single comparison table row from the retrieval summary."""
    entry = summary[method]
    row: dict[str, Any] = {"method": method}
    for metricKey in METRIC_KEYS:
        metric = entry["metrics"][metricKey]
        row[f"{metricKey} mean"] = round(metric["mean"], 4)
        row[f"{metricKey} ci"] = f"[{metric['ciLow']:.4f}, {metric['ciHigh']:.4f}]"
    row["meanLatencyMs"] = entry["meanLatencyMs"]
    return row
