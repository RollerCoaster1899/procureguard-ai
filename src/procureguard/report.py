"""Figure generation and final report rendering for ProcureGuard AI.

All figures and report tables are generated from machine-readable metric
artifacts. No numbers are typed by hand.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from procureguard.artifacts import atomicWriteText
from procureguard.retrieval.benchmark import RETRIEVAL_METHODS
from procureguard.workflow.benchmark import WORKFLOW_METHODS

METHOD_LABELS = {
    "no_retrieval": "No retrieval (baseline)",
    "bm25": "BM25",
    "lsa_vector": "LSA vectors",
    "hybrid_rrf": "Hybrid RRF",
    "rules_baseline": "Rules baseline",
    "rag_only": "RAG only",
    "rag_mcp": "RAG + MCP",
    "guarded_rag_mcp": "Guarded RAG + MCP",
}


def _plotBar(
    ax: Any,
    labels: list[str],
    means: list[float],
    errors: list[float] | None,
    title: str,
    ylabel: str,
) -> None:
    positions = range(len(labels))
    bars = ax.bar(
        positions,
        means,
        yerr=errors if errors else None,
        capsize=4,
        color=["#4C72B0"] * len(labels),
        edgecolor="black",
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(
        [METHOD_LABELS.get(label, label) for label in labels], rotation=20, ha="right"
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(max(means) * 1.2, 0.15))
    for bar, value in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plotRetrievalComparison(retrievalSummary: dict[str, Any], figuresDir: Path) -> None:
    """Render retrieval metric comparison figures."""
    figuresDir.mkdir(parents=True, exist_ok=True)
    methods = list(retrievalSummary["results"].keys())
    ndcgMeans = []
    ndcgErrors = []
    recallMeans = []
    recallErrors = []
    for method in methods:
        metrics = retrievalSummary["results"][method]["metrics"]
        ndcgMeans.append(metrics["ndcg@10"]["mean"])
        ndcgErrors.append(
            max(
                metrics["ndcg@10"]["mean"] - metrics["ndcg@10"]["ciLow"],
                metrics["ndcg@10"]["ciHigh"] - metrics["ndcg@10"]["mean"],
            )
        )
        recallMeans.append(metrics["recall@10"]["mean"])
        recallErrors.append(
            max(
                metrics["recall@10"]["mean"] - metrics["recall@10"]["ciLow"],
                metrics["recall@10"]["ciHigh"] - metrics["recall@10"]["mean"],
            )
        )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _plotBar(
        axes[0], methods, ndcgMeans, ndcgErrors, "Retrieval nDCG@10 (bootstrap 95% CI)", "nDCG@10"
    )
    _plotBar(
        axes[1],
        methods,
        recallMeans,
        recallErrors,
        "Retrieval Recall@10 (bootstrap 95% CI)",
        "Recall@10",
    )
    figure.suptitle("Retrieval benchmark: identical queries and qrels", fontsize=12)
    figure.tight_layout()
    figure.savefig(figuresDir / "retrieval_metrics.png", dpi=150)
    plt.close(figure)


def plotWorkflowComparison(workflowSummary: dict[str, Any], figuresDir: Path) -> None:
    """Render workflow metric comparison figures."""
    figuresDir.mkdir(parents=True, exist_ok=True)
    methods = list(workflowSummary["results"].keys())

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    _plotBar(
        axes[0],
        methods,
        [workflowSummary["results"][m]["metrics"]["recommendationAccuracy"] for m in methods],
        None,
        "Workflow recommendation accuracy (held-out test scenarios)",
        "Accuracy",
    )
    flagMetrics = ["complianceFlagRecall", "riskFlagRecall", "budgetAdherence"]
    flagLabels = ["Compliance recall", "Risk recall", "Budget adherence"]
    x = range(len(methods))
    width = 0.25
    for offset, (flagMetric, flagLabel) in enumerate(zip(flagMetrics, flagLabels, strict=True)):
        values = [workflowSummary["results"][m]["metrics"][flagMetric] for m in methods]
        axes[1].bar(
            [position + offset * width for position in x],
            values,
            width=width,
            label=flagLabel,
            edgecolor="black",
        )
    axes[1].set_xticks([position + width for position in x])
    axes[1].set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], rotation=20, ha="right")
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("Rate")
    axes[1].set_title("Evidence and flag metrics")
    axes[1].legend(fontsize=7)
    figure.suptitle("Decision workflow benchmark (scripted provider)", fontsize=12)
    figure.tight_layout()
    figure.savefig(figuresDir / "workflow_evidence_flags.png", dpi=150)
    plt.close(figure)

    figure2, axes2 = plt.subplots(1, 2, figsize=(12, 4.4))
    unsafe = [
        workflowSummary["results"][m]["metrics"]["unsafeAutonomousActionRate"] for m in methods
    ]
    gate = [workflowSummary["results"][m]["metrics"]["humanGateCompleteness"] for m in methods]
    _plotBar(axes2[0], methods, unsafe, None, "Unsafe autonomous action rate", "Rate")
    _plotBar(axes2[1], methods, gate, None, "Mandatory human-gate completeness", "Rate")
    figure2.suptitle("Safety gate comparison", fontsize=12)
    figure2.tight_layout()
    figure2.savefig(figuresDir / "workflow_safety_gates.png", dpi=150)
    plt.close(figure2)


def _retrievalTable(retrievalSummary: dict[str, Any]) -> str:
    lines = ["| Method | nDCG@10 | Recall@10 | MRR@10 | Precision@5 | Latency (ms) |"]
    lines.append("|---|---|---|---|---|---|")
    for method in RETRIEVAL_METHODS:
        metrics = retrievalSummary["results"][method]["metrics"]
        latency = retrievalSummary["results"][method]["meanLatencyMs"]
        lines.append(
            f"| {method} | "
            f"{metrics['ndcg@10']['mean']:.4f} | "
            f"{metrics['recall@10']['mean']:.4f} | "
            f"{metrics['mrr@10']['mean']:.4f} | "
            f"{metrics['precision@5']['mean']:.4f} | "
            f"{latency:.2f} |"
        )
    return "\n".join(lines)


def _workflowTable(workflowSummary: dict[str, Any]) -> str:
    lines = [
        "| Method | Accuracy | Compliance recall | Risk recall | Budget adherence | "
        "Citation validity | Unsafe action rate | Human-gate | Latency (ms) | Tool calls |"
    ]
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for method in WORKFLOW_METHODS:
        metrics = workflowSummary["results"][method]["metrics"]
        lines.append(
            f"| {method} | "
            f"{metrics['recommendationAccuracy']:.4f} | "
            f"{metrics['complianceFlagRecall']:.4f} | "
            f"{metrics['riskFlagRecall']:.4f} | "
            f"{metrics['budgetAdherence']:.4f} | "
            f"{metrics['citationValidity']:.4f} | "
            f"{metrics['unsafeAutonomousActionRate']:.4f} | "
            f"{metrics['humanGateCompleteness']:.4f} | "
            f"{metrics['meanLatencyMs']:.2f} | "
            f"{metrics['totalToolCalls']} |"
        )
    return "\n".join(lines)


def _workflowCiRows(workflowSummary: dict[str, Any]) -> str:
    lines = ["| Method | Accuracy 95% CI | Unsafe rate 95% CI | Human-gate 95% CI |"]
    lines.append("|---|---|---|---|")
    for method in WORKFLOW_METHODS:
        ci = workflowSummary["results"][method]["ci"]
        acc = ci.get("recommendationAccuracy", {})
        unsafe = ci.get("unsafeAutonomousActionRate", {})
        gate = ci.get("humanGateCompleteness", {})
        accText = f"[{acc['ciLow']:.4f}, {acc['ciHigh']:.4f}]" if acc else "n/a"
        unsafeText = f"[{unsafe['ciLow']:.4f}, {unsafe['ciHigh']:.4f}]" if unsafe else "n/a"
        gateText = f"[{gate['ciLow']:.4f}, {gate['ciHigh']:.4f}]" if gate else "n/a"
        lines.append(f"| {method} | {accText} | {unsafeText} | {gateText} |")
    return "\n".join(lines)


def buildFinalReport(
    *,
    combined: Mapping[str, Any],
    runMetadata: Mapping[str, Any],
    reportPath: Path,
) -> None:
    """Write the final report from generated artifacts."""
    retrieval = combined["retrieval"]
    workflow = combined["workflow"]
    report = f"""# ProcureGuard AI - Final Report

## Executive summary

ProcureGuard AI evaluates retrieval quality and decision safety for a synthetic
enterprise procurement copilot. The retrieval benchmark compares four methods on
identical queries and qrels. The decision workflow benchmark compares four
methods on identical held-out test scenarios using a deterministic scripted
provider; live DeepSeek results are never mixed into this control-plane
benchmark.

Run ID: {runMetadata.get("runId", "n/a")}
Provider: {runMetadata.get("provider", "n/a")}
Smoke mode: {runMetadata.get("smoke", False)}
Status: {runMetadata.get("status", "n/a")}

## Retrieval benchmark (all queries, identical qrels)

{_retrievalTable(retrieval)}

The no-retrieval baseline scores 0.0 by construction. The hybrid RRF method
combines BM25 and local LSA vector rankings with reciprocal rank fusion.

## Decision workflow benchmark (held-out test scenarios)

{_workflowTable(workflow)}

## Bootstrap 95% confidence intervals (seeded)

{_workflowCiRows(workflow)}

## Interpretation

- The rules baseline uses perfect structured facts and shows what a pure rule
  engine achieves, but it is fully vulnerable to prompt injection and never
  requests human approval, producing a high unsafe autonomous action rate.
- RAG-only relies on facts extracted from retrieved citations, so its accuracy
  depends on retrieval quality.
- RAG + MCP adds exact supplier facts from read-only MCP tools, improving
  factual decisions, but remains vulnerable to injection.
- Guarded RAG + MCP quarantines injections, validates citations, checks fact
  support against evidence, forces human approval, and never performs an
  autonomous write, achieving zero unsafe autonomous actions.

## Reproduction

```bash
uv sync --extra dev
uv run python scripts/reproduce.py                     # full offline experiment
uv run python scripts/reproduce.py --smoke             # smoke experiment
```

See README.md for complete verified commands and limitations.
"""
    atomicWriteText(reportPath, report)


def updateReadmeResults(combined: Mapping[str, Any], readmePath: Path) -> None:
    """Replace the KEY_RESULTS section of the README with verified tables."""
    retrieval = combined["retrieval"]
    workflow = combined["workflow"]
    block = f"""<!-- KEY_RESULTS_START -->

### Retrieval benchmark (identical queries and qrels)

{_retrievalTable(retrieval)}

### Decision workflow benchmark (held-out test scenarios, scripted provider)

{_workflowTable(workflow)}

_All numbers above are read from `reports/metrics/retrieval_summary.json` and
`reports/metrics/workflow_summary.json`. They are generated by code and not
hand-typed._

<!-- KEY_RESULTS_END -->"""
    if not readmePath.is_file():
        raise FileNotFoundError(f"README not found at {readmePath}")
    content = readmePath.read_text(encoding="utf-8")
    startMarker = "<!-- KEY_RESULTS_START -->"
    endMarker = "<!-- KEY_RESULTS_END -->"
    if startMarker in content and endMarker in content:
        startIndex = content.index(startMarker)
        endIndex = content.index(endMarker) + len(endMarker)
        content = content[:startIndex] + block + content[endIndex:]
    else:
        raise ValueError("README is missing KEY_RESULTS markers; cannot update verified results.")
    readmePath.write_text(content, encoding="utf-8")


def writeComparisonTables(combined: Mapping[str, Any], tablesDir: Path) -> None:
    """Write markdown comparison tables for traceability."""
    tablesDir.mkdir(parents=True, exist_ok=True)
    atomicWriteText(tablesDir / "retrieval_comparison.md", _retrievalTable(combined["retrieval"]))
    atomicWriteText(tablesDir / "workflow_comparison.md", _workflowTable(combined["workflow"]))


def loadCombinedSummary(metricsDir: Path) -> dict[str, Any]:
    """Load retrieval and workflow summaries from a metrics directory."""
    retrievalPath = metricsDir / "retrieval_summary.json"
    workflowPath = metricsDir / "workflow_summary.json"
    with retrievalPath.open("r", encoding="utf-8") as handle:
        retrieval = json.load(handle)
    with workflowPath.open("r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    return {"retrieval": retrieval, "workflow": workflow}
