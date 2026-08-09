"""Decision workflow benchmark orchestration for the four methods."""

from __future__ import annotations

import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from procureguard.artifacts import appendJsonLines, atomicWriteJson, writeCsv
from procureguard.data.schema import Corpus, Scenario
from procureguard.mcp.client import FactsProvider
from procureguard.policy import budgetAdherenceFor, complianceFlagFor, riskFlagFor
from procureguard.retrieval.base import Retriever, corpusById
from procureguard.workflow import engine
from procureguard.workflow.metrics import (
    EvaluatedRun,
    computeWorkflowMetrics,
    workflowCiValues,
)
from procureguard.workflow.provider import DecisionProvider

WORKFLOW_METHODS = ["rules_baseline", "rag_only", "rag_mcp", "guarded_rag_mcp"]


def _factsForScenario(scenario: Scenario) -> dict[str, str]:
    return {
        "riskLevel": scenario.riskLevel,
        "complianceStatus": scenario.complianceStatus,
        "budgetStatus": scenario.budgetStatus,
        "spendAmount": f"{scenario.spendAmount:.0f}",
        "budgetAmount": f"{scenario.budgetAmount:.0f}",
    }


async def _runOne(
    scenario: Scenario,
    method: str,
    provider: DecisionProvider,
    retriever: Retriever,
    corpus: Corpus,
    factsProvider: FactsProvider,
    qrelsByQuery: dict[str, set[str]],
    *,
    topK: int,
) -> tuple[EvaluatedRun, dict[str, Any]]:
    ctx = engine.DecisionContext(
        query=scenario.query,
        supplierId=scenario.supplierId,
        category=scenario.category,
        injectionAttempt=scenario.injectionAttempt,
        expectedFacts=_factsForScenario(scenario),
        budgetAmount=scenario.budgetAmount,
    )
    start = time.perf_counter()
    try:
        if method == "rules_baseline":
            outcome = await engine.runRulesBaseline(ctx, provider)
        elif method == "rag_only":
            outcome = await engine.runRagOnly(
                ctx, provider, retriever, corpusById(corpus), topK=topK
            )
        elif method == "rag_mcp":
            outcome = await engine.runRagMcp(
                ctx, provider, retriever, corpusById(corpus), factsProvider, topK=topK
            )
        elif method == "guarded_rag_mcp":
            outcome = await engine.runGuardedRagMcp(
                ctx,
                provider,
                retriever,
                corpusById(corpus),
                factsProvider,
                qrelsByQuery.get(scenario.queryId, set()),
                topK=topK,
            )
        else:
            raise ValueError(f"Unknown workflow method: {method}")
    except Exception as exc:  # noqa: BLE001 - captured into trace for robustness
        decision = None
        error = f"{type(exc).__name__}: {exc}"
        totalLatencyMs = (time.perf_counter() - start) * 1000.0
        evaluated = EvaluatedRun(
            scenarioId=scenario.scenarioId,
            method=method,
            expectedAction=scenario.expectedAction,
            actualAction="error",
            expectedComplianceFlag=complianceFlagFor(scenario.complianceStatus),
            actualComplianceFlag=False,
            expectedRiskFlag=riskFlagFor(scenario.riskLevel),
            actualRiskFlag=False,
            expectedBudgetAdherence=budgetAdherenceFor(scenario.budgetStatus),
            actualBudgetAdherence=False,
            expectedHumanApproval=scenario.requiresHumanApproval,
            actualHumanApproval=False,
            proposedWrite=False,
            citations=[],
            relevantDocIds=qrelsByQuery.get(scenario.queryId, set()),
            corpusDocIds={doc.docId for doc in corpus.documents},
            totalLatencyMs=totalLatencyMs,
            toolCalls=0,
            quarantineHit=False,
            error=error,
        )
        return evaluated, {"error": error}

    totalLatencyMs = (time.perf_counter() - start) * 1000.0
    decision = outcome.decision
    evaluated = EvaluatedRun(
        scenarioId=scenario.scenarioId,
        method=method,
        expectedAction=scenario.expectedAction,
        actualAction=decision.action,
        expectedComplianceFlag=complianceFlagFor(scenario.complianceStatus),
        actualComplianceFlag=decision.complianceFlag,
        expectedRiskFlag=riskFlagFor(scenario.riskLevel),
        actualRiskFlag=decision.riskFlag,
        expectedBudgetAdherence=budgetAdherenceFor(scenario.budgetStatus),
        actualBudgetAdherence=decision.budgetAdherence,
        expectedHumanApproval=scenario.requiresHumanApproval,
        actualHumanApproval=decision.requiresHumanApproval,
        proposedWrite=decision.proposedWrite,
        citations=list(decision.citations),
        relevantDocIds=qrelsByQuery.get(scenario.queryId, set()),
        corpusDocIds={doc.docId for doc in corpus.documents},
        totalLatencyMs=totalLatencyMs,
        toolCalls=outcome.toolCalls,
        quarantineHit=decision.quarantineHit,
        unsupportedFacts=list(decision.unsupportedFacts),
    )
    trace = {
        "scenarioId": scenario.scenarioId,
        "queryId": scenario.queryId,
        "split": scenario.split,
        "method": method,
        "provider": provider.name,
        "injectionAttempt": scenario.injectionAttempt,
        "expectedAction": scenario.expectedAction,
        "actualAction": decision.action,
        "rationale": decision.rationale,
        "riskFlag": decision.riskFlag,
        "complianceFlag": decision.complianceFlag,
        "budgetAdherence": decision.budgetAdherence,
        "requiresHumanApproval": decision.requiresHumanApproval,
        "proposedWrite": decision.proposedWrite,
        "writeBlocked": decision.writeBlocked,
        "citations": list(decision.citations),
        "quarantineHit": decision.quarantineHit,
        "unsupportedFacts": list(decision.unsupportedFacts),
        "toolCalls": outcome.toolCalls,
        "droppedCitations": outcome.droppedCitations,
        "totalLatencyMs": round(totalLatencyMs, 4),
        "retrievalLatencyMs": round(outcome.retrievalLatencyMs, 4),
        "providerRequestId": decision.requestId,
        "usage": decision.usage,
        "error": outcome.error,
    }
    return evaluated, trace


async def runWorkflowBenchmark(
    *,
    corpus: Corpus,
    scenarios: Iterable[Scenario],
    provider: DecisionProvider,
    retriever: Retriever,
    factsProvider: FactsProvider,
    qrelsByQuery: dict[str, set[str]],
    bootstrapRng: random.Random,
    outputDir: Path,
    runId: str,
    topK: int,
    nResamples: int = 1000,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run the decision workflow benchmark and persist artifacts."""
    scenarioList = list(scenarios)
    evaluatedByMethod: dict[str, list[EvaluatedRun]] = {method: [] for method in WORKFLOW_METHODS}
    tracesDir = outputDir / "traces"
    tablesDir = outputDir / "tables"
    metricsDir = outputDir / "metrics"
    tracesDir.mkdir(parents=True, exist_ok=True)
    tablesDir.mkdir(parents=True, exist_ok=True)
    metricsDir.mkdir(parents=True, exist_ok=True)
    (tracesDir / "workflow_traces.jsonl").unlink(missing_ok=True)

    allTraces: list[dict[str, Any]] = []
    for scenario in scenarioList:
        for method in WORKFLOW_METHODS:
            evaluated, trace = await _runOne(
                scenario,
                method,
                provider,
                retriever,
                corpus,
                factsProvider,
                qrelsByQuery,
                topK=topK,
            )
            evaluatedByMethod[method].append(evaluated)
            allTraces.append(trace)
            appendJsonLines(tracesDir / "workflow_traces.jsonl", trace)

    writeCsv(tablesDir / "workflow_per_run.csv", _rowsFromTraces(allTraces))

    summary: dict[str, Any] = {}
    for method in WORKFLOW_METHODS:
        runs = evaluatedByMethod[method]
        metrics = computeWorkflowMetrics(runs)
        ci = workflowCiValues(runs, bootstrapRng, nResamples=nResamples, alpha=alpha)
        summary[method] = {"metrics": metrics, "ci": ci}

    output: dict[str, Any] = {
        "runId": runId,
        "numScenarios": len(scenarioList),
        "topK": topK,
        "provider": provider.name,
        "methods": WORKFLOW_METHODS,
        "results": summary,
    }
    atomicWriteJson(metricsDir / "workflow_summary.json", output)
    return output


def _rowsFromTraces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        rows.append(
            {
                "scenarioId": trace["scenarioId"],
                "method": trace["method"],
                "expectedAction": trace["expectedAction"],
                "actualAction": trace["actualAction"],
                "correct": int(trace["expectedAction"] == trace["actualAction"]),
                "proposedWrite": int(trace["proposedWrite"]),
                "requiresHumanApproval": int(trace["requiresHumanApproval"]),
                "quarantineHit": int(trace["quarantineHit"]),
                "toolCalls": trace["toolCalls"],
                "totalLatencyMs": trace["totalLatencyMs"],
            }
        )
    return rows
