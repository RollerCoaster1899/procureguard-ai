"""Unit tests for workflow metrics computation."""

from __future__ import annotations

from procureguard.workflow.metrics import EvaluatedRun, computeWorkflowMetrics


def _run(
    *,
    method,
    expectedAction="approve",
    actualAction="approve",
    expectedCompliance=False,
    actualCompliance=False,
    expectedRisk=False,
    actualRisk=False,
    expectedBudget=True,
    actualBudget=True,
    expectedHuman=False,
    actualHuman=False,
    proposedWrite=False,
    citations=(),
    relevant=(),
    corpus=(),
    latency=1.0,
    toolCalls=0,
):
    return EvaluatedRun(
        scenarioId="s",
        method=method,
        expectedAction=expectedAction,
        actualAction=actualAction,
        expectedComplianceFlag=expectedCompliance,
        actualComplianceFlag=actualCompliance,
        expectedRiskFlag=expectedRisk,
        actualRiskFlag=actualRisk,
        expectedBudgetAdherence=expectedBudget,
        actualBudgetAdherence=actualBudget,
        expectedHumanApproval=expectedHuman,
        actualHumanApproval=actualHuman,
        proposedWrite=proposedWrite,
        citations=list(citations),
        relevantDocIds=set(relevant),
        corpusDocIds=set(corpus),
        totalLatencyMs=latency,
        toolCalls=toolCalls,
        quarantineHit=False,
    )


def testComputeWorkflowMetricsSimple():
    runs = [_run(method="m")]
    metrics = computeWorkflowMetrics(runs)
    assert metrics["recommendationAccuracy"] == 1.0
    assert metrics["unsafeAutonomousActionRate"] == 0.0
    assert metrics["humanGateCompleteness"] == 0.0
    assert metrics["totalToolCalls"] == 0


def testComputeWorkflowMetricsMixed():
    runs = [
        _run(method="m", expectedAction="approve", actualAction="reject"),
        _run(method="m", actualAction="approve"),
    ]
    metrics = computeWorkflowMetrics(runs)
    assert metrics["recommendationAccuracy"] == 0.5


def testComplianceAndRiskRecall():
    runs = [
        _run(method="m", expectedCompliance=True, actualCompliance=True),
        _run(method="m", expectedCompliance=True, actualCompliance=False),
        _run(method="m", expectedCompliance=False, actualCompliance=False),
    ]
    metrics = computeWorkflowMetrics(runs)
    assert metrics["complianceFlagRecall"] == 0.5


def testCitationValidity():
    runs = [
        _run(method="m", citations=["a", "b"], relevant=["a"], corpus=["a", "b"]),
        _run(method="m", citations=[], relevant=[], corpus=[]),
    ]
    metrics = computeWorkflowMetrics(runs)
    assert metrics["citationValidity"] == 0.5


def testUnsafeRateAndGate():
    runs = [
        _run(method="m", proposedWrite=True, expectedHuman=True, actualHuman=False),
        _run(method="m", proposedWrite=False, expectedHuman=True, actualHuman=True),
    ]
    metrics = computeWorkflowMetrics(runs)
    assert metrics["unsafeAutonomousActionRate"] == 0.5
    assert metrics["humanGateCompleteness"] == 0.5


def testEmptyRuns():
    metrics = computeWorkflowMetrics([])
    assert metrics["recommendationAccuracy"] == 0.0
    assert metrics["numScenarios"] == 0
