"""End-to-end smoke tests for the reproduce pipeline."""

from __future__ import annotations

import json

from procureguard.pipeline import runExperiment


def _stripTimings(obj):
    """Recursively remove timing fields that vary between runs."""
    if isinstance(obj, dict):
        return {
            key: _stripTimings(value)
            for key, value in obj.items()
            if key not in {"meanLatencyMs", "startedAt", "finishedAt", "runId"}
        }
    if isinstance(obj, list):
        return [_stripTimings(item) for item in obj]
    return obj


def testSmokeRunProducesArtifacts(tmpRepo):
    output = tmpRepo / "reports_smoke"
    runExperiment(
        rootDir=tmpRepo,
        outputDir=output,
        smoke=True,
        runId="smoke-test-1",
    )
    summaryPath = output / "metrics" / "summary.json"
    assert summaryPath.is_file()
    summary = json.loads(summaryPath.read_text(encoding="utf-8"))
    retrieval = summary["retrieval"]
    assert retrieval["numQueries"] == 6
    assert retrieval["results"]["no_retrieval"]["metrics"]["ndcg@10"]["mean"] == 0.0
    workflow = summary["workflow"]
    guarded = workflow["results"]["guarded_rag_mcp"]["metrics"]
    assert guarded["unsafeAutonomousActionRate"] == 0.0
    assert guarded["humanGateCompleteness"] == 1.0
    assert (output / "final_report.md").is_file()
    assert (output / "run_metadata.json").is_file()
    assert (output / "figures" / "retrieval_metrics.png").is_file()


def testSmokeDeterministicAcrossRuns(tmpRepo):
    firstOutput = tmpRepo / "reports_smoke_a"
    secondOutput = tmpRepo / "reports_smoke_b"
    runExperiment(rootDir=tmpRepo, outputDir=firstOutput, smoke=True, runId="smoke-a")
    runExperiment(rootDir=tmpRepo, outputDir=secondOutput, smoke=True, runId="smoke-b")
    first = json.loads((firstOutput / "metrics" / "summary.json").read_text(encoding="utf-8"))
    second = json.loads((secondOutput / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert _stripTimings(first) == _stripTimings(second)


def testSmokeDoesNotOverwriteFullResults(tmpRepo):
    fullOutput = tmpRepo / "reports"
    smokeOutput = tmpRepo / "reports_smoke"
    runExperiment(rootDir=tmpRepo, outputDir=fullOutput, smoke=False, runId="full-1")
    runExperiment(rootDir=tmpRepo, outputDir=smokeOutput, smoke=True, runId="smoke-1")
    fullSummary = json.loads((fullOutput / "metrics" / "summary.json").read_text(encoding="utf-8"))
    assert fullSummary["workflow"]["numScenarios"] == 6
    smokeSummary = json.loads(
        (smokeOutput / "metrics" / "summary.json").read_text(encoding="utf-8")
    )
    assert smokeSummary["workflow"]["numScenarios"] == 2


def testSmokeForcesScriptedProvider(tmpRepo):
    output = tmpRepo / "reports_smoke_forced"
    # Passing a live provider name with smoke=True must still produce a
    # deterministic scripted run without any API key or network access.
    runExperiment(
        rootDir=tmpRepo,
        outputDir=output,
        smoke=True,
        providerName="deepseek",
        runId="smoke-forced",
    )
    metadata = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["provider"] == "scripted"
