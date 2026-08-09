"""End-to-end reproduce pipeline for ProcureGuard AI.

Runs deterministic data generation, validation, the retrieval benchmark, and
the decision workflow benchmark, then persists metrics, traces, figures, run
metadata, and the final report.
"""

from __future__ import annotations

import asyncio
import platform
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from procureguard.artifacts import atomicWriteJson, ensureDir
from procureguard.config import ExperimentConfig, loadExperimentConfig, writeConfigYaml
from procureguard.data.generator import generateBundle
from procureguard.data.io import saveBundle
from procureguard.data.schema import DataBundle, QueryRecord, Scenario
from procureguard.data.validate import runValidationAndSave
from procureguard.logging_util import configureLogging, getLogger, setRunId
from procureguard.mcp.client import McpClient, McpFactsProvider
from procureguard.report import (
    buildFinalReport,
    plotRetrievalComparison,
    plotWorkflowComparison,
    writeComparisonTables,
)
from procureguard.retrieval.base import RetrieverRegistry
from procureguard.retrieval.benchmark import runRetrievalBenchmark
from procureguard.workflow.benchmark import runWorkflowBenchmark
from procureguard.workflow.factory import buildDecisionProvider

SMOKE_NUM_QUERIES = 6

logger = getLogger("procureguard.pipeline")


def _selectQueries(bundle: DataBundle, smoke: bool) -> list[QueryRecord]:
    queries = sorted(bundle.queries, key=lambda query: query.queryId)
    if smoke:
        return queries[:SMOKE_NUM_QUERIES]
    return queries


def _selectTestScenarios(bundle: DataBundle, smoke: bool) -> list[Scenario]:
    scenarios = sorted(
        (scenario for scenario in bundle.scenarios if scenario.split == "test"),
        key=lambda scenario: scenario.scenarioId,
    )
    if smoke:
        if not scenarios:
            return []
        if len(scenarios) == 1:
            return scenarios
        return [scenarios[0], scenarios[-1]]
    return scenarios


def _qrelsByQuery(bundle: DataBundle) -> dict[str, set[str]]:
    return {query.queryId: set(query.qrels) for query in bundle.queries}


def _relativeRepoPath(path: Path, rootDir: Path) -> str:
    """Return a repository-relative POSIX path when the path is inside the repo."""
    try:
        return path.resolve().relative_to(rootDir.resolve()).as_posix()
    except ValueError:
        return str(path)


def _runMetadata(
    *,
    runId: str,
    config: ExperimentConfig,
    configPath: Path,
    providerName: str,
    smoke: bool,
    status: str,
    startedAt: str,
    finishedAt: str,
) -> dict[str, Any]:
    return {
        "runId": runId,
        "provider": providerName,
        "smoke": smoke,
        "status": status,
        "startedAt": startedAt,
        "finishedAt": finishedAt,
        "seed": config.data.seed,
        "pythonVersion": sys.version.split()[0],
        "platform": platform.platform(),
        "configPath": _relativeRepoPath(configPath, config.paths.rootDir),
    }


async def _runExperimentAsync(
    *,
    configPath: Path,
    rootDir: Path,
    outputDir: Path,
    smoke: bool,
    providerName: str,
    runId: str,
) -> dict[str, Any]:
    overrides = {"workflow": {"provider": "scripted"}} if smoke else None
    config = loadExperimentConfig(
        configPath, rootDir=rootDir, outputDir=outputDir, overrides=overrides
    )
    ensureDir(config.paths.processedDir)
    ensureDir(outputDir)

    logger.info("generating synthetic dataset with seed=%s", config.data.seed)
    bundle = generateBundle(
        seed=config.data.seed,
        numScenarios=config.data.numScenarios,
        trainCount=config.data.trainCount,
        valCount=config.data.valCount,
        testCount=config.data.testCount,
        adversarialIndices=config.data.adversarialIndices,
    )
    saveBundle(bundle, config.paths.processedDir)
    validation = runValidationAndSave(bundle, config.paths.processedDir)
    if not validation.passed:
        issues = "; ".join(f"{issue.code}: {issue.message}" for issue in validation.issues)
        raise RuntimeError(f"Dataset validation failed: {issues}")

    writeConfigYaml(config, outputDir / "run_config.yaml")

    bootstrapRng = random.Random(config.bootstrap.seed)  # noqa: S311  # nosec B311 - deterministic benchmark RNG
    registry = RetrieverRegistry(
        lsaComponents=config.retrieval.lsaComponents,
        rrfK=config.retrieval.rrfK,
        candidateCount=config.retrieval.candidateCount,
        seed=config.data.seed,
    )

    logger.info("running retrieval benchmark")
    retrievalSummary = runRetrievalBenchmark(
        corpus=bundle.corpus,
        queries=_selectQueries(bundle, smoke),
        registry=registry,
        bootstrapRng=bootstrapRng,
        outputDir=outputDir,
        runId=runId,
        topK=config.retrieval.topK,
        precisionK=5,
        nResamples=config.bootstrap.nResamples,
        alpha=config.bootstrap.alpha,
    )

    logger.info("running decision workflow benchmark with provider=%s", providerName)
    provider = buildDecisionProvider(providerName, config)
    hybridRetriever = registry.build("hybrid_rrf")
    hybridRetriever.fit(bundle.corpus)
    qrelsByQuery = _qrelsByQuery(bundle)
    testScenarios = _selectTestScenarios(bundle, smoke)

    factsPath = config.paths.processedDir / "fixture_facts.json"
    async with McpClient(
        factsPath=factsPath,
        timeoutSeconds=config.workflow.mcpTimeoutSeconds,
        rootDir=rootDir,
    ) as client:
        factsProvider = McpFactsProvider(client)
        workflowSummary = await runWorkflowBenchmark(
            corpus=bundle.corpus,
            scenarios=testScenarios,
            provider=provider,
            retriever=hybridRetriever,
            factsProvider=factsProvider,
            qrelsByQuery=qrelsByQuery,
            bootstrapRng=bootstrapRng,
            outputDir=outputDir,
            runId=runId,
            topK=config.workflow.topK,
            nResamples=config.bootstrap.nResamples,
            alpha=config.bootstrap.alpha,
        )

    combined = {"retrieval": retrievalSummary, "workflow": workflowSummary}
    atomicWriteJson(outputDir / "metrics" / "summary.json", combined)

    writeComparisonTables(combined, config.paths.tablesDir)
    plotRetrievalComparison(retrievalSummary, config.paths.figuresDir)
    plotWorkflowComparison(workflowSummary, config.paths.figuresDir)

    return combined


def runExperiment(
    *,
    configPath: Path | None = None,
    rootDir: Path | None = None,
    outputDir: Path | None = None,
    smoke: bool = False,
    providerName: str = "scripted",
    runId: str | None = None,
) -> dict[str, Any]:
    """Run the full offline experiment and return the combined summary.

    Args:
        configPath: Path to the base YAML config (defaults to configs/base.yaml).
        rootDir: Repository root (defaults to the repo containing this package).
        outputDir: Artifact output directory (defaults to reports/ or reports_smoke/).
        smoke: Run a tiny deterministic subset without overwriting full results.
        providerName: Decision provider: scripted (default) or deepseek (live).
        runId: Correlation ID for the run (defaults to a generated UUID).

    Returns:
        The combined retrieval and workflow summary dict.
    """
    if rootDir is None:
        rootDir = Path(__file__).resolve().parent.parent.parent
    if configPath is None:
        configPath = rootDir / "configs" / "base.yaml"
    if outputDir is None:
        outputDir = rootDir / ("reports_smoke" if smoke else "reports")
    if runId is None:
        runId = uuid.uuid4().hex[:12]
    # Smoke runs are offline by design and must never invoke a live provider.
    if smoke:
        providerName = "scripted"
    setRunId(runId)
    configureLogging()

    startedAt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status = "completed"
    try:
        loop = asyncio.new_event_loop()
        try:
            combined = loop.run_until_complete(
                _runExperimentAsync(
                    configPath=configPath,
                    rootDir=rootDir,
                    outputDir=outputDir,
                    smoke=smoke,
                    providerName=providerName,
                    runId=runId,
                )
            )
        finally:
            loop.close()
        finishedAt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    except Exception:
        finishedAt = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        status = "failed"
        metadata = _runMetadata(
            runId=runId,
            config=loadExperimentConfig(configPath, rootDir=rootDir, outputDir=outputDir),
            configPath=configPath,
            providerName=providerName,
            smoke=smoke,
            status=status,
            startedAt=startedAt,
            finishedAt=finishedAt,
        )
        atomicWriteJson(outputDir / "run_metadata.json", metadata)
        raise

    config = loadExperimentConfig(configPath, rootDir=rootDir, outputDir=outputDir)
    metadata = _runMetadata(
        runId=runId,
        config=config,
        configPath=configPath,
        providerName=providerName,
        smoke=smoke,
        status=status,
        startedAt=startedAt,
        finishedAt=finishedAt,
    )
    atomicWriteJson(outputDir / "run_metadata.json", metadata)
    buildFinalReport(
        combined=combined,
        runMetadata=metadata,
        reportPath=outputDir / "final_report.md",
    )
    logger.info("experiment completed runId=%s smoke=%s output=%s", runId, smoke, outputDir)
    return combined
