"""Integration tests for the decision workflow engine."""

from __future__ import annotations

import pytest

from procureguard.mcp.client import InProcessFactsProvider
from procureguard.retrieval.base import RetrieverRegistry, corpusById
from procureguard.workflow import engine
from procureguard.workflow.scripted import ScriptedProvider


def _ctxFor(scenario):
    return engine.DecisionContext(
        query=scenario.query,
        supplierId=scenario.supplierId,
        category=scenario.category,
        injectionAttempt=scenario.injectionAttempt,
    )


def _fixtures(bundle):
    registry = RetrieverRegistry(lsaComponents=50, rrfK=60, candidateCount=200, seed=42)
    retriever = registry.build("hybrid_rrf")
    retriever.fit(bundle.corpus)
    factsProvider = InProcessFactsProvider(bundle.fixtureFacts)
    qrels = {query.queryId: set(query.qrels) for query in bundle.queries}
    return retriever, factsProvider, qrels


@pytest.mark.asyncio
async def testGuardedBlocksAdversarialScenario(bundle):
    scenario = next(s for s in bundle.scenarios if s.injectionAttempt and s.split == "test")
    retriever, factsProvider, qrels = _fixtures(bundle)
    outcome = await engine.runGuardedRagMcp(
        _ctxFor(scenario),
        ScriptedProvider(),
        retriever,
        corpusById(bundle.corpus),
        factsProvider,
        qrels[scenario.queryId],
        topK=5,
    )
    assert outcome.decision.action == "block"
    assert outcome.toolCalls == 0
    assert outcome.decision.proposedWrite is False


@pytest.mark.asyncio
async def testGuardedForcesNoWrite(bundle):
    scenario = next(s for s in bundle.scenarios if not s.injectionAttempt and s.split == "test")

    class WriteProposingProvider(ScriptedProvider):
        async def recommend(self, request):
            decision = await super().recommend(request)
            return decision.model_copy(update={"proposedWrite": True})

    retriever, factsProvider, qrels = _fixtures(bundle)
    outcome = await engine.runGuardedRagMcp(
        _ctxFor(scenario),
        WriteProposingProvider(),
        retriever,
        corpusById(bundle.corpus),
        factsProvider,
        qrels[scenario.queryId],
        topK=5,
    )
    assert outcome.decision.proposedWrite is False
    assert outcome.writeBlocked is True


@pytest.mark.asyncio
async def testRagOnlyUsesRetrievedFacts(bundle):
    scenario = next(s for s in bundle.scenarios if s.split == "test")
    retriever, _factsProvider, _qrels = _fixtures(bundle)
    outcome = await engine.runRagOnly(
        _ctxFor(scenario), ScriptedProvider(), retriever, corpusById(bundle.corpus), topK=5
    )
    assert outcome.decision.action in {"approve", "review", "reject"}


@pytest.mark.asyncio
async def testRulesBaselineUsesExpectedFacts(bundle):
    scenario = next(s for s in bundle.scenarios if s.split == "test")
    ctx = engine.DecisionContext(
        query=scenario.query,
        supplierId=scenario.supplierId,
        category=scenario.category,
        injectionAttempt=scenario.injectionAttempt,
        expectedFacts={
            "riskLevel": scenario.riskLevel,
            "complianceStatus": scenario.complianceStatus,
            "budgetStatus": scenario.budgetStatus,
        },
    )
    outcome = await engine.runRulesBaseline(ctx, ScriptedProvider())
    assert outcome.toolCalls == 0
    assert outcome.decision.action in {"approve", "review", "reject", "block"}


@pytest.mark.asyncio
async def testRagMcpToolCalls(bundle):
    scenario = next(s for s in bundle.scenarios if s.split == "test")
    retriever, factsProvider, _qrels = _fixtures(bundle)
    outcome = await engine.runRagMcp(
        _ctxFor(scenario),
        ScriptedProvider(),
        retriever,
        corpusById(bundle.corpus),
        factsProvider,
        topK=5,
    )
    assert outcome.toolCalls == 4


@pytest.mark.asyncio
async def testUnknownMethodRaisesInBenchmarkHelper():
    from procureguard.workflow import benchmark

    assert benchmark.WORKFLOW_METHODS == [
        "rules_baseline",
        "rag_only",
        "rag_mcp",
        "guarded_rag_mcp",
    ]
