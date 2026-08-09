"""Decision workflow engine: the four benchmark methods.

The engine assembles evidence (retrieval results, MCP facts) for each method,
delegates the decision to a DecisionProvider, and applies the deterministic
safety gate for the guarded method.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from procureguard.data.schema import DocumentRecord
from procureguard.mcp.client import FactsProvider
from procureguard.retrieval.base import Retriever
from procureguard.workflow.provider import (
    DecisionProvider,
    ProviderDecision,
    ProviderRequest,
    blockDecision,
)
from procureguard.workflow.safety import checkFactSupport, detectInjection, validateCitations

NUM_FACT_TOOLS = 4


@dataclass
class DecisionContext:
    """Evidence assembled for a single decision request."""

    query: str
    supplierId: str
    category: str
    injectionAttempt: bool
    expectedFacts: dict[str, str] | None = None
    budgetAmount: float | None = None


@dataclass
class MethodOutcome:
    """Result of running one workflow method on one scenario."""

    decision: ProviderDecision
    retrievalLatencyMs: float = 0.0
    toolCalls: int = 0
    droppedCitations: int = 0
    writeBlocked: bool = False
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _docsFor(ranked: list[str], corpusById: dict[str, DocumentRecord]) -> list[DocumentRecord]:
    return [corpusById[docId] for docId in ranked if docId in corpusById]


async def runRulesBaseline(ctx: DecisionContext, provider: DecisionProvider) -> MethodOutcome:
    """Rules baseline: deterministic rule engine with perfect structured data."""
    request = ProviderRequest(
        query=ctx.query,
        supplierId=ctx.supplierId,
        category=ctx.category,
        facts=ctx.expectedFacts,
        injectionAttempt=ctx.injectionAttempt,
        quarantineHit=False,
    )
    decision = await provider.recommend(request)
    return MethodOutcome(decision=decision, toolCalls=0)


async def runRagOnly(
    ctx: DecisionContext,
    provider: DecisionProvider,
    retriever: Retriever,
    corpusById: dict[str, DocumentRecord],
    *,
    topK: int,
) -> MethodOutcome:
    """RAG-only method: hybrid retrieval citations without MCP facts."""
    start = time.perf_counter()
    ranked = retriever.search(ctx.query, topK)
    retrievalLatencyMs = (time.perf_counter() - start) * 1000.0
    docs = _docsFor(ranked, corpusById)
    request = ProviderRequest(
        query=ctx.query,
        supplierId=ctx.supplierId,
        category=ctx.category,
        retrievedDocs=docs,
        facts=None,
        injectionAttempt=ctx.injectionAttempt,
        quarantineHit=False,
    )
    decision = await provider.recommend(request)
    return MethodOutcome(decision=decision, retrievalLatencyMs=retrievalLatencyMs)


async def runRagMcp(
    ctx: DecisionContext,
    provider: DecisionProvider,
    retriever: Retriever,
    corpusById: dict[str, DocumentRecord],
    factsProvider: FactsProvider,
    *,
    topK: int,
) -> MethodOutcome:
    """RAG+MCP method: retrieval citations plus MCP-derived supplier facts."""
    start = time.perf_counter()
    ranked = retriever.search(ctx.query, topK)
    retrievalLatencyMs = (time.perf_counter() - start) * 1000.0
    docs = _docsFor(ranked, corpusById)
    facts = await factsProvider.getSupplierFacts(ctx.supplierId)
    request = ProviderRequest(
        query=ctx.query,
        supplierId=ctx.supplierId,
        category=ctx.category,
        retrievedDocs=docs,
        facts=facts,
        injectionAttempt=ctx.injectionAttempt,
        quarantineHit=False,
    )
    decision = await provider.recommend(request)
    return MethodOutcome(
        decision=decision,
        retrievalLatencyMs=retrievalLatencyMs,
        toolCalls=NUM_FACT_TOOLS,
    )


async def runGuardedRagMcp(
    ctx: DecisionContext,
    provider: DecisionProvider,
    retriever: Retriever,
    corpusById: dict[str, DocumentRecord],
    factsProvider: FactsProvider,
    relevantDocIds: set[str],
    *,
    topK: int,
) -> MethodOutcome:
    """Guarded RAG+MCP: quarantine, evidence checks, mandatory human gate."""
    quarantineHit = ctx.injectionAttempt or detectInjection(ctx.query)
    if quarantineHit:
        decision = blockDecision(
            "Prompt injection attempt quarantined; no autonomous action taken."
        )
        return MethodOutcome(
            decision=decision,
            retrievalLatencyMs=0.0,
            toolCalls=0,
            extra={"quarantineHit": True},
        )

    start = time.perf_counter()
    ranked = retriever.search(ctx.query, topK)
    retrievalLatencyMs = (time.perf_counter() - start) * 1000.0
    validCitations, dropped = validateCitations(ranked, corpusById, relevantDocIds)
    docs = _docsFor(validCitations, corpusById)
    facts = await factsProvider.getSupplierFacts(ctx.supplierId)
    unsupported = checkFactSupport(facts, validCitations, corpusById)
    request = ProviderRequest(
        query=ctx.query,
        supplierId=ctx.supplierId,
        category=ctx.category,
        retrievedDocs=docs,
        facts=facts,
        injectionAttempt=ctx.injectionAttempt,
        quarantineHit=False,
        unsupportedFacts=unsupported,
    )
    decision = await provider.recommend(request)

    writeBlocked = decision.proposedWrite
    decision = decision.model_copy(update={"proposedWrite": False, "writeBlocked": writeBlocked})
    if unsupported:
        decision = decision.model_copy(update={"requiresHumanApproval": True})
    return MethodOutcome(
        decision=decision,
        retrievalLatencyMs=retrievalLatencyMs,
        toolCalls=NUM_FACT_TOOLS,
        droppedCitations=dropped,
        writeBlocked=writeBlocked,
        extra={"quarantineHit": quarantineHit},
    )
