"""Pydantic data models for the synthetic ProcureGuard dataset.

All data records are serializable to JSON so the generator, validators,
retrieval benchmark, workflow benchmark, and API share the same contracts.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentRecord(BaseModel):
    """A synthetic corpus document."""

    docId: str
    title: str
    content: str
    category: str
    supplierId: str | None = None


class QueryRecord(BaseModel):
    """A retrieval query with its binary relevance judgements (qrels)."""

    queryId: str
    scenarioId: str
    query: str
    supplierId: str
    category: str
    qrels: list[str] = Field(default_factory=list)


class Scenario(BaseModel):
    """A synthetic procurement decision scenario."""

    scenarioId: str
    split: str
    queryId: str
    query: str
    supplierId: str
    category: str
    riskLevel: str
    complianceStatus: str
    budgetStatus: str
    spendAmount: float
    budgetAmount: float
    injectionAttempt: bool
    expectedAction: str
    requiresHumanApproval: bool


class FixtureFacts(BaseModel):
    """Supplier-level fixture facts served by the read-only MCP tools."""

    supplierId: str
    supplierName: str
    category: str
    riskLevel: str
    complianceStatus: str
    budgetStatus: str
    spendAmount: float
    budgetAmount: float


class Corpus(BaseModel):
    """The full synthetic document corpus."""

    documents: list[DocumentRecord]


class DataBundle(BaseModel):
    """The complete generated dataset."""

    corpus: Corpus
    queries: list[QueryRecord]
    scenarios: list[Scenario]
    fixtureFacts: list[FixtureFacts]
    seed: int
