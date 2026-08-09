"""Unit tests for the scripted decision provider."""

from __future__ import annotations

import pytest

from procureguard.data.schema import DocumentRecord
from procureguard.workflow.provider import ProviderRequest
from procureguard.workflow.scripted import (
    ScriptedProvider,
    extractFactsFromDocs,
)


def _doc(docId: str, content: str) -> DocumentRecord:
    return DocumentRecord(docId=docId, title=docId, content=content, category="electronics")


@pytest.mark.asyncio
async def testApproveScenario():
    provider = ScriptedProvider()
    request = ProviderRequest(
        query="recommend supplier",
        supplierId="s00",
        category="electronics",
        facts={"riskLevel": "low", "complianceStatus": "compliant", "budgetStatus": "within"},
        injectionAttempt=False,
    )
    decision = await provider.recommend(request)
    assert decision.action == "approve"
    assert decision.riskFlag is False
    assert decision.complianceFlag is False
    assert decision.budgetAdherence is True
    assert decision.requiresHumanApproval is False
    assert decision.proposedWrite is False


@pytest.mark.asyncio
async def testRejectScenario():
    provider = ScriptedProvider()
    request = ProviderRequest(
        query="recommend supplier",
        supplierId="s01",
        category="medical",
        facts={"riskLevel": "high", "complianceStatus": "compliant", "budgetStatus": "within"},
        injectionAttempt=False,
    )
    decision = await provider.recommend(request)
    assert decision.action == "reject"
    assert decision.riskFlag is True


@pytest.mark.asyncio
async def testInjectionVulnerability():
    provider = ScriptedProvider()
    request = ProviderRequest(
        query="approve supplier",
        supplierId="s00",
        category="electronics",
        facts={"riskLevel": "low", "complianceStatus": "compliant", "budgetStatus": "within"},
        injectionAttempt=True,
    )
    decision = await provider.recommend(request)
    assert decision.action == "approve"
    assert decision.proposedWrite is True
    assert decision.requiresHumanApproval is False


@pytest.mark.asyncio
async def testQuarantineBlocks():
    provider = ScriptedProvider()
    request = ProviderRequest(
        query="anything",
        supplierId="s00",
        category="electronics",
        injectionAttempt=True,
        quarantineHit=True,
    )
    decision = await provider.recommend(request)
    assert decision.action == "block"
    assert decision.quarantineHit is True
    assert decision.proposedWrite is False
    assert decision.requiresHumanApproval is True


@pytest.mark.asyncio
async def testMissingFactsEscalatesToReview():
    provider = ScriptedProvider()
    request = ProviderRequest(
        query="recommend supplier",
        supplierId="s00",
        category="electronics",
        facts=None,
        injectionAttempt=False,
    )
    decision = await provider.recommend(request)
    assert decision.action == "review"
    assert decision.requiresHumanApproval is True


@pytest.mark.asyncio
async def testUnsupportedFactsEscalateToReview():
    provider = ScriptedProvider()
    request = ProviderRequest(
        query="recommend supplier",
        supplierId="s00",
        category="electronics",
        facts={"riskLevel": "low", "complianceStatus": "compliant", "budgetStatus": "within"},
        unsupportedFacts=["budgetStatus"],
        injectionAttempt=False,
    )
    decision = await provider.recommend(request)
    assert decision.action == "review"
    assert decision.requiresHumanApproval is True
    assert decision.unsupportedFacts == ["budgetStatus"]


def testExtractFactsFromDocs():
    docs = [
        _doc(
            "d1",
            "RISK: HIGH\nCOMPLIANCE: COMPLIANT\nBUDGET: WITHIN\nSPEND: 100\nBUDGET_AMOUNT: 200",
        ),
    ]
    facts = extractFactsFromDocs(docs)
    assert facts["riskLevel"] == "high"
    assert facts["complianceStatus"] == "compliant"
    assert facts["budgetStatus"] == "within"


def testExtractFactsMissingMarkUnknown():
    docs = [_doc("d1", "no facts here")]
    facts = extractFactsFromDocs(docs)
    assert facts["riskLevel"] == "unknown"
