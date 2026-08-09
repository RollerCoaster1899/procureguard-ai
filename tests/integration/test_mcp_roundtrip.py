"""Integration tests for the MCP stdio roundtrip."""

from __future__ import annotations

import asyncio

import pytest

from procureguard.data.io import saveBundle
from procureguard.mcp.client import (
    InProcessFactsProvider,
    McpClient,
    McpClientError,
    McpFactsProvider,
)


@pytest.mark.asyncio
async def testMcpStdioRoundtrip(tmp_path, bundle):
    processed = tmp_path / "data" / "processed"
    saveBundle(bundle, processed)
    factsPath = processed / "fixture_facts.json"
    async with McpClient(factsPath=factsPath, timeoutSeconds=10, rootDir=tmp_path) as client:
        assert (
            await client.callTool("get_supplier_risk", {"supplier_id": "s00"}) == "risk level: low"
        )
        assert (
            await client.callTool("get_budget_status", {"supplier_id": "s00"})
            == "budget status: within"
        )
        assert (
            await client.callTool("get_compliance_status", {"supplier_id": "s00"})
            == "compliance status: compliant"
        )
        spend = await client.callTool("get_spend_facts", {"supplier_id": "s00"})
        assert "spend amount" in spend


@pytest.mark.asyncio
async def testMcpUnknownSupplier(tmp_path, bundle):
    processed = tmp_path / "data" / "processed"
    saveBundle(bundle, processed)
    async with McpClient(
        factsPath=processed / "fixture_facts.json", timeoutSeconds=10, rootDir=tmp_path
    ) as client:
        assert (
            await client.callTool("get_supplier_risk", {"supplier_id": "nope"})
            == "unknown supplier: nope"
        )


@pytest.mark.asyncio
async def testMcpFactsProvider(tmp_path, bundle):
    processed = tmp_path / "data" / "processed"
    saveBundle(bundle, processed)
    async with McpClient(
        factsPath=processed / "fixture_facts.json", timeoutSeconds=10, rootDir=tmp_path
    ) as client:
        provider = McpFactsProvider(client)
        facts = await provider.getSupplierFacts("s08")
        assert facts["riskLevel"] == "low"
        assert facts["complianceStatus"] == "compliant"
        assert facts["budgetStatus"] == "within"


@pytest.mark.asyncio
async def testMcpClientCallBeforeInitFails(tmp_path):
    client = McpClient(factsPath=tmp_path / "x.json", timeoutSeconds=1, rootDir=tmp_path)
    with pytest.raises(McpClientError):
        await client.callTool("get_supplier_risk", {"supplier_id": "s00"})


@pytest.mark.asyncio
async def testMcpMissingFactsFileFails(tmp_path):
    client = McpClient(factsPath=tmp_path / "missing.json", timeoutSeconds=5, rootDir=tmp_path)
    async with client:
        with pytest.raises(McpClientError):
            await client.callTool("get_supplier_risk", {"supplier_id": "s00"})


def testInProcessFactsProvider(bundle):
    provider = InProcessFactsProvider(bundle.fixtureFacts)
    facts = asyncio.run(provider.getSupplierFacts("s08"))
    assert facts["riskLevel"] == "low"
    assert "spend amount" in facts["spendFacts"]


def testInProcessUnknownSupplier(bundle):
    provider = InProcessFactsProvider(bundle.fixtureFacts)
    facts = asyncio.run(provider.getSupplierFacts("ghost"))
    assert facts["riskLevel"] == "unknown"
