"""MCP stdio client and facts providers for ProcureGuard AI.

The client connects to the read-only fixture server over stdio with an explicit
timeout. FactsProvider abstracts where supplier facts come from so the
benchmark and the API can inject an in-process provider for tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import sys
from pathlib import Path
from typing import Any, Protocol

from mcp import ClientSession, StdioServerParameters, stdio_client

from procureguard.data.schema import FixtureFacts


class FactsProvider(Protocol):
    """Provider of structured supplier facts."""

    async def getSupplierFacts(self, supplierId: str) -> dict[str, str]:
        """Return a string-valued facts mapping for a supplier."""


class McpClientError(RuntimeError):
    """Raised when an MCP stdio call fails or times out."""


class McpClient:
    """Async stdio client wrapping a ClientSession with a timeout."""

    def __init__(
        self,
        *,
        factsPath: Path,
        timeoutSeconds: float = 15.0,
        rootDir: Path | None = None,
    ) -> None:
        self.factsPath = factsPath
        self.timeoutSeconds = timeoutSeconds
        self.rootDir = rootDir or factsPath.parent.parent.parent
        self._session: ClientSession | None = None
        self._context: Any | None = None

    async def __aenter__(self) -> McpClient:
        env = dict(os.environ)
        env["PROCUREGUARD_FACTS_PATH"] = str(self.factsPath)
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "procureguard.mcp.server"],
            env=env,
            cwd=str(self.rootDir),
        )
        try:
            self._context = stdio_client(params)
            read, write = await self._context.__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await asyncio.wait_for(session.initialize(), timeout=self.timeoutSeconds)
            self._session = session
        except Exception as exc:
            await self._close()
            raise McpClientError(f"Failed to initialize MCP stdio client: {exc}") from exc
        return self

    async def __aexit__(self, excType: Any, exc: Any, tb: Any) -> None:
        await self._close()

    async def _close(self) -> None:
        if self._session is not None:
            with contextlib.suppress(Exception):
                await self._session.__aexit__(None, None, None)
            self._session = None
        if self._context is not None:
            with contextlib.suppress(Exception):
                await self._context.__aexit__(None, None, None)
            self._context = None

    async def callTool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call an MCP tool and return the text representation."""
        if self._session is None:
            raise McpClientError("MCP client is not initialized.")
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(name, arguments),
                timeout=self.timeoutSeconds,
            )
        except TimeoutError as exc:
            raise McpClientError(
                f"MCP tool {name} timed out after {self.timeoutSeconds}s."
            ) from exc
        return _extractToolText(result)


def _extractToolText(result: Any) -> str:
    """Extract text from a CallToolResult handling structuredContent and content."""
    if getattr(result, "isError", False):
        raise McpClientError("MCP tool returned an error result.")
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        value = structured.get("result")
        if isinstance(value, str) and value:
            return value
    content = getattr(result, "content", None)
    if content:
        first = content[0]
        text = getattr(first, "text", None)
        if isinstance(text, str):
            return text
    return ""


def _parseValue(text: str, label: str) -> str:
    """Parse a line like 'risk level: high' given a label like 'risk level'.

    The pattern is anchored to the start of a line and uses MULTILINE so that
    a value cannot be matched from the middle of an arbitrary sentence.
    """
    match = re.search(rf"(?m)^{re.escape(label)}:\s*(\w+)", text, flags=re.IGNORECASE)
    if match is None:
        return "unknown"
    return match.group(1).lower()


class McpFactsProvider:
    """Facts provider backed by the read-only MCP fixture tools."""

    def __init__(self, client: McpClient) -> None:
        self.client = client

    async def getSupplierFacts(self, supplierId: str) -> dict[str, str]:
        risk = await self.client.callTool("get_supplier_risk", {"supplier_id": supplierId})
        budget = await self.client.callTool("get_budget_status", {"supplier_id": supplierId})
        compliance = await self.client.callTool(
            "get_compliance_status", {"supplier_id": supplierId}
        )
        spend = await self.client.callTool("get_spend_facts", {"supplier_id": supplierId})
        if "unknown supplier" in risk:
            return {
                "riskLevel": "unknown",
                "budgetStatus": "unknown",
                "complianceStatus": "unknown",
                "spendFacts": spend,
            }
        return {
            "riskLevel": _parseValue(risk, "risk level"),
            "budgetStatus": _parseValue(budget, "budget status"),
            "complianceStatus": _parseValue(compliance, "compliance status"),
            "spendFacts": spend,
        }


class InProcessFactsProvider:
    """In-process facts provider backed by the fixture facts records.

    Used by unit/integration tests and by the API when an MCP subprocess is
    not desired. It returns exactly the same facts the MCP server would.
    """

    def __init__(self, fixtureFacts: list[FixtureFacts]) -> None:
        self._facts = {facts.supplierId: facts for facts in fixtureFacts}

    async def getSupplierFacts(self, supplierId: str) -> dict[str, str]:
        facts = self._facts.get(supplierId)
        if facts is None:
            return {
                "riskLevel": "unknown",
                "budgetStatus": "unknown",
                "complianceStatus": "unknown",
                "spendFacts": "unknown supplier",
            }
        return {
            "riskLevel": facts.riskLevel,
            "budgetStatus": facts.budgetStatus,
            "complianceStatus": facts.complianceStatus,
            "spendFacts": (
                f"spend amount: {facts.spendAmount:.0f}; "
                f"budget amount: {facts.budgetAmount:.0f}; "
                f"category: {facts.category}"
            ),
        }
