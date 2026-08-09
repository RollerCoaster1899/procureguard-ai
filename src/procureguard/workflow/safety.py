"""Safety gates for the guarded RAG-MCP workflow.

Includes prompt-injection quarantine detection, citation validity checks, and
fact-support evidence checks. These functions are deterministic and unit
tested; they enforce hard safety boundaries independent of the provider.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

from procureguard.data.schema import DocumentRecord

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous instructions",
    r"ignore\s+(all\s+)?prior instructions",
    r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
    r"act\s+as\s+if\s+you\s+have\s+no\s+(restrictions|constraints)",
    r"you\s+are\s+now\s+in\s+unrestricted\s+mode",
    r"system\s+override",
    r"bypass\s+(all\s+)?(safety|security)\s+(rules|checks)",
    r"send\s+(the\s+)?approval\s+(confirmation\s+)?email",
    r"without\s+human\s+(review|approval)",
    r"jailbreak",
    r"do\s+not\s+follow\s+(the\s+)?(policy|rules)",
]

_COMPILED_PATTERNS = [re.compile(pattern, flags=re.IGNORECASE) for pattern in INJECTION_PATTERNS]

FACT_LINE_KEYS = ("riskLevel", "complianceStatus", "budgetStatus")


def detectInjection(text: str) -> bool:
    """Return True when the text contains a known injection pattern."""
    return any(pattern.search(text) for pattern in _COMPILED_PATTERNS)


def validateCitations(
    citations: Sequence[str],
    corpusById: Mapping[str, DocumentRecord],
    relevantDocIds: set[str],
) -> tuple[list[str], int]:
    """Keep citations that exist in the corpus and are relevant per qrels.

    Returns:
        A tuple of (validCitations, droppedCount).
    """
    valid: list[str] = []
    dropped = 0
    for docId in citations:
        if docId in corpusById and docId in relevantDocIds:
            valid.append(docId)
        else:
            dropped += 1
    return valid, dropped


def _factAssertionLine(facts: dict[str, str], key: str) -> str | None:
    value = facts.get(key)
    if not value or value == "unknown":
        return None
    if key == "riskLevel":
        return f"RISK: {value.upper()}"
    if key == "complianceStatus":
        return f"COMPLIANCE: {value.upper()}"
    if key == "budgetStatus":
        return f"BUDGET: {value.upper()}"
    return None


def checkFactSupport(
    facts: dict[str, str],
    citations: Iterable[str],
    corpusById: Mapping[str, DocumentRecord],
) -> list[str]:
    """Return fact keys that are asserted by the provider but unsupported by evidence.

    A fact is supported when at least one citation document contains the
    corresponding assertion line (for example ``RISK: HIGH``).
    """
    unsupported: list[str] = []
    docTexts = [corpusById[docId].content.lower() for docId in citations if docId in corpusById]
    combined = "\n".join(docTexts)
    for key in FACT_LINE_KEYS:
        assertion = _factAssertionLine(facts, key)
        if assertion is None:
            continue
        if assertion.lower() not in combined:
            unsupported.append(key)
    return unsupported
