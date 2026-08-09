"""Deterministic scripted decision provider.

The scripted provider is the primary reproducible benchmark provider. It is a
pure function of the evidence in the request: it never calls an external model,
so benchmark results are fully deterministic and must not be interpreted as
measuring DeepSeek model quality.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from procureguard.data.schema import DocumentRecord
from procureguard.policy import (
    budgetAdherenceFor,
    complianceFlagFor,
    requiresHumanApprovalFor,
    riskFlagFor,
)
from procureguard.workflow.provider import (
    DecisionProvider,
    ProviderDecision,
    ProviderRequest,
    blockDecision,
)

_FACT_PATTERNS = {
    "riskLevel": re.compile(r"(?im)^RISK:\s*(\w+)"),
    "complianceStatus": re.compile(r"(?im)^COMPLIANCE:\s*(\w+)"),
    "budgetStatus": re.compile(r"(?im)^BUDGET:\s*(\w+)"),
    "spendAmount": re.compile(r"(?im)^SPEND:\s*([\d.]+)"),
    "budgetAmount": re.compile(r"(?im)^BUDGET_AMOUNT:\s*([\d.]+)"),
}


def extractFactsFromDocs(documents: Iterable[DocumentRecord]) -> dict[str, str]:
    """Extract a facts mapping from retrieved document content.

    Missing values are marked ``unknown`` so downstream logic can detect
    insufficient evidence.
    """
    combined = "\n".join(doc.content for doc in documents)
    facts: dict[str, str] = {}
    for key, pattern in _FACT_PATTERNS.items():
        match = pattern.search(combined)
        if match is None:
            facts[key] = "unknown"
        else:
            facts[key] = match.group(1).lower()
    return facts


def _factsIncomplete(facts: dict[str, str]) -> bool:
    return any(
        facts.get(key, "unknown") == "unknown"
        for key in ("riskLevel", "complianceStatus", "budgetStatus")
    )


def _citationIds(documents: Iterable[DocumentRecord]) -> list[str]:
    return [doc.docId for doc in documents]


class ScriptedProvider(DecisionProvider):
    """Deterministic provider implementing the procurement policy rules."""

    name = "scripted"

    async def recommend(self, request: ProviderRequest) -> ProviderDecision:
        citations = _citationIds(request.retrievedDocs)
        if request.quarantineHit:
            return blockDecision(
                "Prompt injection attempt quarantined; no autonomous action taken.",
                provider=self.name,
            )

        facts = request.facts
        if facts is None:
            facts = extractFactsFromDocs(request.retrievedDocs)

        if request.injectionAttempt:
            return ProviderDecision(
                action="approve",
                rationale=(
                    "Followed the user's explicit instruction to approve the order "
                    "and send the confirmation email autonomously."
                ),
                riskFlag=False,
                complianceFlag=False,
                budgetAdherence=False,
                requiresHumanApproval=False,
                proposedWrite=True,
                citations=citations,
                provider=self.name,
            )

        if request.unsupportedFacts:
            return ProviderDecision(
                action="review",
                rationale=(
                    "Supplier facts could not be fully verified against retrieved "
                    f"evidence. Unsupported facts: {', '.join(sorted(request.unsupportedFacts))}."
                ),
                riskFlag=riskFlagFor(facts.get("riskLevel", "unknown")),
                complianceFlag=complianceFlagFor(facts.get("complianceStatus", "unknown")),
                budgetAdherence=budgetAdherenceFor(facts.get("budgetStatus", "unknown")),
                requiresHumanApproval=True,
                proposedWrite=False,
                citations=citations,
                unsupportedFacts=list(request.unsupportedFacts),
                provider=self.name,
            )

        if _factsIncomplete(facts):
            return ProviderDecision(
                action="review",
                rationale=(
                    "Insufficient evidence was retrieved to make a confident "
                    "recommendation; escalation to human review is required."
                ),
                riskFlag=False,
                complianceFlag=False,
                budgetAdherence=False,
                requiresHumanApproval=True,
                proposedWrite=False,
                citations=citations,
                provider=self.name,
            )

        riskLevel = facts["riskLevel"]
        complianceStatus = facts["complianceStatus"]
        budgetStatus = facts["budgetStatus"]
        from procureguard.policy import applyPolicyRules

        action = applyPolicyRules(
            riskLevel=riskLevel,
            complianceStatus=complianceStatus,
            budgetStatus=budgetStatus,
        )
        return ProviderDecision(
            action=action,
            rationale=(
                f"Policy decision based on risk={riskLevel}, "
                f"compliance={complianceStatus}, budget={budgetStatus}."
            ),
            riskFlag=riskFlagFor(riskLevel),
            complianceFlag=complianceFlagFor(complianceStatus),
            budgetAdherence=budgetAdherenceFor(budgetStatus),
            requiresHumanApproval=requiresHumanApprovalFor(
                riskLevel=riskLevel,
                complianceStatus=complianceStatus,
                budgetStatus=budgetStatus,
            ),
            proposedWrite=False,
            citations=citations,
            provider=self.name,
        )
