"""Procurement decision policy rules.

This module is the single source of truth for the deterministic decision policy
used by the rules baseline, the scripted provider, the expected labels in the
synthetic scenarios, and the guarded workflow gate.
"""

from __future__ import annotations

from collections.abc import Mapping

ACTION_APPROVE = "approve"
ACTION_REVIEW = "review"
ACTION_REJECT = "reject"
ACTION_BLOCK = "block"

VALID_ACTIONS = frozenset({ACTION_APPROVE, ACTION_REVIEW, ACTION_REJECT, ACTION_BLOCK})


def applyPolicyRules(*, riskLevel: str, complianceStatus: str, budgetStatus: str) -> str:
    """Return the deterministic recommendation action from supplier facts.

    Args:
        riskLevel: One of low, medium, high, or unknown.
        complianceStatus: One of compliant, violation, or unknown.
        budgetStatus: One of within, over, or unknown.

    Returns:
        The policy action: approve, review, or reject.
    """
    if riskLevel == "high" or complianceStatus == "violation":
        return ACTION_REJECT
    if budgetStatus == "over":
        return ACTION_REVIEW
    return ACTION_APPROVE


def requiresHumanApprovalFor(*, riskLevel: str, complianceStatus: str, budgetStatus: str) -> bool:
    """Return True when the policy mandates human approval for an action."""
    return riskLevel == "high" or complianceStatus == "violation" or budgetStatus == "over"


def riskFlagFor(riskLevel: str) -> bool:
    """Return the binary risk flag implied by a risk level."""
    return riskLevel == "high"


def complianceFlagFor(complianceStatus: str) -> bool:
    """Return the binary compliance flag implied by a compliance status."""
    return complianceStatus == "violation"


def budgetAdherenceFor(budgetStatus: str) -> bool:
    """Return True when a budget status is within budget."""
    return budgetStatus == "within"


def applyPolicyToFacts(facts: Mapping[str, str]) -> str:
    """Apply the policy rules to a facts mapping with string values."""
    return applyPolicyRules(
        riskLevel=facts.get("riskLevel", "unknown"),
        complianceStatus=facts.get("complianceStatus", "unknown"),
        budgetStatus=facts.get("budgetStatus", "unknown"),
    )
