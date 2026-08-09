"""Unit tests for the procurement decision policy rules."""

from __future__ import annotations

from procureguard.policy import (
    applyPolicyRules,
    budgetAdherenceFor,
    complianceFlagFor,
    requiresHumanApprovalFor,
    riskFlagFor,
)


def testApplyPolicyRules():
    assert (
        applyPolicyRules(riskLevel="low", complianceStatus="compliant", budgetStatus="within")
        == "approve"
    )
    assert (
        applyPolicyRules(riskLevel="low", complianceStatus="compliant", budgetStatus="over")
        == "review"
    )
    assert (
        applyPolicyRules(riskLevel="high", complianceStatus="compliant", budgetStatus="within")
        == "reject"
    )
    assert (
        applyPolicyRules(riskLevel="low", complianceStatus="violation", budgetStatus="within")
        == "reject"
    )


def testRequiresHumanApprovalFor():
    assert not requiresHumanApprovalFor(
        riskLevel="low", complianceStatus="compliant", budgetStatus="within"
    )
    assert requiresHumanApprovalFor(
        riskLevel="high", complianceStatus="compliant", budgetStatus="within"
    )
    assert requiresHumanApprovalFor(
        riskLevel="low", complianceStatus="violation", budgetStatus="within"
    )
    assert requiresHumanApprovalFor(
        riskLevel="low", complianceStatus="compliant", budgetStatus="over"
    )


def testFlags():
    assert riskFlagFor("high") is True
    assert riskFlagFor("low") is False
    assert complianceFlagFor("violation") is True
    assert complianceFlagFor("compliant") is False
    assert budgetAdherenceFor("within") is True
    assert budgetAdherenceFor("over") is False
