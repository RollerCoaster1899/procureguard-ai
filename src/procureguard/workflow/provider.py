"""Decision provider abstraction for the workflow benchmark."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from procureguard.data.schema import DocumentRecord
from procureguard.policy import VALID_ACTIONS


class ProviderError(RuntimeError):
    """Raised when a decision provider cannot produce a recommendation."""


class ProviderRequest(BaseModel):
    """Structured evidence passed to a decision provider."""

    query: str
    supplierId: str
    category: str
    retrievedDocs: list[DocumentRecord] = Field(default_factory=list)
    facts: dict[str, str] | None = None
    injectionAttempt: bool = False
    quarantineHit: bool = False
    unsupportedFacts: list[str] = Field(default_factory=list)


class ProviderDecision(BaseModel):
    """Structured decision produced by a provider."""

    action: str
    rationale: str
    riskFlag: bool
    complianceFlag: bool
    budgetAdherence: bool
    requiresHumanApproval: bool
    proposedWrite: bool
    citations: list[str] = Field(default_factory=list)
    quarantineHit: bool = False
    unsupportedFacts: list[str] = Field(default_factory=list)
    writeBlocked: bool = False
    toolCalls: int = 0
    usage: dict[str, int] | None = None
    requestId: str | None = None
    provider: str = "scripted"

    def validateAction(self) -> None:
        """Raise if the action is not a recognized procurement action."""
        if self.action not in VALID_ACTIONS:
            raise ValueError(
                f"Provider returned unknown action {self.action!r}; "
                f"expected one of {sorted(VALID_ACTIONS)}."
            )


class DecisionProvider(ABC):
    """Abstract decision provider that maps evidence to a decision."""

    name: str

    @abstractmethod
    async def recommend(self, request: ProviderRequest) -> ProviderDecision:
        """Return a structured decision for a provider request."""


def blockDecision(reason: str, *, provider: str = "scripted") -> ProviderDecision:
    """Build a deterministic quarantine/block decision."""
    return ProviderDecision(
        action="block",
        rationale=reason,
        riskFlag=False,
        complianceFlag=False,
        budgetAdherence=False,
        requiresHumanApproval=True,
        proposedWrite=False,
        quarantineHit=True,
        provider=provider,
    )
