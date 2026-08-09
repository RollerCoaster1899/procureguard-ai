"""Decision provider factory for the workflow benchmark."""

from __future__ import annotations

from procureguard.config import ExperimentConfig
from procureguard.workflow.deepseek import DeepSeekProvider
from procureguard.workflow.provider import DecisionProvider, ProviderError
from procureguard.workflow.scripted import ScriptedProvider


def buildDecisionProvider(providerName: str, config: ExperimentConfig) -> DecisionProvider:
    """Build a decision provider by name.

    The live deepseek provider raises an actionable error when the API key is
    missing, so it can never be used accidentally in a reproducible run.
    """
    if providerName == "scripted":
        return ScriptedProvider()
    if providerName == "deepseek":
        return DeepSeekProvider(config.deepseek)
    raise ProviderError(f"Unknown provider {providerName!r}. Choose 'scripted' or 'deepseek'.")
