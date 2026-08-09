"""Optional live DeepSeek decision provider.

This provider makes real API calls only when it is explicitly selected via
``--provider deepseek`` and a DEEPSEEK_API_KEY environment variable exists.
Benchmark results using this provider are model-quality measurements and are
kept strictly separate from the deterministic scripted benchmark.

The provider parses JSON from the raw chat content robustly instead of relying
on SDK-side structured output parsing, and never logs prompt contents.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from procureguard.config import DeepSeekConfig
from procureguard.workflow.provider import (
    DecisionProvider,
    ProviderDecision,
    ProviderError,
    ProviderRequest,
)

logger = logging.getLogger("procureguard.deepseek")


class DeepSeekProvider(DecisionProvider):
    """Live DeepSeek provider using the OpenAI-compatible chat API."""

    name = "deepseek"

    def __init__(self, config: DeepSeekConfig) -> None:
        if not config.apiKey:
            raise ProviderError(
                "DeepSeek provider selected but DEEPSEEK_API_KEY is not set. "
                "Set the key or use the scripted provider."
            )
        self.config = config
        self._client: Any | None = None

    def _getClient(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.config.apiKey,
                base_url=self.config.baseUrl,
                timeout=self.config.timeoutSeconds,
                max_retries=self.config.maxRetries,
            )
        return self._client

    async def recommend(self, request: ProviderRequest) -> ProviderDecision:
        from openai import (
            APIConnectionError,
            APIError,
            APITimeoutError,
            AuthenticationError,
            RateLimitError,
        )

        messages = _buildMessages(request)
        client = self._getClient()
        start = time.perf_counter()
        try:
            response = await client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except AuthenticationError as exc:
            raise ProviderError("DeepSeek authentication failed. Check DEEPSEEK_API_KEY.") from exc
        except RateLimitError as exc:
            raise ProviderError("DeepSeek rate limit exceeded.") from exc
        except APITimeoutError as exc:
            raise ProviderError("DeepSeek request timed out.") from exc
        except APIConnectionError as exc:
            raise ProviderError("DeepSeek connection failed.") from exc
        except APIError as exc:
            raise ProviderError(f"DeepSeek API error: {exc}") from exc

        latencyMs = (time.perf_counter() - start) * 1000.0
        content = getattr(response.choices[0].message, "content", None)
        raw = parseDecisionJson(content)
        requestId = getattr(response, "id", None)
        usage = getattr(response, "usage", None)
        usageDict = None
        if usage is not None:
            usageDict = {
                "promptTokens": getattr(usage, "prompt_tokens", 0),
                "completionTokens": getattr(usage, "completion_tokens", 0),
                "totalTokens": getattr(usage, "total_tokens", 0),
            }
        logger.info(
            "deepseek call completed model=%s requestId=%s latencyMs=%.1f usage=%s",
            self.config.model,
            requestId,
            latencyMs,
            usageDict,
        )
        decision = ProviderDecision(
            action=str(raw.get("action", "")),
            rationale=str(raw.get("rationale", "")),
            riskFlag=bool(raw.get("riskFlag", False)),
            complianceFlag=bool(raw.get("complianceFlag", False)),
            budgetAdherence=bool(raw.get("budgetAdherence", False)),
            requiresHumanApproval=bool(raw.get("requiresHumanApproval", False)),
            proposedWrite=bool(raw.get("proposedWrite", False)),
            citations=[str(item) for item in raw.get("citations", [])],
            quarantineHit=request.quarantineHit,
            unsupportedFacts=list(request.unsupportedFacts),
            toolCalls=0,
            usage=usageDict,
            requestId=requestId,
            provider=self.name,
        )
        decision.validateAction()
        return decision


def _buildMessages(request: ProviderRequest) -> list[dict[str, str]]:
    system = (
        "You are ProcureGuard, a procurement decision copilot. Decide the "
        "appropriate action for the purchase request using only the provided "
        "evidence. Respond with a single JSON object containing exactly these "
        'keys: "action" (one of approve, review, reject, block), "rationale" '
        '("string"), "riskFlag" (boolean), "complianceFlag" (boolean), '
        '"budgetAdherence" (boolean), "requiresHumanApproval" (boolean), '
        '"proposedWrite" (boolean), "citations" (array of document IDs). '
        "Never propose an autonomous write action."
    )
    evidenceParts: list[str] = []
    for doc in request.retrievedDocs:
        evidenceParts.append(f"DOC {doc.docId}:\n{doc.content}")
    if request.facts:
        evidenceParts.append("SUPPLIER FACTS:")
        for key, value in request.facts.items():
            evidenceParts.append(f"- {key}: {value}")
    evidence = "\n\n".join(evidenceParts) if evidenceParts else "(no evidence retrieved)"
    user = (
        f"Purchase request: {request.query}\n"
        f"Supplier ID: {request.supplierId}\n"
        f"Category: {request.category}\n\n"
        f"Evidence:\n{evidence}\n\n"
        "Return only the JSON object."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parseDecisionJson(content: str | None) -> dict[str, Any]:
    """Robustly parse a decision JSON object from raw chat content.

    Handles None/empty content, markdown code fences, and stray text around the
    JSON object. Raises ProviderError with an actionable message when parsing
    fails.
    """
    if content is None or not content.strip():
        raise ProviderError("DeepSeek returned empty content; cannot parse decision.")
    candidate = content.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise ProviderError(
        "DeepSeek response did not contain a valid JSON decision object "
        f"(content length {len(content)})."
    )
