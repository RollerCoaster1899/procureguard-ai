"""Unit tests for robust DeepSeek JSON decision parsing."""

from __future__ import annotations

import pytest

from procureguard.workflow.deepseek import parseDecisionJson
from procureguard.workflow.provider import ProviderError


def testParsePlainJson():
    parsed = parseDecisionJson('{"action": "approve", "riskFlag": false}')
    assert parsed["action"] == "approve"
    assert parsed["riskFlag"] is False


def testParseWithCodeFences():
    content = '```json\n{"action": "reject", "rationale": "high risk"}\n```'
    parsed = parseDecisionJson(content)
    assert parsed["action"] == "reject"


def testParseWithSurroundingText():
    content = (
        'Sure! Here is the decision:\n{"action": "review", "citations": ["a", "b"]}\nHope it helps.'
    )
    parsed = parseDecisionJson(content)
    assert parsed["action"] == "review"
    assert parsed["citations"] == ["a", "b"]


def testParseNoneRaises():
    with pytest.raises(ProviderError):
        parseDecisionJson(None)
    with pytest.raises(ProviderError):
        parseDecisionJson("   ")


def testParseInvalidRaises():
    with pytest.raises(ProviderError):
        parseDecisionJson("this is not json at all")


def testParseNonDictJsonRaises():
    with pytest.raises(ProviderError):
        parseDecisionJson('["not", "a", "dict"]')
