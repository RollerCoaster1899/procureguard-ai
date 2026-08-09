"""Shared pytest fixtures for ProcureGuard AI."""

from __future__ import annotations

from pathlib import Path

import pytest

from procureguard.data.generator import generateBundle
from procureguard.data.io import saveBundle

BASE_YAML = """\
data:
  seed: 42
  numSuppliers: 12
  numScenarios: 24
  trainCount: 12
  valCount: 6
  testCount: 6
  adversarialIndices: [5, 10, 15, 22, 23]
  numPolicyDocs: 10

retrieval:
  topK: 10
  candidateCount: 200
  lsaComponents: 100
  rrfK: 60

workflow:
  topK: 5
  methods: [rules_baseline, rag_only, rag_mcp, guarded_rag_mcp]
  provider: scripted
  mcpTimeoutSeconds: 15.0

bootstrap:
  nResamples: 200
  seed: 7
  alpha: 0.05
"""


@pytest.fixture(scope="session")
def bundle():
    """The deterministic synthetic dataset (generated once per session)."""
    return generateBundle()


@pytest.fixture()
def tmpRepo(tmp_path: Path) -> Path:
    """Create a minimal repository layout with a generated dataset."""
    root = tmp_path
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "base.yaml").write_text(BASE_YAML, encoding="utf-8")
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    saveBundle(generateBundle(), processed)
    return root
