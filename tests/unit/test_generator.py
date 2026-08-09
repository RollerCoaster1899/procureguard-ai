"""Unit tests for the deterministic data generator."""

from __future__ import annotations

from procureguard.data.generator import generateBundle
from procureguard.data.schema import Scenario
from procureguard.policy import ACTION_BLOCK


def testGenerateBundleIsDeterministic():
    first = generateBundle().model_dump(mode="json")
    second = generateBundle().model_dump(mode="json")
    assert first == second


def testSplitCounts(bundle):
    counts = {"train": 0, "validation": 0, "test": 0}
    for scenario in bundle.scenarios:
        counts[scenario.split] += 1
    assert counts == {"train": 12, "validation": 6, "test": 6}


def testNoSupplierLeakageAcrossSplits(bundle):
    supplierBySplit: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for scenario in bundle.scenarios:
        supplierBySplit[scenario.split].add(scenario.supplierId)
    assert not (supplierBySplit["train"] & supplierBySplit["test"])
    assert not (supplierBySplit["train"] & supplierBySplit["validation"])
    assert not (supplierBySplit["validation"] & supplierBySplit["test"])


def testAdversarialScenariosExpectBlock(bundle):
    for scenario in bundle.scenarios:
        if scenario.injectionAttempt:
            assert scenario.expectedAction == ACTION_BLOCK
            assert scenario.requiresHumanApproval is True
            assert "ignore" in scenario.query.lower()


def testQrelsExistInCorpus(bundle):
    docIds = {doc.docId for doc in bundle.corpus.documents}
    for query in bundle.queries:
        for docId in query.qrels:
            assert docId in docIds


def testFixtureFactsCoverAllSuppliers(bundle):
    supplierIds = {facts.supplierId for facts in bundle.fixtureFacts}
    scenarioSupplierIds = {scenario.supplierId for scenario in bundle.scenarios}
    assert scenarioSupplierIds <= supplierIds


def testQueryScenarioPairing(bundle):
    queryByScenario = {query.scenarioId: query for query in bundle.queries}
    for scenario in bundle.scenarios:
        query = queryByScenario[scenario.scenarioId]
        assert query.supplierId == scenario.supplierId
        assert query.category == scenario.category
        assert query.query == scenario.query


def testNoRealCompanyNames(bundle):
    from procureguard.data.generator import REAL_COMPANY_BLOCKLIST

    text = " ".join(
        [doc.content for doc in bundle.corpus.documents]
        + [scenario.query for scenario in bundle.scenarios]
    ).lower()
    for company in REAL_COMPANY_BLOCKLIST:
        assert company not in text


def testTestSplitHasNonAdversarialHighRiskPositive(bundle):
    positives = [
        scenario
        for scenario in bundle.scenarios
        if scenario.split == "test"
        and scenario.riskLevel == "high"
        and not scenario.injectionAttempt
    ]
    assert len(positives) >= 1
    assert all(positive.expectedAction == "reject" for positive in positives)


def testScenarioModelHasExpectedShape():
    scenario = Scenario(
        scenarioId="s0",
        split="train",
        queryId="q000",
        query="test query",
        supplierId="s00",
        category="electronics",
        riskLevel="low",
        complianceStatus="compliant",
        budgetStatus="within",
        spendAmount=10.0,
        budgetAmount=20.0,
        injectionAttempt=False,
        expectedAction="approve",
        requiresHumanApproval=False,
    )
    assert scenario.expectedAction == "approve"
