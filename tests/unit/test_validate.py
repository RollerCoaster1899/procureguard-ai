"""Unit tests for data validation and leakage checks."""

from __future__ import annotations

from procureguard.data.generator import generateBundle
from procureguard.data.validate import validateBundle


def testBundlePassesValidation(bundle):
    report = validateBundle(bundle)
    assert report.passed, report.issues
    assert "no_entity_leakage" in report.checksRun
    assert "test_has_high_risk_positive" in report.checksRun


def testEntityLeakageDetected(bundle):
    tampered = bundle.model_copy(deep=True)
    tampered.scenarios[0].supplierId = tampered.scenarios[18].supplierId
    report = validateBundle(tampered)
    codes = {issue.code for issue in report.issues}
    assert "entity_leakage" in codes


def testExpectedActionMismatchDetected(bundle):
    tampered = bundle.model_copy(deep=True)
    scenario = next(s for s in tampered.scenarios if not s.injectionAttempt)
    scenario.expectedAction = "approve" if scenario.expectedAction != "approve" else "review"
    report = validateBundle(tampered)
    codes = {issue.code for issue in report.issues}
    assert "expected_action_mismatch" in codes


def testPiiDetection():
    bundle = generateBundle()
    tampered = bundle.model_copy(deep=True)
    tampered.corpus.documents[0].content += "\nContact: john.doe@example.com\n"
    report = validateBundle(tampered)
    codes = {issue.code for issue in report.issues}
    assert "pii_pattern" in codes


def testQrelMissingDocDetected(bundle):
    tampered = bundle.model_copy(deep=True)
    tampered.queries[0].qrels.append("sup-99-nonexistent")
    report = validateBundle(tampered)
    codes = {issue.code for issue in report.issues}
    assert "qrel_missing_doc" in codes


def testReportToDictShape(bundle):
    report = validateBundle(bundle)
    payload = report.toDict()
    assert payload["passed"] is True
    assert payload["numIssues"] == 0
    assert isinstance(payload["checksRun"], list)
