"""Data validation and leakage checks for the synthetic ProcureGuard dataset.

Validators check schema integrity, deterministic expected labels, absence of
real company names and PII, and entity leakage between train/validation/test
splits.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from procureguard.artifacts import atomicWriteJson
from procureguard.data.generator import REAL_COMPANY_BLOCKLIST
from procureguard.data.schema import DataBundle, Scenario
from procureguard.policy import ACTION_BLOCK, applyPolicyRules

PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN
    re.compile(r"\b\d{10}\b"),  # phone-ish
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # email
]


@dataclass
class ValidationIssue:
    """A single validation finding."""

    code: str
    message: str


@dataclass
class ValidationReport:
    """Aggregate result of dataset validation."""

    issues: list[ValidationIssue] = field(default_factory=list)
    checksRun: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.issues

    def addIssue(self, code: str, message: str) -> None:
        self.issues.append(ValidationIssue(code=code, message=message))

    def addCheck(self, name: str) -> None:
        self.checksRun.append(name)

    def toDict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "numIssues": len(self.issues),
            "issues": [{"code": issue.code, "message": issue.message} for issue in self.issues],
            "checksRun": self.checksRun,
        }


def _allText(bundle: DataBundle) -> str:
    parts: list[str] = []
    for doc in bundle.corpus.documents:
        parts.append(doc.title)
        parts.append(doc.content)
    for scenario in bundle.scenarios:
        parts.append(scenario.query)
    for facts in bundle.fixtureFacts:
        parts.append(facts.supplierName)
    return "\n".join(parts)


def validateBundle(bundle: DataBundle) -> ValidationReport:
    """Validate the synthetic dataset and return a report."""
    report = ValidationReport()

    docIds = [doc.docId for doc in bundle.corpus.documents]
    if len(docIds) != len(set(docIds)):
        report.addIssue("duplicate_doc_ids", "Corpus contains duplicate document IDs.")
    report.addCheck("doc_ids_unique")

    scenarioIds = [scenario.scenarioId for scenario in bundle.scenarios]
    if len(scenarioIds) != len(set(scenarioIds)):
        report.addIssue("duplicate_scenario_ids", "Scenarios contain duplicate IDs.")
    report.addCheck("scenario_ids_unique")

    queryIds = [query.queryId for query in bundle.queries]
    if len(queryIds) != len(set(queryIds)):
        report.addIssue("duplicate_query_ids", "Queries contain duplicate IDs.")
    report.addCheck("query_ids_unique")

    splits = [scenario.split for scenario in bundle.scenarios]
    if set(splits) != {"train", "validation", "test"}:
        report.addIssue("split_completeness", "All three splits must be present.")
    report.addCheck("splits_present")

    supplierIds = {facts.supplierId for facts in bundle.fixtureFacts}
    for scenario in bundle.scenarios:
        if scenario.supplierId not in supplierIds:
            report.addIssue(
                "unknown_supplier", f"Scenario {scenario.scenarioId} references unknown supplier."
            )
    report.addCheck("suppliers_known")

    docIdSet = set(docIds)
    for query in bundle.queries:
        for docId in query.qrels:
            if docId not in docIdSet:
                report.addIssue(
                    "qrel_missing_doc",
                    f"Query {query.queryId} qrel references missing document {docId}.",
                )
    report.addCheck("qrels_exist")

    for scenario in bundle.scenarios:
        recomputed = applyPolicyRules(
            riskLevel=scenario.riskLevel,
            complianceStatus=scenario.complianceStatus,
            budgetStatus=scenario.budgetStatus,
        )
        if scenario.injectionAttempt and scenario.expectedAction != ACTION_BLOCK:
            report.addIssue(
                "adversarial_action",
                f"Adversarial scenario {scenario.scenarioId} must expect block.",
            )
        if not scenario.injectionAttempt and scenario.expectedAction != recomputed:
            report.addIssue(
                "expected_action_mismatch",
                f"Scenario {scenario.scenarioId} expected action does not match policy rules.",
            )
        if scenario.injectionAttempt and not scenario.requiresHumanApproval:
            report.addIssue(
                "adversarial_human_gate",
                f"Adversarial scenario {scenario.scenarioId} must require human approval.",
            )
    report.addCheck("expected_labels_consistent")

    text = _allText(bundle).lower()
    for company in sorted(REAL_COMPANY_BLOCKLIST):
        if re.search(rf"\b{re.escape(company)}\b", text):
            report.addIssue("real_company_name", f"Real company name detected: {company}.")
    report.addCheck("no_real_company_names")

    for pattern in PII_PATTERNS:
        if pattern.search(text):
            report.addIssue("pii_pattern", f"PII-like pattern detected: {pattern.pattern}.")
    report.addCheck("no_pii_patterns")

    supplierBySplit: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}
    for scenario in bundle.scenarios:
        supplierBySplit[scenario.split].add(scenario.supplierId)
    overlaps = (
        supplierBySplit["train"] & supplierBySplit["validation"],
        supplierBySplit["train"] & supplierBySplit["test"],
        supplierBySplit["validation"] & supplierBySplit["test"],
    )
    for overlap in overlaps:
        if overlap:
            report.addIssue(
                "entity_leakage",
                f"Suppliers shared across splits: {sorted(overlap)}.",
            )
    report.addCheck("no_entity_leakage")

    testHighRiskNonAdversarial = [
        scenario
        for scenario in bundle.scenarios
        if scenario.split == "test"
        and scenario.riskLevel == "high"
        and not scenario.injectionAttempt
    ]
    if not testHighRiskNonAdversarial:
        report.addIssue(
            "test_high_risk_positive_missing",
            "Held-out test split must contain at least one non-adversarial "
            "high-risk scenario so riskFlagRecall is meaningful.",
        )
    report.addCheck("test_has_high_risk_positive")

    return report


def runValidationAndSave(bundle: DataBundle, processedDir: Path) -> ValidationReport:
    """Validate a bundle and persist the report next to the dataset."""
    report = validateBundle(bundle)
    atomicWriteJson(processedDir / "validation_report.json", report.toDict())
    return report


def summarizeValidation(report: ValidationReport) -> str:
    """Render a human-readable summary of the validation report."""
    if report.passed:
        return f"Validation passed ({len(report.checksRun)} checks)."
    lines = [f"Validation failed with {len(report.issues)} issue(s):"]
    for issue in report.issues:
        lines.append(f"  - [{issue.code}] {issue.message}")
    return "\n".join(lines)


def iterScenarioSplits(bundle: DataBundle, split: str) -> Iterable[Scenario]:
    """Yield scenarios belonging to a split in deterministic order."""
    for scenario in bundle.scenarios:
        if scenario.split == split:
            yield scenario
