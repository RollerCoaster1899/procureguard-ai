# Data Card: ProcureGuard AI Synthetic Dataset

## Dataset Name

ProcureGuard AI synthetic procurement benchmark dataset (version 0.1.0).

## Source

Generated deterministically by `src/procureguard/data/generator.py` with
`seed = 42`. No external data, downloads, or scraping.

## License

MIT (same as the repository).

## Contents

- **Corpus**: 45 documents.
  - 24 supplier documents (2 per fictional supplier: profile + capabilities)
  - 10 policy documents
  - 11 category guides
- **Scenarios**: 24 procurement scenarios:
  - 12 train / 6 validation / 6 test
  - 5 adversarial (prompt-injection) scenarios: 2 train, 1 validation, 2 test
- **Queries and qrels**: 24 queries with binary relevance judgements.
- **Fixture facts**: 12 supplier records with risk level, compliance status,
  budget status, spend amount, and budget amount.

## Collection Process

Not collected. All content is written by code from fixed fictional templates
and a fixed supplier profile table. No real companies, people, or addresses
are used.

## Time Period

Synthetic; no time dimension. Scenario dates are not part of the data.

## Feature Definitions

| Field | Domain | Meaning |
|---|---|---|
| `riskLevel` | low, medium, high | Supplier risk assessment |
| `complianceStatus` | compliant, violation | Compliance status |
| `budgetStatus` | within, over | Budget adherence |
| `spendAmount` | float >= 0 | Requested spend |
| `budgetAmount` | float >= 0 | Approved budget |
| `split` | train, validation, test | Split assignment |
| `injectionAttempt` | bool | Whether the scenario includes a prompt-injection attempt |
| `expectedAction` | approve, review, reject, block | Ground-truth decision label |

## Target Definition

The decision target is `expectedAction`. It is derived from the same policy
rules used by the scripted provider:

- reject if `riskLevel == high` or `complianceStatus == violation`
- review if `budgetStatus == over`
- approve otherwise
- block for any adversarial scenario

## Known Quality Issues

- Small synthetic corpus; retrieval findings may not transfer to real data.
- Supplier-level facts are reused across scenarios within a split, reducing
  scenario independence.
- The supplier profile table is hand-curated to balance outcomes, which is not
  representative of a real supplier base.

## Biases

- Synthetic category and supplier distributions are not drawn from any real
  population.
- High-risk and compliance-violation profiles are over-represented relative to
  a typical healthy supplier base to exercise the safety gates.

## Sensitive Information

None. The data contains no PII, credentials, or real entity names. Validation
actively rejects emails, SSN-like patterns, and a blocklist of real companies.

## Permitted Uses

- Retrieval benchmarking
- RAG and MCP decision-workflow benchmarking
- Educational demonstrations of synthetic enterprise copilots

## Prohibited or Discouraged Uses

- Presenting synthetic results as real-world procurement validation
- Training production models on this data without external validation
- Making procurement decisions about real suppliers based on this data
