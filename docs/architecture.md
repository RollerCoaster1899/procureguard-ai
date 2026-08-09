# Architecture

## Overview

ProcureGuard AI is organized as a Python package (`src/procureguard`) with a
data layer, retrieval layer, workflow layer, MCP adapter, API, and CLI. The
reproduce pipeline orchestrates data generation, validation, and the two
benchmarks, then writes machine-readable artifacts.

```text
+-------------------+    +----------------------+    +--------------------+
| scripts/reproduce | -> | data/generator       | -> | data/validate      |
| (pipeline)        |    | (corpus, scenarios,  |    | (schema/leakage)   |
+-------------------+    |  qrels, fixture facts)|    +--------------------+
                         +----------------------+
                                 |  corpus + queries + scenarios
                                 v
+---------------------+   +--------------------+   +----------------------+
| retrieval benchmark |   | workflow benchmark |   | report + artifacts   |
| no_retrieval/bm25/  |   | rules_baseline/    |   | metrics, traces,     |
| lsa_vector/hybrid   |   | rag_only/rag_mcp/  |   | figures, final report|
+---------------------+   | guarded_rag_mcp    |   +----------------------+
                          +--------------------+
                                |        |
                    read-only MCP |        | FastAPI
                    fixture server|        v
                          +---------------------+
                          | provider abstraction|
                          | scripted | deepseek |
                          +---------------------+
```

## Layers

### Data Layer (`src/procureguard/data`)

- `generator.py`: deterministic synthetic generation (corpus, scenarios,
  queries/qrels, fixture facts).
- `validate.py`: schema checks, PII/company-name checks, expected-label
  recomputation, and entity-leakage checks.
- `io.py`: JSON persistence for the dataset.
- `schema.py`: pydantic contracts shared across layers.

### Retrieval Layer (`src/procureguard/retrieval`)

- `base.py`: `Retriever` protocol and `RetrieverRegistry` factory.
- `bm25.py`, `lsa.py`, `hybrid.py`, `no_retrieval.py`: the four methods.
- `metrics.py`: nDCG@10, recall@10, MRR@10, precision@5, bootstrap CIs.
- `benchmark.py`: runs the retrieval benchmark and persists artifacts.

### Workflow Layer (`src/procureguard/workflow`)

- `provider.py`: `DecisionProvider` abstraction and structured decision/output
  models.
- `scripted.py`: deterministic control-plane provider.
- `deepseek.py`: optional live provider (OpenAI-compatible DeepSeek API).
- `safety.py`: injection quarantine, citation validation, fact-support checks.
- `engine.py`: the four decision methods and the guarded gate.
- `metrics.py`: workflow metrics with bootstrap CIs.
- `benchmark.py`: runs the workflow benchmark and persists artifacts.

### MCP Layer (`src/procureguard/mcp`)

- `server.py`: read-only FastMCP server exposing supplier risk, budget,
  compliance, and spend tools over stdio.
- `client.py`: `McpClient` (stdio, timeout, structured content extraction),
  `McpFactsProvider`, and `InProcessFactsProvider`.

### Service and CLI

- `api.py`: FastAPI app with `/health` and `/v1/recommendations`, dependency
  injection via `Container`.
- `cli.py`: typer CLI (`run`, `serve`, `mcp-server`, `version`).
- `pipeline.py`: end-to-end reproduce orchestration.
- `report.py`: figures, comparison tables, final report, README update.

## Key Design Decisions

- Determinism: `random.Random` with fixed seeds only; data generation and the
  scripted provider are pure functions of their inputs.
- Single source of truth: `policy.py` defines the decision rules used by data
  generation, expected labels, and the scripted provider.
- Safety outside the model: the guarded gate enforces quarantine, evidence
  checks, and no-write behavior deterministically in the engine, independent of
  the provider.
- Read-only MCP: the fixture server has no write-capable tools.
- Secret hygiene: `.env*` ignored; DeepSeek keys are read from environment
  only; prompts are never logged.

## Data Flow

1. `scripts/reproduce.py` calls `pipeline.runExperiment`.
2. The pipeline generates and validates the dataset, saves it under
   `data/processed/`.
3. Retrieval benchmark runs four methods over all (or smoke) queries.
4. Workflow benchmark runs four methods over held-out test scenarios using the
   hybrid retriever and an MCP-backed facts provider.
5. Metrics, traces, tables, figures, run metadata, and the final report are
   written under the output directory (`reports/` or `reports_smoke/`).
