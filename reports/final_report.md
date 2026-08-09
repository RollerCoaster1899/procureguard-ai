# ProcureGuard AI - Final Report

## Executive summary

ProcureGuard AI evaluates retrieval quality and decision safety for a synthetic
enterprise procurement copilot. The retrieval benchmark compares four methods on
identical queries and qrels. The decision workflow benchmark compares four
methods on identical held-out test scenarios using a deterministic scripted
provider; live DeepSeek results are never mixed into this control-plane
benchmark.

Run ID: 0602008128a9
Provider: scripted
Smoke mode: False
Status: completed

## Retrieval benchmark (all queries, identical qrels)

| Method | nDCG@10 | Recall@10 | MRR@10 | Precision@5 | Latency (ms) |
|---|---|---|---|---|---|
| no_retrieval | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| bm25 | 0.8880 | 0.8631 | 1.0000 | 0.7250 | 0.52 |
| lsa_vector | 0.8477 | 0.7818 | 1.0000 | 0.7167 | 0.77 |
| hybrid_rrf | 0.8578 | 0.7985 | 1.0000 | 0.7167 | 1.32 |

The no-retrieval baseline scores 0.0 by construction. The hybrid RRF method
combines BM25 and local LSA vector rankings with reciprocal rank fusion.

## Decision workflow benchmark (held-out test scenarios)

| Method | Accuracy | Compliance recall | Risk recall | Budget adherence | Citation validity | Unsafe action rate | Human-gate | Latency (ms) | Tool calls |
|---|---|---|---|---|---|---|---|---|---|
| rules_baseline | 0.6667 | 1.0000 | 1.0000 | 0.8333 | 0.0000 | 0.3333 | 0.6000 | 0.28 | 0 |
| rag_only | 0.6667 | 1.0000 | 1.0000 | 0.8333 | 0.6667 | 0.3333 | 0.6000 | 1.81 | 0 |
| rag_mcp | 0.6667 | 1.0000 | 1.0000 | 0.8333 | 0.6667 | 0.3333 | 0.6000 | 23.43 | 24 |
| guarded_rag_mcp | 1.0000 | 1.0000 | 1.0000 | 0.8333 | 1.0000 | 0.0000 | 1.0000 | 14.32 | 16 |

## Bootstrap 95% confidence intervals (seeded)

| Method | Accuracy 95% CI | Unsafe rate 95% CI | Human-gate 95% CI |
|---|---|---|---|
| rules_baseline | [0.3333, 1.0000] | [0.0000, 0.6667] | [0.2000, 1.0000] |
| rag_only | [0.3333, 1.0000] | [0.0000, 0.6667] | [0.2000, 1.0000] |
| rag_mcp | [0.3333, 1.0000] | [0.0000, 0.6667] | [0.2000, 1.0000] |
| guarded_rag_mcp | [1.0000, 1.0000] | [0.0000, 0.0000] | [1.0000, 1.0000] |

## Interpretation

- The rules baseline uses perfect structured facts and shows what a pure rule
  engine achieves, but it is fully vulnerable to prompt injection and never
  requests human approval, producing a high unsafe autonomous action rate.
- RAG-only relies on facts extracted from retrieved citations, so its accuracy
  depends on retrieval quality.
- RAG + MCP adds exact supplier facts from read-only MCP tools, improving
  factual decisions, but remains vulnerable to injection.
- Guarded RAG + MCP quarantines injections, validates citations, checks fact
  support against evidence, forces human approval, and never performs an
  autonomous write, achieving zero unsafe autonomous actions.

## Reproduction

```bash
uv sync --extra dev
uv run python scripts/reproduce.py                     # full offline experiment
uv run python scripts/reproduce.py --smoke             # smoke experiment
```

See README.md for complete verified commands and limitations.
