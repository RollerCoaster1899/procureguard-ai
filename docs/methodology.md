# Methodology

## Retrieval Benchmark

Four retrieval methods are evaluated on identical queries and qrels:

1. `no_retrieval` (baseline): returns an empty ranking, scoring 0.0 by
   construction.
2. `bm25`: Okapi BM25 from `rank-bm25` over lowercased alphanumeric tokens.
3. `lsa_vector`: scikit-learn `TfidfVectorizer` plus `TruncatedSVD` document
   and query embeddings with cosine similarity. CPU-only and network-free.
4. `hybrid_rrf`: reciprocal rank fusion of the BM25 and LSA rankings with
   `rrfK=60`.

Metrics: nDCG@10, recall@10, MRR@10, precision@5, and retrieval latency.
Bootstrap 95% confidence intervals use a fixed seed (`bootstrap.seed = 7`).

## Decision Workflow Benchmark

Four methods are evaluated on identical held-out test scenarios:

1. `rules_baseline`: a deterministic rule engine with perfect structured facts
   (the stand-in for an ERP-integrated rule system). It does not use
   retrieval, the LLM, or safety gates.
2. `rag_only`: hybrid retrieval citations only; facts are parsed from retrieved
   document text.
3. `rag_mcp`: hybrid retrieval citations plus supplier risk/budget/compliance/
   spend facts obtained from read-only MCP tools.
4. `guarded_rag_mcp`: adds prompt-injection quarantine, citation validity
   checks, fact-support evidence checks, mandatory human approval, and a hard
   block on autonomous write actions.

### Providers

- `scripted` (default): a deterministic provider that maps evidence to a
  structured decision. It is the primary reproducible benchmark and must not be
  interpreted as measuring DeepSeek model quality.
- `deepseek` (optional): a live provider using the OpenAI-compatible DeepSeek
  chat API (`https://api.deepseek.com`, model `deepseek-v4-flash`). It is used
  only when explicitly selected with `--provider deepseek` and a valid
  `DEEPSEEK_API_KEY`. Live results are labeled with the `deepseek` provider and
  stored separately from the scripted control-plane benchmark.

### Decision Policy

Given supplier facts (risk, compliance, budget), the policy is:

- Reject when risk is high or compliance is a violation.
- Review when the order is over budget.
- Approve otherwise.

Human approval is mandatory when risk is high, compliance is a violation, the
order is over budget, evidence is unsupported, or an injection was detected.

### Safety Gates (guarded method)

1. Prompt-injection quarantine: known injection patterns are detected and the
   request is blocked before any tool call or model call.
2. Citation validity: cited documents must exist in the corpus and be relevant
   per qrels.
3. Fact support: each fact asserted by the MCP tools must appear in at least
   one citation document; unsupported facts force human review.
4. Mandatory human gate: high-risk, compliance, budget, unsupported-evidence,
   and quarantined cases require human approval.
5. No autonomous write: the guarded workflow always forces `proposedWrite =
   False` and records whether a write was blocked.

## Validation Protocol

- Data generation is deterministic with `seed = 42`.
- Splits never share a supplier (no entity leakage).
- Expected labels are recomputed from the same policy rules used by the
  providers, so evaluation is internally consistent.
- qrels reference only documents that exist in the corpus.
- No real company names or PII are permitted; validation enforces this.

## Metrics Computed

Retrieval: nDCG@10, recall@10, MRR@10, precision@5, latency, bootstrap 95% CIs.

Workflow: recommendation accuracy, compliance flag recall, risk flag recall,
budget adherence, citation validity, unsafe autonomous action rate, mandatory
human-gate completeness, latency, tool calls, bootstrap 95% CIs.
