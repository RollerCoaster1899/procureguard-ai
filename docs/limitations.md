# Limitations

## Data

- The dataset is entirely synthetic. Results do not generalize to real
  procurement data without external validation on real corpora.
- The corpus is small (45 documents) and hand-curated. Retrieval findings,
  including any advantage of hybrid RRF over BM25, may not transfer to larger
  real-world collections.
- Supplier facts are reused across scenarios within a split, so scenario
  outcomes are not fully independent.
- The supplier profile table balances approve/review/reject outcomes, which is
  not representative of a real supplier base.

## Methodology

- The scripted provider is the primary reproducible benchmark. It is a
  deterministic rule function and must not be interpreted as measuring DeepSeek
  model quality.
- Workflow metrics on only 6 held-out test scenarios have wide bootstrap
  intervals; treat point estimates as indicative, not precise.
- The guarded method's injection patterns are a fixed list; novel injection
  strategies outside the list would not be quarantined. This is a defense-in-
  depth demonstration, not a complete security guarantee.
- Citation validity in the benchmark uses the generated qrels; in a production
  deployment qrels would be unavailable and evidence checks would rely on
  corpus membership and fact support alone (as the API does).

## Provider

- Live DeepSeek results require a `DEEPSEEK_API_KEY` and network access. They
  were not run in the reproducible offline benchmark. No DeepSeek model quality
  is claimed here.
- The DeepSeek adapter parses JSON from raw chat content; responses that do not
  contain a valid JSON object raise an actionable error.

## Operational

- The API default uses an in-process facts provider (identical to the MCP
  fixture data) for speed and testability. The standard MCP stdio transport is
  exercised by the benchmark and integration tests.
- Latency figures include process overhead for the MCP stdio subprocess and are
  measured on the machine that ran the experiment.

## Ethics and Safety

- Do not use synthetic results for real procurement decisions without human
  oversight and external validation.
- The system is designed to require human approval for risky, non-compliant,
  over-budget, or unsupported decisions; autonomous actions are never allowed in
  the guarded path.
