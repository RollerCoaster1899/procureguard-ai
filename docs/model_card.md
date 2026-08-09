# Model Card: ProcureGuard AI Decision Providers

## Model Purpose

ProcureGuard AI evaluates decision providers for a synthetic procurement
copilot. The card covers the providers used by the workflow benchmark and the
API.

## Providers

### Scripted Provider (control plane)

- **Purpose**: deterministic, reproducible decision provider for benchmarking.
- **Training data**: none. It is a hand-written rule function.
- **Evaluation data**: held-out synthetic test scenarios.
- **Metrics**: see `docs/methodology.md` and `reports/final_report.md`.
- **Limitations**: it does not measure DeepSeek model quality. It encodes the
  same policy rules used to generate expected labels.

### DeepSeek Provider (optional, live)

- **Purpose**: optional live model-backed provider using the OpenAI-compatible
  DeepSeek chat API.
- **Model**: `deepseek-v4-flash` (configurable via `DEEPSEEK_MODEL`).
- **Base URL**: `https://api.deepseek.com` (configurable via
  `DEEPSEEK_BASE_URL`).
- **Intended use**: only when explicitly selected with `--provider deepseek`
  and a valid `DEEPSEEK_API_KEY`.
- **Out-of-scope use**: the offline benchmark never calls this provider.

## Intended Use

- Benchmarking retrieval and guarded decision workflows on synthetic data.
- Demonstrating prompt-injection quarantine, evidence checks, and human gates.
- Serving guarded recommendations through the FastAPI endpoint.

## Out-of-Scope Use

- Real procurement decision-making without human oversight and external
  validation.
- Claims about DeepSeek model quality from the scripted benchmark.

## Limitations and Failure Modes

- Synthetic data only.
- The scripted provider cannot generalize beyond its encoded rules.
- The live provider requires network access, an API key, and incurs cost;
  results are nondeterministic and model-version dependent.

## Operational Requirements

- Python >= 3.11, pinned dependencies via `uv.lock`.
- No network or API key required for the offline benchmark.
- Read-only MCP fixture server; no write-capable tools.

## Monitoring Recommendations

- Track unsafe autonomous action rate and human-gate completeness on held-out
  adversarial scenarios.
- Track citation validity and fact-support violations to detect retrieval or
  evidence drift.
- For live providers, record request IDs, token usage, latency, and model
  version without logging prompts.
