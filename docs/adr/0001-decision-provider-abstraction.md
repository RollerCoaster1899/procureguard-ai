# ADR-0001: Decision Provider Abstraction and Control-Plane Benchmark

- Status: Accepted
- Date: 2026-08-09

## Context

ProcureGuard AI needs to benchmark decision workflows reproducibly while also
supporting an optional live LLM. A naive approach would call DeepSeek for every
scenario, producing nondeterministic, costly, network-dependent results and
conflating model quality with workflow safety.

## Decision

Introduce a `DecisionProvider` abstraction. The default `scripted` provider is
a pure deterministic function of the evidence in the request and is the primary
reproducible benchmark. The optional `deepseek` provider uses the OpenAI-
compatible DeepSeek chat API and is used only when explicitly selected with
`--provider deepseek` and a valid `DEEPSEEK_API_KEY`.

Safety behavior (quarantine, evidence checks, no autonomous write, mandatory
human approval) is enforced deterministically in the workflow engine, not
delegated to the provider. This means safety metrics are reproducible and do
not depend on model behavior.

## Alternatives Considered

- Always use the live LLM: rejected because it is nondeterministic, requires a
  key/network, costs money, and would make CI and smoke runs impossible.
- Enforce safety only in the prompt: rejected because prompt-level defenses are
  not hard guarantees; a hard gate is required for the claimed safety
  properties.

## Consequences

- Control-plane results are deterministic and CI-friendly.
- Live DeepSeek results, when run, are clearly labeled with the `deepseek`
  provider and kept separate from scripted results.
- Safety claims are backed by deterministic gate logic and regression tests,
  independent of any model.
