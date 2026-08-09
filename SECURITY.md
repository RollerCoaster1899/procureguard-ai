# Security Policy

## Supported Versions

This project is a research benchmark. Only the latest revision of the main
branch is supported.

## Reporting a Vulnerability

Do not open a public issue for security vulnerabilities. Report vulnerabilities
privately to the repository maintainer with a clear description, the affected
module, and a minimal reproduction if possible.

## Security Properties

- The dataset is fully synthetic. No real company names, personal data, or
  credentials are used.
- `.env` and `.env.*` files are excluded from version control. Secrets are
  never logged.
- The MCP fixture server exposes only read-only tools. There is no
  write-capable MCP tool and no state mutation.
- The guarded decision workflow quarantines prompt-injection attempts,
  validates citations, checks fact support against evidence, forces human
  approval, and blocks autonomous write actions.
- The live DeepSeek provider requires an explicit `--provider deepseek` flag
  and a `DEEPSEEK_API_KEY`; it cannot be reached by accident during the
  offline benchmark.
- Prompts are never logged. Only request IDs, model names, and token usage are
  recorded for live provider calls.

## Responsible Disclosure

Please allow maintainers a reasonable period to respond before any public
disclosure.
