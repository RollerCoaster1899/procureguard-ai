#!/usr/bin/env python3
"""Reproduce the ProcureGuard AI offline experiment.

Usage:
    python scripts/reproduce.py                       # full offline experiment
    python scripts/reproduce.py --smoke               # tiny deterministic smoke run
    python scripts/reproduce.py --provider deepseek   # live DeepSeek (requires key)

Smoke runs write only to reports_smoke/ and never overwrite the full results
in reports/.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ProcureGuard AI offline benchmark.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny deterministic subset instead of the full experiment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for artifacts (default reports/ or reports_smoke/).",
    )
    parser.add_argument(
        "--provider",
        default="scripted",
        help="Decision provider: scripted (default) or deepseek (live API).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the base YAML configuration file.",
    )
    parser.add_argument("--run-id", default=None, help="Correlation run ID.")
    args = parser.parse_args()

    from procureguard.pipeline import runExperiment

    combined = runExperiment(
        configPath=args.config,
        outputDir=args.output,
        smoke=args.smoke,
        providerName=args.provider,
        runId=args.run_id,
    )
    print(
        f"run completed: retrieval queries={combined['retrieval']['numQueries']}, "
        f"workflow scenarios={combined['workflow']['numScenarios']}"
    )


if __name__ == "__main__":
    main()
