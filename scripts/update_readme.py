#!/usr/bin/env python3
"""Refresh the README Key Results section from generated full-run artifacts.

Usage:
    python scripts/update_readme.py

Reads reports/metrics/summary.json and rewrites the KEY_RESULTS block of
README.md. Fails with an actionable message when the artifacts are missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh README key results from full-run artifacts."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Path to metrics/summary.json (default reports/metrics/summary.json).",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="Path to README.md (default README.md).",
    )
    args = parser.parse_args()

    rootDir = Path(__file__).resolve().parent.parent
    metricsPath = args.metrics or (rootDir / "reports" / "metrics" / "summary.json")
    readmePath = args.readme or (rootDir / "README.md")

    if not metricsPath.is_file():
        raise SystemExit(
            f"Metrics not found at {metricsPath}. Run the full offline experiment "
            "first: `uv run python scripts/reproduce.py`."
        )

    from procureguard.report import updateReadmeResults

    with metricsPath.open("r", encoding="utf-8") as handle:
        combined = json.load(handle)
    updateReadmeResults(combined, readmePath)
    print(f"README key results updated from {metricsPath}")


if __name__ == "__main__":
    main()
