#!/usr/bin/env python3
"""Generate and validate the deterministic synthetic dataset without running benchmarks.

Used by the Docker image build to produce fresh processed data instead of
accidentally copying stale artifacts from the build context.

Usage:
    python scripts/generate_data.py [--config configs/base.yaml]
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and validate the ProcureGuard AI synthetic dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the base YAML configuration file.",
    )
    args = parser.parse_args()

    rootDir = Path(__file__).resolve().parent.parent
    configPath = args.config or (rootDir / "configs" / "base.yaml")

    from procureguard.config import loadExperimentConfig
    from procureguard.data.generator import generateBundle
    from procureguard.data.io import saveBundle
    from procureguard.data.validate import runValidationAndSave, summarizeValidation

    config = loadExperimentConfig(configPath, rootDir=rootDir, outputDir=rootDir / "reports")
    config.paths.processedDir.mkdir(parents=True, exist_ok=True)
    bundle = generateBundle(
        seed=config.data.seed,
        numScenarios=config.data.numScenarios,
        trainCount=config.data.trainCount,
        valCount=config.data.valCount,
        testCount=config.data.testCount,
        adversarialIndices=config.data.adversarialIndices,
    )
    saveBundle(bundle, config.paths.processedDir)
    validation = runValidationAndSave(bundle, config.paths.processedDir)
    if not validation.passed:
        raise SystemExit(summarizeValidation(validation))
    print(
        f"generated and validated dataset: {len(bundle.corpus.documents)} documents, "
        f"{len(bundle.scenarios)} scenarios"
    )


if __name__ == "__main__":
    main()
