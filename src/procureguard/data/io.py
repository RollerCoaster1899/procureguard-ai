"""Dataset persistence helpers for ProcureGuard AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from procureguard.artifacts import atomicWriteJson, atomicWriteText
from procureguard.data.schema import DataBundle


def saveBundle(bundle: DataBundle, processedDir: Path) -> None:
    """Persist the generated dataset as versioned JSON files."""
    processedDir.mkdir(parents=True, exist_ok=True)
    atomicWriteJson(processedDir / "corpus.json", bundle.corpus.model_dump(mode="json"))
    atomicWriteJson(
        processedDir / "queries.json",
        [query.model_dump(mode="json") for query in bundle.queries],
    )
    atomicWriteJson(
        processedDir / "scenarios.json",
        [scenario.model_dump(mode="json") for scenario in bundle.scenarios],
    )
    atomicWriteJson(
        processedDir / "fixture_facts.json",
        [facts.model_dump(mode="json") for facts in bundle.fixtureFacts],
    )
    atomicWriteJson(
        processedDir / "qrels.json",
        [{"queryId": query.queryId, "docIds": query.qrels} for query in bundle.queries],
    )
    atomicWriteJson(processedDir / "seed.json", {"seed": bundle.seed})
    atomicWriteText(
        processedDir / "README.txt",
        "Generated synthetic dataset for ProcureGuard AI. "
        "All data is deterministic, fictional, and contains no real companies or PII.\n",
    )


def loadBundle(processedDir: Path) -> DataBundle:
    """Load a previously generated dataset from JSON files."""
    corpus_path = processedDir / "corpus.json"
    if not corpus_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found in {processedDir}. Run the data generation step first."
        )
    from procureguard.data.schema import Corpus

    corpus = Corpus.model_validate_json(corpus_path.read_text(encoding="utf-8"))
    queries_path = processedDir / "queries.json"
    scenarios_path = processedDir / "scenarios.json"
    facts_path = processedDir / "fixture_facts.json"
    seed_path = processedDir / "seed.json"
    from procureguard.data.schema import FixtureFacts, QueryRecord, Scenario

    queries = [QueryRecord.model_validate(item) for item in _loadJsonList(queries_path)]
    scenarios = [Scenario.model_validate(item) for item in _loadJsonList(scenarios_path)]
    facts = [FixtureFacts.model_validate(item) for item in _loadJsonList(facts_path)]
    seed = _loadJsonDict(seed_path).get("seed", 42)
    return DataBundle(
        corpus=corpus,
        queries=queries,
        scenarios=scenarios,
        fixtureFacts=facts,
        seed=seed,
    )


def _loadJsonList(path: Path) -> list[dict[str, Any]]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return payload


def _loadJsonDict(path: Path) -> dict[str, Any]:
    import json

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload
