"""Unit tests for artifact persistence helpers."""

from __future__ import annotations

import json

from procureguard.artifacts import (
    appendJsonLines,
    atomicWriteJson,
    atomicWriteText,
    listJsonLines,
    writeCsv,
    writeRunMetadata,
)


def testAtomicWriteJson(tmp_path):
    path = tmp_path / "nested" / "data.json"
    atomicWriteJson(path, {"b": 2, "a": 1})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"b": 2, "a": 1}
    # deterministic key ordering
    content = path.read_text(encoding="utf-8")
    assert content.index('"a"') < content.index('"b"')


def testAtomicWriteOverwrites(tmp_path):
    path = tmp_path / "data.json"
    atomicWriteJson(path, {"value": 1})
    atomicWriteJson(path, {"value": 2})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"value": 2}


def testAppendAndReadJsonLines(tmp_path):
    path = tmp_path / "traces.jsonl"
    appendJsonLines(path, {"id": 1})
    appendJsonLines(path, {"id": 2})
    records = listJsonLines(path)
    assert [record["id"] for record in records] == [1, 2]


def testListJsonLinesMissingFile(tmp_path):
    assert listJsonLines(tmp_path / "missing.jsonl") == []


def testWriteCsv(tmp_path):
    path = tmp_path / "table.csv"
    writeCsv(path, [{"method": "bm25", "score": 0.5}, {"method": "lsa", "score": 0.6}])
    content = path.read_text(encoding="utf-8")
    assert "method,score" in content
    assert "bm25" in content


def testWriteCsvEmpty(tmp_path):
    path = tmp_path / "empty.csv"
    writeCsv(path, [])
    assert path.read_text(encoding="utf-8") == ""


def testWriteRunMetadata(tmp_path):
    path = tmp_path / "run.json"
    writeRunMetadata(path, {"runId": "abc"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["runId"] == "abc"


def testAtomicWriteText(tmp_path):
    path = tmp_path / "report.md"
    atomicWriteText(path, "# hi")
    assert path.read_text(encoding="utf-8") == "# hi"
