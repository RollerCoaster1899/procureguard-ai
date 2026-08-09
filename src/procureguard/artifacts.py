"""Artifact persistence helpers: atomic JSON writes, CSV, traces, metadata.

Atomic writes prevent a partially written artifact from being mistaken for a
valid result. All artifact functions create parent directories as needed and
keep deterministic ordering where it matters.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def ensureDir(path: Path) -> None:
    """Create a directory and all parents if they do not exist."""
    path.mkdir(parents=True, exist_ok=True)


def atomicWriteText(path: Path, content: str) -> None:
    """Write text atomically by writing a temp file and renaming it."""
    ensureDir(path.parent)
    tempPath = path.with_suffix(path.suffix + ".tmp")
    with tempPath.open("w", encoding="utf-8") as handle:
        handle.write(content)
    os.replace(tempPath, path)


def atomicWriteJson(path: Path, payload: Any, *, sortKeys: bool = True) -> None:
    """Write a JSON document atomically with stable key ordering."""
    content = json.dumps(payload, indent=2, sort_keys=sortKeys)
    atomicWriteText(path, content + "\n")


def appendJsonLines(path: Path, record: Mapping[str, Any]) -> None:
    """Append a JSON object as one line to a JSONL trace file.

    Note: this helper is intended for a single sequential writer (the reproduce
    pipeline). It opens the file in append mode without file locking, so
    concurrent writers or parallel processes must not share the same path.
    """
    ensureDir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def writeCsv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows to a CSV file with a consistent header order."""
    ensureDir(path.parent)
    if not rows:
        atomicWriteText(path, "")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def writeRunMetadata(path: Path, fields: Mapping[str, Any]) -> None:
    """Write run metadata with a version field to a JSON file."""
    payload = {"version": 1, **dict(fields)}
    atomicWriteJson(path, payload)


def listJsonLines(path: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file into a list of dicts."""
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def iterJsonLines(path: Path) -> Iterable[dict[str, Any]]:
    """Yield records from a JSONL file lazily."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)
