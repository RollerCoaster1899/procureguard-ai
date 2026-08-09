"""Configuration models and loading for ProcureGuard AI.

Configuration is loaded from YAML files and can be overridden by environment
variables for the optional live DeepSeek provider. All runtime paths are
computed relative to an explicit root directory so the package never relies on
the current working directory.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError


class DataConfig(BaseModel):
    """Deterministic data generation parameters."""

    seed: int = 42
    numSuppliers: int = 12
    numScenarios: int = 24
    trainCount: int = 12
    valCount: int = 6
    testCount: int = 6
    adversarialIndices: list[int] = Field(default_factory=lambda: [5, 10, 15, 22, 23])
    numPolicyDocs: int = 10


class RetrievalConfig(BaseModel):
    """Retrieval benchmark parameters."""

    topK: int = 10
    candidateCount: int = 200
    lsaComponents: int = 100
    rrfK: int = 60


class WorkflowConfig(BaseModel):
    """Decision workflow benchmark parameters."""

    topK: int = 5
    methods: list[str] = Field(
        default_factory=lambda: [
            "rules_baseline",
            "rag_only",
            "rag_mcp",
            "guarded_rag_mcp",
        ]
    )
    provider: str = "scripted"
    mcpTimeoutSeconds: float = 15.0


class BootstrapConfig(BaseModel):
    """Bootstrap confidence interval parameters."""

    nResamples: int = 1000
    seed: int = 7
    alpha: float = 0.05


class PathsConfig(BaseModel):
    """Computed runtime directory layout."""

    rootDir: Path
    dataDir: Path
    rawDir: Path
    interimDir: Path
    processedDir: Path
    outputDir: Path
    figuresDir: Path
    metricsDir: Path
    tablesDir: Path
    tracesDir: Path


class DeepSeekConfig(BaseModel):
    """Live DeepSeek provider configuration (optional)."""

    apiKey: str | None = None
    baseUrl: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    timeoutSeconds: float = 30.0
    maxRetries: int = 2


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    data: DataConfig
    retrieval: RetrievalConfig
    workflow: WorkflowConfig
    bootstrap: BootstrapConfig
    paths: PathsConfig
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)


def _deepMerge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge override mapping into base mapping."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deepMerge(result[key], value)
        else:
            result[key] = value
    return result


def _loadYaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return payload


def _defaultDirs(rootDir: Path, outputDir: Path) -> dict[str, Path]:
    dataDir = rootDir / "data"
    return {
        "rootDir": rootDir,
        "dataDir": dataDir,
        "rawDir": dataDir / "raw",
        "interimDir": dataDir / "interim",
        "processedDir": dataDir / "processed",
        "outputDir": outputDir,
        "figuresDir": outputDir / "figures",
        "metricsDir": outputDir / "metrics",
        "tablesDir": outputDir / "tables",
        "tracesDir": outputDir / "traces",
    }


def loadExperimentConfig(
    configPath: Path,
    *,
    rootDir: Path,
    outputDir: Path,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Load and validate the experiment configuration.

    Args:
        configPath: Path to the base YAML configuration file.
        rootDir: Repository root directory used to resolve data paths.
        outputDir: Directory where artifacts will be written.
        overrides: Optional nested mapping merged on top of the YAML payload.

    Returns:
        A validated ExperimentConfig instance.
    """
    base = _loadYaml(configPath)
    if overrides:
        base = _deepMerge(base, dict(overrides))
    base.setdefault("data", {})
    base.setdefault("retrieval", {})
    base.setdefault("workflow", {})
    base.setdefault("bootstrap", {})
    dirs = _defaultDirs(rootDir, outputDir)
    base["paths"] = {key: str(value) for key, value in dirs.items()}
    base.setdefault("deepseek", {})
    base["deepseek"].setdefault("apiKey", os.environ.get("DEEPSEEK_API_KEY"))
    base["deepseek"].setdefault(
        "baseUrl", os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    base["deepseek"].setdefault("model", os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    try:
        return ExperimentConfig.model_validate(base)
    except ValidationError as exc:
        raise ValueError(f"Invalid experiment configuration: {exc}") from exc


def _relativePath(path: Path, rootDir: Path) -> str:
    """Return a repository-relative POSIX path when possible.

    Paths outside the repository root are returned in absolute form so the
    serialized config never depends on the current working directory.
    """
    try:
        return path.resolve().relative_to(rootDir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sanitizedConfigPayload(config: ExperimentConfig) -> dict[str, Any]:
    """Serialize a config for provenance with secrets and absolute paths removed.

    The DeepSeek API key is always persisted as null even when it was loaded
    from the environment, and every path is stored relative to the repository
    root so tracked artifacts do not leak local absolute paths.
    """
    payload = config.model_dump(mode="json")
    deepseek = payload.get("deepseek")
    if isinstance(deepseek, dict):
        deepseek["apiKey"] = None
    paths = payload.get("paths")
    if isinstance(paths, dict):
        rootDir = Path(config.paths.rootDir)
        payload["paths"] = {
            key: _relativePath(Path(value), rootDir) if isinstance(value, str) else value
            for key, value in paths.items()
        }
    return payload


def writeConfigYaml(config: ExperimentConfig, path: Path) -> None:
    """Persist a sanitized configuration as YAML for run provenance.

    Secrets (the DeepSeek API key) are always serialized as null and all paths
    are stored repository-relative.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = sanitizedConfigPayload(config)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)


def deepSeekFromEnv() -> DeepSeekConfig:
    """Build DeepSeek config from environment variables."""
    return DeepSeekConfig(
        apiKey=os.environ.get("DEEPSEEK_API_KEY"),
        baseUrl=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        timeoutSeconds=float(os.environ.get("DEEPSEEK_TIMEOUT", "30")),
        maxRetries=int(os.environ.get("DEEPSEEK_MAX_RETRIES", "2")),
    )
