"""Unit tests for configuration loading."""

from __future__ import annotations

import yaml

from procureguard.config import (
    DeepSeekConfig,
    loadExperimentConfig,
    writeConfigYaml,
)


def testLoadConfigDefaults(tmpRepo):
    config = loadExperimentConfig(
        tmpRepo / "configs" / "base.yaml", rootDir=tmpRepo, outputDir=tmpRepo / "reports"
    )
    assert config.data.seed == 42
    assert config.retrieval.topK == 10
    assert config.workflow.provider == "scripted"
    assert config.paths.processedDir == tmpRepo / "data" / "processed"
    assert config.paths.outputDir == tmpRepo / "reports"
    assert config.paths.rootDir == tmpRepo


def testLoadConfigOverrides(tmpRepo):
    overrides = {"data": {"seed": 99}, "retrieval": {"topK": 5}}
    config = loadExperimentConfig(
        tmpRepo / "configs" / "base.yaml",
        rootDir=tmpRepo,
        outputDir=tmpRepo / "reports",
        overrides=overrides,
    )
    assert config.data.seed == 99
    assert config.retrieval.topK == 5


def testLoadConfigMissingFile(tmp_path):
    try:
        loadExperimentConfig(
            tmp_path / "nope.yaml", rootDir=tmp_path, outputDir=tmp_path / "reports"
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")


def testWriteConfigRoundTrip(tmpRepo, tmp_path):
    config = loadExperimentConfig(
        tmpRepo / "configs" / "base.yaml", rootDir=tmpRepo, outputDir=tmpRepo / "reports"
    )
    outPath = tmp_path / "out.yaml"
    writeConfigYaml(config, outPath)
    reloaded = loadExperimentConfig(outPath, rootDir=tmpRepo, outputDir=tmpRepo / "reports")
    assert reloaded.data.seed == config.data.seed
    assert reloaded.retrieval.topK == config.retrieval.topK


def testDeepSeekDefaults():
    config = DeepSeekConfig()
    assert config.baseUrl == "https://api.deepseek.com"
    assert config.model == "deepseek-v4-flash"
    assert config.apiKey is None


def testWriteConfigYamlRedactsApiKey(tmpRepo, tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-sentinel-123")
    config = loadExperimentConfig(
        tmpRepo / "configs" / "base.yaml", rootDir=tmpRepo, outputDir=tmpRepo / "reports"
    )
    assert config.deepseek.apiKey == "sk-test-sentinel-123"
    outPath = tmp_path / "run_config.yaml"
    writeConfigYaml(config, outPath)
    content = outPath.read_text(encoding="utf-8")
    assert "sk-test-sentinel-123" not in content
    payload = yaml.safe_load(content)
    assert payload["deepseek"]["apiKey"] is None


def testWriteConfigYamlUsesRelativePaths(tmpRepo, tmp_path):
    config = loadExperimentConfig(
        tmpRepo / "configs" / "base.yaml", rootDir=tmpRepo, outputDir=tmpRepo / "reports"
    )
    outPath = tmp_path / "run_config.yaml"
    writeConfigYaml(config, outPath)
    payload = yaml.safe_load(outPath.read_text(encoding="utf-8"))
    assert payload["paths"]["rootDir"] == "."
    assert payload["paths"]["processedDir"] == "data/processed"
    assert payload["paths"]["outputDir"] == "reports"
    assert payload["paths"]["metricsDir"] == "reports/metrics"
