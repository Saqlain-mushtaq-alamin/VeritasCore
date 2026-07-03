"""Unit tests for veritascore.core.config (Phase 0)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from veritascore.core.config import EngineConfig, ModelConfig, SearchConfig


class TestModelConfig:
    def test_defaults(self) -> None:
        config = ModelConfig()
        assert config.nli_model == "cross-encoder/nli-deberta-v3-base"
        assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.device == "auto"
        assert config.max_batch_size == 16

    def test_valid_devices(self) -> None:
        for device in ["auto", "cuda", "cpu", "cuda:0", "cuda:1"]:
            config = ModelConfig(device=device)
            assert config.device == device

    def test_invalid_device(self) -> None:
        with pytest.raises(ValueError):
            ModelConfig(device="tpu")


class TestSearchConfig:
    def test_defaults(self) -> None:
        config = SearchConfig()
        assert config.provider == "tavily"
        assert config.max_results == 5
        assert config.timeout_seconds == 10.0
        assert config.cache_enabled is True

    def test_no_api_keys_by_default(self) -> None:
        config = SearchConfig()
        assert config.tavily_api_key is None
        assert config.brave_api_key is None


class TestEngineConfig:
    def test_default(self) -> None:
        config = EngineConfig.default()
        assert config.domain_profile == "general"
        assert config.verification_mode == "auto"
        assert config.log_level == "INFO"
        assert config.max_claims_per_response == 50

    def test_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""
            models:
              nli_model: "cross-encoder/nli-deberta-v3-base"
              device: "cpu"
              max_batch_size: 8
            search:
              provider: "none"
              max_results: 3
            domain_profile: "medical"
            log_level: "DEBUG"
        """)
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)

        config = EngineConfig.from_yaml(config_file)
        assert config.models.device == "cpu"
        assert config.models.max_batch_size == 8
        assert config.search.provider == "none"
        assert config.search.max_results == 3
        assert config.domain_profile == "medical"
        assert config.log_level == "DEBUG"

    def test_from_yaml_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            EngineConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_from_yaml_partial(self, tmp_path: Path) -> None:
        """YAML with only some fields — others should use defaults."""
        yaml_content = "log_level: WARNING\n"
        config_file = tmp_path / "partial.yaml"
        config_file.write_text(yaml_content)

        config = EngineConfig.from_yaml(config_file)
        assert config.log_level == "WARNING"
        assert config.domain_profile == "general"  # default
