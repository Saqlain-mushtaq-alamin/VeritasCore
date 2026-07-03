"""Configuration management for VeritasCore.

Supports loading config from YAML files, environment variables, and
programmatic construction. Uses pydantic-settings for env var binding.

Example:
    >>> config = EngineConfig.default()
    >>> config = EngineConfig.from_yaml("configs/default.yaml")
    >>> config = EngineConfig(models=ModelConfig(device="cpu"))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Configuration for ML models used by the engine.

    Environment variable overrides (prefix: VERITASCORE_):
        VERITASCORE_NLI_MODEL, VERITASCORE_EMBEDDING_MODEL, etc.
    """

    model_config = SettingsConfigDict(env_prefix="VERITASCORE_", extra="ignore")

    nli_model: str = Field(
        default="cross-encoder/nli-deberta-v3-base",
        description="HuggingFace model ID for NLI (grounded verification)",
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model ID for sentence embeddings",
    )
    decomposer_model: str = Field(
        default="microsoft/Phi-3-mini-4k-instruct",
        description="HuggingFace model ID for claim decomposition LLM",
    )
    device: str = Field(
        default="auto",
        description="Inference device: 'auto', 'cuda', 'cpu'",
    )
    max_batch_size: int = Field(
        default=16,
        description="Maximum batch size for model inference",
    )

    @model_validator(mode="after")
    def validate_device(self) -> ModelConfig:
        valid_devices = {"auto", "cuda", "cpu"}
        if self.device not in valid_devices and not self.device.startswith("cuda:"):
            raise ValueError(f"device must be one of {valid_devices} or 'cuda:N', got '{self.device}'")
        return self


class SearchConfig(BaseSettings):
    """Configuration for web search providers (ungrounded verification).

    Environment variable overrides:
        TAVILY_API_KEY, BRAVE_API_KEY
    """

    model_config = SettingsConfigDict(extra="ignore")

    provider: str = Field(
        default="tavily",
        description="Search provider: 'tavily', 'brave', 'none'",
    )
    tavily_api_key: str | None = Field(
        default=None,
        alias="TAVILY_API_KEY",
        description="Tavily Search API key",
    )
    brave_api_key: str | None = Field(
        default=None,
        alias="BRAVE_API_KEY",
        description="Brave Search API key",
    )
    max_results: int = Field(
        default=5,
        description="Maximum search results to retrieve per query",
    )
    timeout_seconds: float = Field(
        default=10.0,
        description="HTTP timeout for search requests",
    )
    cache_enabled: bool = Field(
        default=True,
        description="Cache search results to disk",
    )
    cache_dir: str = Field(
        default=".cache/search",
        description="Directory for search result cache",
    )

    model_config = SettingsConfigDict(populate_by_name=True, extra="ignore")


class EngineConfig(BaseSettings):
    """Top-level engine configuration.

    Composes ModelConfig and SearchConfig. Can be loaded from YAML,
    environment variables, or constructed programmatically.

    Example:
        >>> config = EngineConfig.default()
        >>> config = EngineConfig.from_yaml("configs/default.yaml")
        >>> # Override specific fields:
        >>> config = EngineConfig(models=ModelConfig(device="cpu"))
    """

    model_config = SettingsConfigDict(env_prefix="VERITASCORE_", extra="ignore")

    models: ModelConfig = Field(default_factory=ModelConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    domain_profile: str = Field(
        default="general",
        description="Domain profile to apply: 'general', 'medical', 'legal'",
    )
    verification_mode: str = Field(
        default="auto",
        description="Verification mode: 'auto', 'grounded', 'ungrounded', 'offline'",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR",
    )
    max_claims_per_response: int = Field(
        default=50,
        description="Maximum number of claims to extract per response",
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> EngineConfig:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            EngineConfig instance with values from the file.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the YAML content is invalid.

        Example:
            >>> config = EngineConfig.from_yaml("configs/default.yaml")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        # Nest sub-configs
        if "models" in data:
            data["models"] = ModelConfig(**data["models"])
        if "search" in data:
            data["search"] = SearchConfig(**data["search"])
        return cls(**data)

    @classmethod
    def default(cls) -> EngineConfig:
        """Return default configuration with sensible defaults.

        Returns:
            EngineConfig instance with all defaults applied.
        """
        return cls(
            models=ModelConfig(),
            search=SearchConfig(),
        )
