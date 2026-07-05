"""Domain profile base class and configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class VerificationConstraints(BaseModel):
    """Constraints on how verification is performed for this domain."""

    allowed_sources: list[str] = Field(default=["web", "context"])
    blocked_domains: list[str] = Field(default_factory=list)
    required_source_count: int = Field(default=1)
    max_search_results: int = Field(default=5)


class SeverityWeights(BaseModel):
    """Weights controlling how verdict types affect the response trust score."""

    contradicted_weight: float = Field(default=2.0)
    unsupported_weight: float = Field(default=1.0)
    supported_weight: float = Field(default=1.0)
    off_topic_penalty: float = Field(default=0.3)


class ThresholdConfig(BaseModel):
    """Score thresholds tuned per domain — the primary differentiation axis."""

    entailment_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    contradiction_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    consistency_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    overall_trust_warning: float = Field(default=0.6, ge=0.0, le=1.0)
    overall_trust_critical: float = Field(default=0.3, ge=0.0, le=1.0)


class DomainProfile(BaseModel):
    """Complete domain-specific configuration profile.

    Example:
        >>> profile = DomainProfile.from_yaml("configs/profiles/medical.yaml")
        >>> profile.apply_to_nli_verifier(nli_verifier)
    """

    name: str
    display_name: str
    description: str
    constraints: VerificationConstraints = Field(default_factory=VerificationConstraints)
    severity: SeverityWeights = Field(default_factory=SeverityWeights)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    claim_decomposition_hints: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> DomainProfile:
        """Load a DomainProfile from a YAML file.

        Uses model_validate() instead of cls(**data) so nested dicts in
        the YAML are correctly coerced into sub-models — the reference
        spec's cls(**data) would raise a Pydantic ValidationError for
        nested sub-model fields (VerificationConstraints, etc.).

        Raises:
            FileNotFoundError: If the YAML file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Profile YAML not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def apply_to_nli_verifier(self, verifier: Any) -> None:
        """Apply this profile's NLI thresholds to an NLIVerifier instance.

        Satisfies G6 criterion 6 — the reference spec listed this as a
        criterion but never implemented the integration point. Modifies
        verifier.entailment_threshold and verifier.contradiction_threshold
        in-place.

        Args:
            verifier: An NLIVerifier instance (or any object with those attrs).
        """
        verifier.entailment_threshold = self.thresholds.entailment_threshold
        verifier.contradiction_threshold = self.thresholds.contradiction_threshold

    def apply_to_consistency_checker(self, checker: Any) -> None:
        """Apply this profile's consistency threshold to a SemanticConsistencyChecker."""
        checker.relevance_threshold = self.thresholds.consistency_threshold

    def get_trust_level(self, trust_score: float) -> str:
        """Classify a trust score as 'ok', 'warning', or 'critical'.

        Returns:
            'critical' | 'warning' | 'ok'
        """
        if trust_score < self.thresholds.overall_trust_critical:
            return "critical"
        if trust_score < self.thresholds.overall_trust_warning:
            return "warning"
        return "ok"
