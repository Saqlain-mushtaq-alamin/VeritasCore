"""Core data types for the VeritasCore verification pipeline.

These types form the canonical schema shared across all phases.
All phase implementations MUST use these types as their interface contract.

Cross-phase contracts (from Master Plan §7):
    Phase 1 → Phases 2,3,4 : list[Claim]
    Phase 2 → Phase 5      : ClaimVerdict with .nli_score populated
    Phase 3 → Phase 5      : ClaimVerdict with .retrieval_score populated
    Phase 4 → Phase 5      : ClaimVerdict with .consistency_score populated
    Phase 5 → Phase 6      : ClaimVerdict with .confidence (fused) populated
    Phase 6 → Phase 7      : Complete VerificationReport
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    """Three-way verification verdict for an atomic claim.

    Attributes:
        SUPPORTED: Evidence confirms the claim is accurate.
        CONTRADICTED: Evidence directly refutes the claim.
        UNSUPPORTED: No evidence found to confirm or refute the claim.
    """

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


class VerificationMode(str, Enum):
    """Operating mode for the VeritasCore engine.

    Attributes:
        GROUNDED: Verify claims against a provided source context (NLI-based).
        UNGROUNDED: Verify claims by retrieving evidence via web search.
        OFFLINE: Use only local models, no network access.
        AUTO: Automatically select mode based on available inputs.
    """

    GROUNDED = "grounded"
    UNGROUNDED = "ungrounded"
    OFFLINE = "offline"
    AUTO = "auto"


class Claim(BaseModel):
    """An atomic, independently-checkable factual claim.

    Each claim is extracted from an LLM response and represents a single,
    verifiable unit of factual content. Claims are the primary unit of
    analysis throughout the verification pipeline.

    Attributes:
        id: Unique identifier for this claim (short UUID prefix).
        text: The atomic claim text, suitable for NLI or search.
        source_span: Character offsets (start, end) in the original response.
        source_text: The exact text from the original response this claim
            was extracted from (may be longer than the atomic claim text).

    Example:
        >>> claim = Claim(
        ...     text="The Eiffel Tower is 330 meters tall.",
        ...     source_span=(0, 35),
        ...     source_text="The Eiffel Tower is 330 meters tall.",
        ... )
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Short unique identifier for this claim",
    )
    text: str = Field(..., description="The atomic claim text")
    source_span: tuple[int, int] = Field(
        ...,
        description="Character offsets (start, end) in the original LLM response",
    )
    source_text: str = Field(
        ...,
        description="Exact text from the original response this claim maps to",
    )


class ClaimVerdict(BaseModel):
    """Verification result for a single atomic claim.

    This is the central output artifact of the verification pipeline.
    Fields are populated progressively across phases:
        - Phase 2 populates: nli_score
        - Phase 3 populates: retrieval_score
        - Phase 4 populates: consistency_score
        - Phase 5 populates: confidence (fused from above signals)
        - Phase 6 populates: evidence, reason (explainability)

    Attributes:
        claim: The original atomic claim being verified.
        verdict: Three-way verdict: SUPPORTED, CONTRADICTED, or UNSUPPORTED.
        confidence: Calibrated confidence score in the verdict [0.0, 1.0].
        nli_score: Raw NLI entailment score from Phase 2 (optional).
        retrieval_score: Retrieval-based verification score from Phase 3 (optional).
        consistency_score: Self-consistency score from Phase 4 (optional).
        evidence: The evidence snippet supporting or contradicting the verdict.
        reason: Human-readable explanation of why this verdict was assigned.
        verification_mode: Which verification mode was used for this claim.
    """

    claim: Claim
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in verdict [0, 1]")
    nli_score: Optional[float] = Field(
        default=None, description="NLI entailment score from grounded verifier"
    )
    retrieval_score: Optional[float] = Field(
        default=None, description="Score from retrieval-based ungrounded verifier"
    )
    consistency_score: Optional[float] = Field(
        default=None, description="Self-consistency score across multiple generations"
    )
    evidence: Optional[str] = Field(
        default=None, description="Evidence snippet supporting or contradicting the claim"
    )
    reason: str = Field(..., description="Human-readable explanation of the verdict")
    verification_mode: VerificationMode


class VerificationReport(BaseModel):
    """Complete verification report for an LLM response.

    The top-level output artifact of VeritasCore. Produced by Phase 6 and
    consumed by Phase 7 (API/interfaces).

    Attributes:
        query: The original user query that prompted the LLM response (optional).
        response_text: The full LLM response text that was verified.
        claims: List of individual claim verdicts.
        overall_trust_score: Aggregate trust score for the full response [0.0, 1.0].
            1.0 = fully supported, 0.0 = fully contradicted/unsupported.
        overall_verdict: Aggregate verdict derived from individual claim verdicts.
        verification_mode: The mode used for this verification run.
        domain_profile: The domain profile applied (e.g. "general", "medical").
        processing_time_ms: Total end-to-end processing time in milliseconds.
        metadata: Arbitrary additional metadata (model names, versions, etc.).
    """

    query: Optional[str] = Field(
        default=None, description="Original user query"
    )
    response_text: str = Field(..., description="Full LLM response text that was verified")
    claims: list[ClaimVerdict] = Field(
        default_factory=list, description="Individual claim verdicts"
    )
    overall_trust_score: float = Field(
        ge=0.0, le=1.0, description="Aggregate trust score for the full response"
    )
    overall_verdict: Verdict
    verification_mode: VerificationMode
    domain_profile: str = Field(default="general", description="Domain profile applied")
    processing_time_ms: float = Field(description="Total processing time in milliseconds")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @property
    def n_supported(self) -> int:
        """Count of supported claims."""
        return sum(1 for c in self.claims if c.verdict == Verdict.SUPPORTED)

    @property
    def n_contradicted(self) -> int:
        """Count of contradicted claims."""
        return sum(1 for c in self.claims if c.verdict == Verdict.CONTRADICTED)

    @property
    def n_unsupported(self) -> int:
        """Count of unsupported claims."""
        return sum(1 for c in self.claims if c.verdict == Verdict.UNSUPPORTED)

    @property
    def n_claims(self) -> int:
        """Total number of claims."""
        return len(self.claims)
