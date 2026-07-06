"""Pydantic models for the VeritasCore REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VerifyRequest(BaseModel):
    response: str = Field(..., min_length=1)
    query: str | None = None
    context: str | None = None
    mode: str = "auto"
    domain: str = "general"


class ClaimVerdictResponse(BaseModel):
    claim_id: str
    claim_text: str
    source_span: tuple[int, int]
    verdict: str
    confidence: float
    nli_score: float | None = None
    retrieval_score: float | None = None
    consistency_score: float | None = None
    evidence: str | None = None
    reason: str
    verification_mode: str


class VerifyResponse(BaseModel):
    query: str | None = None
    response_text: str
    overall_trust_score: float
    overall_verdict: str
    verification_mode: str
    domain_profile: str
    processing_time_ms: float
    n_claims: int
    n_supported: int
    n_contradicted: int
    n_unsupported: int
    claims: list[ClaimVerdictResponse]
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    version: str
    mode: str


class ProfilesResponse(BaseModel):
    profiles: list[str]


def report_to_response(report: Any) -> VerifyResponse:
    """Convert internal VerificationReport to VerifyResponse."""
    claims_resp = [
        ClaimVerdictResponse(
            claim_id=v.claim.id,
            claim_text=v.claim.text,
            source_span=v.claim.source_span,
            verdict=v.verdict.value,
            confidence=v.confidence,
            nli_score=v.nli_score,
            retrieval_score=v.retrieval_score,
            consistency_score=v.consistency_score,
            evidence=v.evidence,
            reason=v.reason,
            verification_mode=v.verification_mode.value,
        )
        for v in report.claims
    ]
    return VerifyResponse(
        query=report.query,
        response_text=report.response_text,
        overall_trust_score=report.overall_trust_score,
        overall_verdict=report.overall_verdict.value,
        verification_mode=report.verification_mode.value,
        domain_profile=report.domain_profile,
        processing_time_ms=report.processing_time_ms,
        n_claims=report.n_claims,
        n_supported=report.n_supported,
        n_contradicted=report.n_contradicted,
        n_unsupported=report.n_unsupported,
        claims=claims_resp,
        metadata=report.metadata,
    )
