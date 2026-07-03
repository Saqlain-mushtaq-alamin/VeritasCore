"""Unit tests for veritascore.core.types (Phase 0).

Tests cover:
- Claim creation and field validation
- ClaimVerdict creation and optional fields
- VerificationReport creation and computed properties
- Verdict and VerificationMode enums
- Pydantic validation (confidence bounds, required fields)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from veritascore.core.types import (
    Claim,
    ClaimVerdict,
    Verdict,
    VerificationMode,
    VerificationReport,
)


class TestVerdict:
    def test_values(self) -> None:
        assert Verdict.SUPPORTED == "supported"
        assert Verdict.CONTRADICTED == "contradicted"
        assert Verdict.UNSUPPORTED == "unsupported"

    def test_from_string(self) -> None:
        assert Verdict("supported") == Verdict.SUPPORTED

    def test_invalid_string(self) -> None:
        with pytest.raises(ValueError):
            Verdict("invalid")


class TestVerificationMode:
    def test_values(self) -> None:
        assert VerificationMode.GROUNDED == "grounded"
        assert VerificationMode.UNGROUNDED == "ungrounded"
        assert VerificationMode.OFFLINE == "offline"
        assert VerificationMode.AUTO == "auto"


class TestClaim:
    def test_creation_minimal(self) -> None:
        claim = Claim(
            text="Paris is the capital of France.",
            source_span=(0, 30),
            source_text="Paris is the capital of France.",
        )
        assert claim.text == "Paris is the capital of France."
        assert claim.source_span == (0, 30)
        assert len(claim.id) > 0  # Auto-generated

    def test_custom_id(self) -> None:
        claim = Claim(
            id="my-id",
            text="Test.",
            source_span=(0, 5),
            source_text="Test.",
        )
        assert claim.id == "my-id"

    def test_id_auto_generated_unique(self) -> None:
        c1 = Claim(text="A.", source_span=(0, 2), source_text="A.")
        c2 = Claim(text="B.", source_span=(0, 2), source_text="B.")
        assert c1.id != c2.id

    def test_source_span_is_tuple(self) -> None:
        claim = Claim(text="X.", source_span=(5, 10), source_text="X.")
        assert claim.source_span[0] == 5
        assert claim.source_span[1] == 10

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValidationError):
            Claim(source_span=(0, 5), source_text="X.")  # type: ignore[call-arg]


class TestClaimVerdict:
    def test_creation_grounded(self, claim_supported: Claim) -> None:
        verdict = ClaimVerdict(
            claim=claim_supported,
            verdict=Verdict.SUPPORTED,
            confidence=0.95,
            nli_score=0.93,
            reason="Context confirms this claim.",
            verification_mode=VerificationMode.GROUNDED,
        )
        assert verdict.verdict == Verdict.SUPPORTED
        assert verdict.confidence == 0.95
        assert verdict.nli_score == 0.93
        assert verdict.retrieval_score is None
        assert verdict.consistency_score is None
        assert verdict.evidence is None

    def test_confidence_bounds(self, claim_supported: Claim) -> None:
        # Valid bounds
        ClaimVerdict(
            claim=claim_supported,
            verdict=Verdict.SUPPORTED,
            confidence=0.0,
            reason="Test.",
            verification_mode=VerificationMode.GROUNDED,
        )
        ClaimVerdict(
            claim=claim_supported,
            verdict=Verdict.SUPPORTED,
            confidence=1.0,
            reason="Test.",
            verification_mode=VerificationMode.GROUNDED,
        )

    def test_confidence_out_of_bounds(self, claim_supported: Claim) -> None:
        with pytest.raises(ValidationError):
            ClaimVerdict(
                claim=claim_supported,
                verdict=Verdict.SUPPORTED,
                confidence=1.1,  # Invalid
                reason="Test.",
                verification_mode=VerificationMode.GROUNDED,
            )

    def test_all_optional_scores(self, claim_supported: Claim) -> None:
        verdict = ClaimVerdict(
            claim=claim_supported,
            verdict=Verdict.SUPPORTED,
            confidence=0.8,
            nli_score=0.85,
            retrieval_score=0.75,
            consistency_score=0.90,
            evidence="Evidence snippet here.",
            reason="All signals agree.",
            verification_mode=VerificationMode.GROUNDED,
        )
        assert verdict.nli_score == 0.85
        assert verdict.retrieval_score == 0.75
        assert verdict.consistency_score == 0.90


class TestVerificationReport:
    def _make_report(
        self,
        response: str,
        claims: list[ClaimVerdict],
    ) -> VerificationReport:
        return VerificationReport(
            response_text=response,
            claims=claims,
            overall_trust_score=0.7,
            overall_verdict=Verdict.SUPPORTED,
            verification_mode=VerificationMode.GROUNDED,
            processing_time_ms=123.4,
        )

    def test_creation_minimal(self, sample_response: str) -> None:
        report = self._make_report(sample_response, [])
        assert report.query is None
        assert report.domain_profile == "general"
        assert report.metadata == {}
        assert report.n_claims == 0

    def test_with_query(self, sample_response: str) -> None:
        report = VerificationReport(
            query="Tell me about the Eiffel Tower.",
            response_text=sample_response,
            claims=[],
            overall_trust_score=0.8,
            overall_verdict=Verdict.SUPPORTED,
            verification_mode=VerificationMode.GROUNDED,
            processing_time_ms=50.0,
        )
        assert report.query == "Tell me about the Eiffel Tower."

    def test_computed_counts(
        self,
        sample_response: str,
        verdict_supported: ClaimVerdict,
        verdict_contradicted: ClaimVerdict,
        verdict_unsupported: ClaimVerdict,
    ) -> None:
        report = self._make_report(
            sample_response,
            [verdict_supported, verdict_contradicted, verdict_unsupported],
        )
        assert report.n_claims == 3
        assert report.n_supported == 1
        assert report.n_contradicted == 1
        assert report.n_unsupported == 1

    def test_trust_score_bounds(self, sample_response: str) -> None:
        with pytest.raises(ValidationError):
            self._make_report(sample_response, [])
            VerificationReport(
                response_text=sample_response,
                claims=[],
                overall_trust_score=1.5,  # Invalid
                overall_verdict=Verdict.SUPPORTED,
                verification_mode=VerificationMode.GROUNDED,
                processing_time_ms=1.0,
            )

    def test_metadata_dict(self, sample_response: str) -> None:
        report = VerificationReport(
            response_text=sample_response,
            claims=[],
            overall_trust_score=0.5,
            overall_verdict=Verdict.UNSUPPORTED,
            verification_mode=VerificationMode.UNGROUNDED,
            processing_time_ms=200.0,
            metadata={"model": "phi-3", "version": "0.0.1"},
        )
        assert report.metadata["model"] == "phi-3"
