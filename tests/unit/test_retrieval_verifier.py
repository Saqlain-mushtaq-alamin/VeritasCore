"""Unit tests for RetrievalVerifier (Phase 3) using mocked retriever + NLI.

No real network calls and no real model loading — both the retriever and
the NLI verifier's _run_nli are replaced with test doubles.
"""
# ruff: noqa: E501

from __future__ import annotations

import numpy as np
import pytest

from veritascore.core.types import Claim, Verdict, VerificationMode
from veritascore.retriever.base import BaseRetriever, SearchResult
from veritascore.verifier.retrieval_verifier import RetrievalVerifier


class _StubRetriever(BaseRetriever):
    """Test double returning pre-programmed results for any query."""

    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results if results is not None else []
        self.queries_received: list[str] = []

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        self.queries_received.append(query)
        return self._results[:max_results]

    def is_available(self) -> bool:
        return True


class _StubNLIVerifier:
    """Test double for NLIVerifier — _run_nli returns pre-programmed probs
    based on a lookup keyed by the evidence snippet."""

    def __init__(self, probs_by_snippet: dict[str, np.ndarray] | None = None) -> None:
        self._probs_by_snippet = probs_by_snippet or {}
        self._default = np.array([0.2, 0.6, 0.2])

    def _load_model(self) -> None:
        pass

    def is_available(self) -> bool:
        return True

    def _run_nli(self, premise: str, hypothesis: str) -> np.ndarray:
        return self._probs_by_snippet.get(premise, self._default)


@pytest.fixture
def sample_claim() -> Claim:
    return Claim(
        id="c1",
        text="The Eiffel Tower was built in 1889.",
        source_span=(0, 35),
        source_text="The Eiffel Tower was built in 1889.",
    )


def make_verifier(
    results: list[SearchResult] | None = None,
    probs_by_snippet: dict[str, np.ndarray] | None = None,
    agreement_threshold: float = 0.6,
) -> RetrievalVerifier:
    retriever = _StubRetriever(results)
    nli = _StubNLIVerifier(probs_by_snippet)
    return RetrievalVerifier(
        retriever=retriever,
        nli_verifier=nli,  # type: ignore[arg-type]
        agreement_threshold=agreement_threshold,
    )


# ── Query Formulation ──────────────────────────────────────────────────────────


class TestQueryFormulation:
    def test_simple_claim(self) -> None:
        verifier = make_verifier()
        claim = Claim(text="Paris is the capital of France.", source_span=(0, 32), source_text="x")
        query = verifier._formulate_query(claim, None)
        assert query == "Paris is the capital of France."

    def test_removes_hedge_prefixes(self) -> None:
        verifier = make_verifier()
        claim = Claim(
            text="It is the case that water boils at 100C.", source_span=(0, 41), source_text="x"
        )
        query = verifier._formulate_query(claim, None)
        assert not query.lower().startswith("it is")

    def test_removes_there_is_prefix(self) -> None:
        verifier = make_verifier()
        claim = Claim(
            text="There is a large desert in Africa.", source_span=(0, 35), source_text="x"
        )
        query = verifier._formulate_query(claim, None)
        assert not query.lower().startswith("there is")

    def test_caps_query_length(self) -> None:
        verifier = make_verifier()
        long_text = "A" * 300
        claim = Claim(text=long_text, source_span=(0, 300), source_text=long_text)
        query = verifier._formulate_query(claim, None)
        assert len(query) <= 200

    def test_no_prefix_match_unchanged(self) -> None:
        verifier = make_verifier()
        claim = Claim(
            text="Mount Everest is the tallest mountain.", source_span=(0, 39), source_text="x"
        )
        query = verifier._formulate_query(claim, None)
        assert query == "Mount Everest is the tallest mountain."


# ── Evidence Aggregation ────────────────────────────────────────────────────────


class TestEvidenceAggregation:
    def test_strong_single_source_support(self, sample_claim: Claim) -> None:
        results = [
            SearchResult(
                title="T",
                url="http://x.com",
                snippet="The tower was completed in 1889.",
                relevance_score=0.9,
            )
        ]
        probs = {"The tower was completed in 1889.": np.array([0.05, 0.10, 0.90])}
        verifier = make_verifier(results, probs)
        verdicts = verifier.verify([sample_claim])
        assert verdicts[0].verdict == Verdict.SUPPORTED

    def test_majority_support(self, sample_claim: Claim) -> None:
        """3/3 sources support -> SUPPORTED."""
        results = [
            SearchResult(title="A", url="http://a.com", snippet="snippet a", relevance_score=0.9),
            SearchResult(title="B", url="http://b.com", snippet="snippet b", relevance_score=0.9),
            SearchResult(title="C", url="http://c.com", snippet="snippet c", relevance_score=0.9),
        ]
        probs = {
            "snippet a": np.array([0.05, 0.10, 0.85]),
            "snippet b": np.array([0.05, 0.15, 0.80]),
            "snippet c": np.array([0.10, 0.15, 0.75]),
        }
        verifier = make_verifier(results, probs)
        verdicts = verifier.verify([sample_claim])
        assert verdicts[0].verdict == Verdict.SUPPORTED

    def test_majority_contradict(self, sample_claim: Claim) -> None:
        """Strong contradiction signal -> CONTRADICTED."""
        results = [
            SearchResult(title="A", url="http://a.com", snippet="snippet a", relevance_score=0.9),
            SearchResult(title="B", url="http://b.com", snippet="snippet b", relevance_score=0.9),
        ]
        probs = {
            "snippet a": np.array([0.85, 0.10, 0.05]),
            "snippet b": np.array([0.80, 0.10, 0.10]),
        }
        verifier = make_verifier(results, probs)
        verdicts = verifier.verify([sample_claim])
        assert verdicts[0].verdict == Verdict.CONTRADICTED

    def test_no_agreement(self, sample_claim: Claim) -> None:
        """Mixed/weak signals -> UNSUPPORTED."""
        results = [
            SearchResult(title="A", url="http://a.com", snippet="snippet a", relevance_score=0.9),
        ]
        probs = {"snippet a": np.array([0.3, 0.4, 0.3])}
        verifier = make_verifier(results, probs)
        verdicts = verifier.verify([sample_claim])
        assert verdicts[0].verdict == Verdict.UNSUPPORTED

    def test_no_results(self, sample_claim: Claim) -> None:
        """Empty results -> UNSUPPORTED with explanatory reason."""
        verifier = make_verifier(results=[])
        verdicts = verifier.verify([sample_claim])
        assert verdicts[0].verdict == Verdict.UNSUPPORTED
        assert "no evidence" in verdicts[0].reason.lower()
        assert verdicts[0].retrieval_score == 0.0

    def test_results_with_empty_snippets_treated_as_no_evidence(self, sample_claim: Claim) -> None:
        results = [SearchResult(title="T", url="http://x.com", snippet="   ")]
        verifier = make_verifier(results)
        verdicts = verifier.verify([sample_claim])
        assert verdicts[0].verdict == Verdict.UNSUPPORTED

    def test_evidence_includes_source_url(self, sample_claim: Claim) -> None:
        results = [
            SearchResult(
                title="T", url="http://example.com/page", snippet="snippet", relevance_score=0.9
            )
        ]
        probs = {"snippet": np.array([0.05, 0.10, 0.85])}
        verifier = make_verifier(results, probs)
        verdicts = verifier.verify([sample_claim])
        assert "http://example.com/page" in verdicts[0].evidence

    def test_verification_mode_is_ungrounded(self, sample_claim: Claim) -> None:
        results = [
            SearchResult(title="T", url="http://x.com", snippet="snippet", relevance_score=0.9)
        ]
        probs = {"snippet": np.array([0.05, 0.10, 0.85])}
        verifier = make_verifier(results, probs)
        verdicts = verifier.verify([sample_claim])
        assert verdicts[0].verification_mode == VerificationMode.UNGROUNDED

    def test_relevance_weighting_affects_outcome(self, sample_claim: Claim) -> None:
        """A high-entailment but low-relevance source should contribute less
        than the relevance weighting implies; verify weighting is applied."""
        results = [
            SearchResult(
                title="Low relevance", url="http://low.com", snippet="low rel", relevance_score=0.1
            ),
        ]
        probs = {"low rel": np.array([0.05, 0.10, 0.95])}
        verifier = make_verifier(results, probs, agreement_threshold=0.5)
        verdicts = verifier.verify([sample_claim])
        # entailment 0.95 * relevance 0.1 = 0.095, well below threshold -> UNSUPPORTED
        assert verdicts[0].verdict == Verdict.UNSUPPORTED

    def test_multiple_claims_independent(self) -> None:
        claims = [
            Claim(id="c1", text="Claim one.", source_span=(0, 10), source_text="Claim one."),
            Claim(id="c2", text="Claim two.", source_span=(11, 21), source_text="Claim two."),
        ]
        results = [
            SearchResult(title="T", url="http://x.com", snippet="snippet", relevance_score=0.9)
        ]
        probs = {"snippet": np.array([0.05, 0.10, 0.85])}
        verifier = make_verifier(results, probs)
        verdicts = verifier.verify(claims)
        assert len(verdicts) == 2
        assert verdicts[0].claim.id == "c1"
        assert verdicts[1].claim.id == "c2"

    def test_evidence_and_score_consistency(self, sample_claim: Claim) -> None:
        """The reported nli_score must come from the SAME source as the
        reported evidence (regression test for the chunk-consistency
        bug class fixed in Phase 2)."""
        results = [
            SearchResult(
                title="Weak", url="http://weak.com", snippet="weak evidence", relevance_score=0.9
            ),
            SearchResult(
                title="Strong",
                url="http://strong.com",
                snippet="strong evidence",
                relevance_score=0.9,
            ),
        ]
        probs = {
            "weak evidence": np.array([0.1, 0.7, 0.2]),
            "strong evidence": np.array([0.02, 0.03, 0.95]),
        }
        verifier = make_verifier(results, probs)
        verdicts = verifier.verify([sample_claim])
        assert "strong.com" in verdicts[0].evidence
        assert verdicts[0].nli_score == pytest.approx(0.95)


# ── Init / Validation ──────────────────────────────────────────────────────────


class TestRetrievalVerifierInit:
    def test_invalid_agreement_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            RetrievalVerifier(agreement_threshold=1.5, nli_verifier=_StubNLIVerifier())  # type: ignore[arg-type]

    def test_empty_claims_returns_empty(self) -> None:
        verifier = make_verifier()
        assert verifier.verify([]) == []

    def test_is_available_checks_both(self) -> None:
        verifier = make_verifier()
        assert verifier.is_available() is True
