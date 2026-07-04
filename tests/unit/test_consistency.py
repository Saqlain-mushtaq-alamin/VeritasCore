"""Unit tests for SemanticConsistencyChecker (Phase 4) using a stubbed
embedding model — no real sentence-transformers model is loaded.

The stub maps known text snippets to hand-picked unit vectors so cosine
similarity (dot product, since vectors are pre-normalized) is fully
deterministic and predictable in tests.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from veritascore.core.exceptions import VerificationError
from veritascore.core.types import Claim
from veritascore.verifier.consistency import ConsistencyResult, SemanticConsistencyChecker

FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "consistency_samples.json"


class _StubEmbeddingModel:
    """Deterministic fake embedding model.

    Maps text -> a fixed 2D unit vector via a lookup table (falling back to
    a hash-based pseudo-random unit vector for unknown text, so unlisted
    inputs don't crash but also don't collide by default).
    """

    def __init__(self, vectors: dict[str, np.ndarray] | None = None) -> None:
        self.vectors = vectors or {}

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        out = []
        for t in texts:
            if t in self.vectors:
                v = self.vectors[t]
            else:
                # Deterministic pseudo-random unit vector from hash, so
                # unknown text doesn't crash tests but is reproducible.
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                v = rng.normal(size=2)
            norm = np.linalg.norm(v)
            out.append(v / norm if norm > 0 else v)
        return np.array(out)

    def to(self, device: str) -> _StubEmbeddingModel:
        return self


def make_checker(
    vectors: dict[str, np.ndarray] | None = None,
    relevance_threshold: float = 0.3,
) -> SemanticConsistencyChecker:
    checker = SemanticConsistencyChecker(relevance_threshold=relevance_threshold)
    checker._model = _StubEmbeddingModel(vectors)
    checker._loaded = True
    return checker


# Two "topic directions" used across tests: same-topic vectors point
# roughly the same way (high cosine sim); different-topic vectors are
# roughly perpendicular or opposite (low/negative cosine sim).
FRANCE_TOPIC = np.array([1.0, 0.1])
PYTHON_TOPIC = np.array([0.05, 1.0])
OPPOSITE_OF_FRANCE = np.array([-1.0, -0.1])


@pytest.fixture
def sample_claim() -> Claim:
    return Claim(
        id="c1",
        text="Paris is the capital of France.",
        source_span=(0, 32),
        source_text="Paris is the capital of France.",
    )


# ── Query-Claim Relevance ──────────────────────────────────────────────────────

class TestQueryClaimRelevance:
    def test_on_topic_claim_high_score(self, sample_claim: Claim) -> None:
        """Claim directly answering the query -> high relevance score."""
        query = "What is the capital of France?"
        checker = make_checker({
            query: FRANCE_TOPIC,
            sample_claim.text: FRANCE_TOPIC,
            "irrelevant response text": FRANCE_TOPIC,
        })
        result = checker.check_consistency([sample_claim], query, "irrelevant response text")
        assert result.claim_scores[sample_claim.id] > 0.5

    def test_off_topic_claim_low_score(self) -> None:
        """Claim irrelevant to query -> low relevance score."""
        query = "What is the capital of France?"
        claim = Claim(
            id="c2", text="Python was created by Guido van Rossum.",
            source_span=(0, 39), source_text="x",
        )
        checker = make_checker({
            query: FRANCE_TOPIC,
            claim.text: PYTHON_TOPIC,
            "response": FRANCE_TOPIC,
        })
        result = checker.check_consistency([claim], query, "response")
        assert result.claim_scores[claim.id] < 0.3

    def test_partially_relevant(self) -> None:
        """Claim somewhat related but not directly answering -> medium score."""
        query = "What is the capital of France?"
        claim = Claim(
            id="c3", text="France has a population of 67 million.",
            source_span=(0, 38), source_text="x",
        )
        # A vector at ~45 degrees from FRANCE_TOPIC gives a mid-range cosine sim
        mid_vector = np.array([0.6, 0.6])
        checker = make_checker({
            query: FRANCE_TOPIC,
            claim.text: mid_vector,
            "response": FRANCE_TOPIC,
        })
        result = checker.check_consistency([claim], query, "response")
        assert 0.3 < result.claim_scores[claim.id] < 0.95

    def test_off_topic_detection(self) -> None:
        """Off-topic claims should be flagged in off_topic_claims list."""
        query = "What is the capital of France?"
        on_topic = Claim(id="on", text="Paris is the capital.", source_span=(0, 21), source_text="x")
        off_topic = Claim(id="off", text="Python is a language.", source_span=(0, 21), source_text="x")
        checker = make_checker({
            query: FRANCE_TOPIC,
            on_topic.text: FRANCE_TOPIC,
            off_topic.text: PYTHON_TOPIC,
            "response": FRANCE_TOPIC,
        }, relevance_threshold=0.3)
        result = checker.check_consistency([on_topic, off_topic], query, "response")
        assert off_topic.id in result.off_topic_claims
        assert on_topic.id not in result.off_topic_claims

    def test_clamping_does_not_hide_negative_similarity_from_threshold(self) -> None:
        """Even though per-claim scores are clamped to [0,1], the threshold
        check must use the RAW (unclamped) similarity, so a claim with
        negative similarity is still correctly flagged as off-topic rather
        than silently passing because its clamped score reads as 0.0 which
        could be misread as 'right at the boundary'."""
        query = "q"
        claim = Claim(id="c", text="opposite", source_span=(0, 8), source_text="x")
        checker = make_checker({
            query: FRANCE_TOPIC,
            claim.text: OPPOSITE_OF_FRANCE,
            "response": FRANCE_TOPIC,
        }, relevance_threshold=0.3)
        result = checker.check_consistency([claim], query, "response")
        assert claim.id in result.off_topic_claims
        assert result.claim_scores[claim.id] == 0.0  # clamped, but still flagged


# ── Coherence ──────────────────────────────────────────────────────────────────

class TestCoherence:
    def test_coherence_consistent_claims(self) -> None:
        """Claims on same topic -> high coherence."""
        claims = [
            Claim(id="a", text="claim a", source_span=(0, 7), source_text="x"),
            Claim(id="b", text="claim b", source_span=(0, 7), source_text="x"),
            Claim(id="c", text="claim c", source_span=(0, 7), source_text="x"),
        ]
        checker = make_checker({
            "query": FRANCE_TOPIC,
            "claim a": FRANCE_TOPIC,
            "claim b": FRANCE_TOPIC,
            "claim c": FRANCE_TOPIC,
            "response": FRANCE_TOPIC,
        })
        result = checker.check_consistency(claims, "query", "response")
        assert result.coherence_score > 0.8

    def test_coherence_mixed_claims(self) -> None:
        """Claims on wildly different topics -> low coherence."""
        claims = [
            Claim(id="a", text="claim a", source_span=(0, 7), source_text="x"),
            Claim(id="b", text="claim b", source_span=(0, 7), source_text="x"),
        ]
        checker = make_checker({
            "query": FRANCE_TOPIC,
            "claim a": FRANCE_TOPIC,
            "claim b": OPPOSITE_OF_FRANCE,
            "response": FRANCE_TOPIC,
        })
        result = checker.check_consistency(claims, "query", "response")
        assert result.coherence_score < 0.0

    def test_single_claim_is_vacuously_coherent(self, sample_claim: Claim) -> None:
        checker = make_checker({
            "query": FRANCE_TOPIC, sample_claim.text: PYTHON_TOPIC, "response": FRANCE_TOPIC,
        })
        result = checker.check_consistency([sample_claim], "query", "response")
        assert result.coherence_score == 1.0

    def test_coherence_distinguishes_consistent_vs_inconsistent(self) -> None:
        """G4 criterion 3: coherence score must actually distinguish the two cases."""
        consistent = [
            Claim(id="a", text="claim a", source_span=(0, 7), source_text="x"),
            Claim(id="b", text="claim b", source_span=(0, 7), source_text="x"),
        ]
        inconsistent = [
            Claim(id="c", text="claim c", source_span=(0, 7), source_text="x"),
            Claim(id="d", text="claim d", source_span=(0, 7), source_text="x"),
        ]
        checker = make_checker({
            "query": FRANCE_TOPIC,
            "claim a": FRANCE_TOPIC, "claim b": FRANCE_TOPIC,
            "claim c": FRANCE_TOPIC, "claim d": OPPOSITE_OF_FRANCE,
            "response": FRANCE_TOPIC,
        })
        consistent_result = checker.check_consistency(consistent, "query", "response")
        inconsistent_result = checker.check_consistency(inconsistent, "query", "response")
        assert consistent_result.coherence_score > inconsistent_result.coherence_score


# ── Empty / Edge Cases ──────────────────────────────────────────────────────────

class TestEmptyAndEdgeCases:
    def test_empty_claims(self) -> None:
        """Empty claim list -> default scores."""
        checker = make_checker({"query": FRANCE_TOPIC, "response": FRANCE_TOPIC})
        result = checker.check_consistency([], "query", "response")
        assert result.claim_scores == {}
        assert result.coherence_score == 1.0
        assert result.off_topic_claims == []

    def test_empty_claims_empty_response_defaults_relevance_to_one(self) -> None:
        checker = make_checker({"query": FRANCE_TOPIC})
        result = checker.check_consistency([], "query", "")
        assert result.response_relevance == 1.0

    def test_no_query_raises(self, sample_claim: Claim) -> None:
        """Should raise a clear error on missing query, not silently misbehave."""
        checker = make_checker()
        with pytest.raises(VerificationError):
            checker.check_consistency([sample_claim], "", "response")
        with pytest.raises(VerificationError):
            checker.check_consistency([sample_claim], None, "response")  # type: ignore[arg-type]
        with pytest.raises(VerificationError):
            checker.check_consistency([sample_claim], "   ", "response")

    def test_score_claim_no_query_raises(self, sample_claim: Claim) -> None:
        checker = make_checker()
        with pytest.raises(VerificationError):
            checker.score_claim(sample_claim, "")


# ── Response Relevance ──────────────────────────────────────────────────────────

class TestResponseRelevance:
    def test_response_relevance_computed(self, sample_claim: Claim) -> None:
        """Overall response relevance should be computed and present."""
        query = "What is the capital of France?"
        checker = make_checker({
            query: FRANCE_TOPIC,
            sample_claim.text: FRANCE_TOPIC,
            "Paris is the capital of France.": FRANCE_TOPIC,
        })
        result = checker.check_consistency(
            [sample_claim], query, "Paris is the capital of France."
        )
        assert result.response_relevance > 0.5

    def test_off_topic_response_has_low_relevance(self, sample_claim: Claim) -> None:
        query = "What is the capital of France?"
        checker = make_checker({
            query: FRANCE_TOPIC,
            sample_claim.text: FRANCE_TOPIC,
            "Python is a programming language.": PYTHON_TOPIC,
        })
        result = checker.check_consistency(
            [sample_claim], query, "Python is a programming language."
        )
        assert result.response_relevance < 0.3


# ── score_claim ──────────────────────────────────────────────────────────────────

class TestScoreClaim:
    def test_score_claim_matches_check_consistency(self, sample_claim: Claim) -> None:
        query = "What is the capital of France?"
        checker = make_checker({query: FRANCE_TOPIC, sample_claim.text: FRANCE_TOPIC})
        score = checker.score_claim(sample_claim, query)
        assert score > 0.5

    def test_score_claim_clamps_negative_to_zero(self) -> None:
        query = "q"
        claim = Claim(id="c", text="opposite", source_span=(0, 8), source_text="x")
        checker = make_checker({query: FRANCE_TOPIC, claim.text: OPPOSITE_OF_FRANCE})
        score = checker.score_claim(claim, query)
        assert score == 0.0


# ── Init / Validation ────────────────────────────────────────────────────────────

class TestInit:
    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            SemanticConsistencyChecker(relevance_threshold=1.5)
        with pytest.raises(ValueError):
            SemanticConsistencyChecker(relevance_threshold=-1.5)

    def test_valid_threshold_boundaries(self) -> None:
        SemanticConsistencyChecker(relevance_threshold=-1.0)
        SemanticConsistencyChecker(relevance_threshold=1.0)

    def test_consistency_result_to_dict(self) -> None:
        result = ConsistencyResult(
            claim_scores={"a": 0.8}, coherence_score=0.5,
            response_relevance=0.6, off_topic_claims=[],
        )
        d = result.to_dict()
        assert d == {
            "claim_scores": {"a": 0.8},
            "coherence_score": 0.5,
            "response_relevance": 0.6,
            "off_topic_claims": [],
        }

    def test_consistency_result_repr(self) -> None:
        result = ConsistencyResult(
            claim_scores={"a": 0.8}, coherence_score=0.5,
            response_relevance=0.6, off_topic_claims=["a"],
        )
        assert "n_claims=1" in repr(result)
        assert "n_off_topic=1" in repr(result)


# ── Lifecycle ──────────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_not_loaded_initially(self) -> None:
        checker = SemanticConsistencyChecker()
        assert checker._loaded is False

    def test_unload_when_not_loaded_is_noop(self) -> None:
        checker = SemanticConsistencyChecker()
        checker.unload()
        assert checker._loaded is False

    def test_unload_clears_model(self) -> None:
        checker = make_checker()
        checker.unload()
        assert checker._model is None
        assert checker._loaded is False

    def test_is_available_false_on_load_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        checker = SemanticConsistencyChecker()

        def fail_load() -> None:
            from veritascore.core.exceptions import ModelLoadError
            raise ModelLoadError("simulated failure")

        monkeypatch.setattr(checker, "_load_model", fail_load)
        assert checker.is_available() is False

    def test_resolve_device_auto_no_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import torch

        checker = SemanticConsistencyChecker()
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert checker._resolve_device() == "cpu"

    def test_resolve_device_explicit_cuda_unavailable_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import torch

        from veritascore.core.config import EngineConfig, ModelConfig

        config = EngineConfig(models=ModelConfig(device="cuda"))
        checker = SemanticConsistencyChecker(config=config)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert checker._resolve_device() == "cpu"

    def test_resolve_device_explicit_cpu(self) -> None:
        from veritascore.core.config import EngineConfig, ModelConfig

        config = EngineConfig(models=ModelConfig(device="cpu"))
        checker = SemanticConsistencyChecker(config=config)
        assert checker._resolve_device() == "cpu"


# ── Curated Fixture Evaluation (Quality Gate G4) ────────────────────────────────

class TestCuratedFixtureSpotCheck:
    """A lightweight structural spot-check on the fixture file itself —
    the full embedding-based evaluation runs via scripts/evaluate_consistency.py
    against the real model (see docs/phase4_evaluation_log.md)."""

    def test_fixture_file_loads(self) -> None:
        with open(FIXTURES_PATH) as f:
            data = json.load(f)
        assert len(data["samples"]) >= 10

    def test_fixture_has_required_fields(self) -> None:
        with open(FIXTURES_PATH) as f:
            data = json.load(f)
        for sample in data["samples"]:
            assert "query" in sample
            assert "on_topic_claims" in sample
            assert "off_topic_claims" in sample
            assert len(sample["on_topic_claims"]) >= 1
            assert len(sample["off_topic_claims"]) >= 1

    def test_fixture_total_claims_at_least_fifty(self) -> None:
        with open(FIXTURES_PATH) as f:
            data = json.load(f)
        total = sum(
            len(s["on_topic_claims"]) + len(s["off_topic_claims"])
            + len(s.get("partially_relevant_claims", []))
            for s in data["samples"]
        )
        assert total >= 50
