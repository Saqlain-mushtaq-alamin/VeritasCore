"""Unit tests for NLIVerifier (Phase 2) using a mocked model.

No real model is loaded — _run_nli is monkeypatched directly, or the
underlying transformers calls are mocked, depending on what's being tested.
Real-model integration tests live in tests/integration/test_grounded_pipeline.py.
"""
# ruff: noqa: E501

from __future__ import annotations

import numpy as np
import pytest

from veritascore.core.exceptions import VerificationError
from veritascore.core.types import Claim, Verdict, VerificationMode
from veritascore.verifier.base import BaseVerifier
from veritascore.verifier.nli_verifier import EXPECTED_NLI_LABEL_ORDER, NLIVerifier
from veritascore.verifier.utils import (
    chunk_context,
    extract_evidence_snippet,
    preprocess_text,
    split_into_sentences,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def verifier() -> NLIVerifier:
    """NLIVerifier with model loading short-circuited (mark as loaded)."""
    v = NLIVerifier()
    v._loaded = True  # Skip _load_model() entirely
    v._device = "cpu"
    v._tokenizer = _FakeTokenizer()
    return v


@pytest.fixture
def sample_claim() -> Claim:
    return Claim(
        id="c1",
        text="The Eiffel Tower was built in 1889.",
        source_span=(0, 35),
        source_text="The Eiffel Tower was built in 1889.",
    )


class _FakeTokenizer:
    """Minimal fake tokenizer providing encode/decode for chunking tests."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # 1 "token" per word, deterministic and simple
        return list(range(len(text.split())))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"tok{i}" for i in ids)


# ── BaseVerifier Contract ───────────────────────────────────────────────────


class TestBaseVerifier:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseVerifier()  # type: ignore[abstract]

    def test_subclass_must_implement_verify(self) -> None:
        class Incomplete(BaseVerifier):
            def is_available(self) -> bool:
                return True

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_must_implement_is_available(self) -> None:
        class Incomplete(BaseVerifier):
            def verify(self, claims, context=None, query=None):  # type: ignore[no-untyped-def]
                return []

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


# ── NLIVerifier — Verdict Logic ──────────────────────────────────────────────


class TestNLIVerifierVerdicts:
    def test_supported_verdict(self, verifier: NLIVerifier, sample_claim: Claim) -> None:
        """High entailment score → SUPPORTED."""
        verifier._run_nli = lambda premise, hypothesis: np.array([0.05, 0.10, 0.85])
        verdicts = verifier.verify([sample_claim], context="The tower was completed in 1889.")
        assert verdicts[0].verdict == Verdict.SUPPORTED
        assert verdicts[0].nli_score > 0.7
        assert verdicts[0].confidence == pytest.approx(0.85)

    def test_contradicted_verdict(self, verifier: NLIVerifier, sample_claim: Claim) -> None:
        """High contradiction score → CONTRADICTED."""
        verifier._run_nli = lambda premise, hypothesis: np.array([0.80, 0.10, 0.10])
        verdicts = verifier.verify([sample_claim], context="The tower was built in 1900.")
        assert verdicts[0].verdict == Verdict.CONTRADICTED
        assert verdicts[0].confidence == pytest.approx(0.80)

    def test_unsupported_verdict(self, verifier: NLIVerifier, sample_claim: Claim) -> None:
        """Low entailment and contradiction → UNSUPPORTED."""
        verifier._run_nli = lambda premise, hypothesis: np.array([0.15, 0.60, 0.25])
        verdicts = verifier.verify([sample_claim], context="Paris is a beautiful city.")
        assert verdicts[0].verdict == Verdict.UNSUPPORTED

    def test_no_context_raises_error(self, verifier: NLIVerifier, sample_claim: Claim) -> None:
        with pytest.raises(VerificationError):
            verifier.verify([sample_claim], context=None)

    def test_empty_context_raises_error(self, verifier: NLIVerifier, sample_claim: Claim) -> None:
        with pytest.raises(VerificationError):
            verifier.verify([sample_claim], context="   ")

    def test_empty_claims_returns_empty(self, verifier: NLIVerifier) -> None:
        assert verifier.verify([], context="Some context.") == []

    def test_evidence_included(self, verifier: NLIVerifier, sample_claim: Claim) -> None:
        verifier._run_nli = lambda premise, hypothesis: np.array([0.05, 0.10, 0.85])
        verdicts = verifier.verify([sample_claim], context="The tower was completed in 1889.")
        assert verdicts[0].evidence is not None
        assert len(verdicts[0].evidence) > 0

    def test_verification_mode_is_grounded(
        self, verifier: NLIVerifier, sample_claim: Claim
    ) -> None:
        verifier._run_nli = lambda premise, hypothesis: np.array([0.05, 0.10, 0.85])
        verdicts = verifier.verify([sample_claim], context="Some context.")
        assert all(v.verification_mode == VerificationMode.GROUNDED for v in verdicts)

    def test_multiple_claims(self, verifier: NLIVerifier) -> None:
        claims = [
            Claim(id="c1", text="Fact 1.", source_span=(0, 7), source_text="Fact 1."),
            Claim(id="c2", text="Fact 2.", source_span=(8, 15), source_text="Fact 2."),
        ]
        verifier._run_nli = lambda premise, hypothesis: np.array([0.1, 0.1, 0.8])
        verdicts = verifier.verify(claims, context="Context text.")
        assert len(verdicts) == 2
        assert verdicts[0].claim.id == "c1"
        assert verdicts[1].claim.id == "c2"

    def test_threshold_boundary_entailment(
        self, verifier: NLIVerifier, sample_claim: Claim
    ) -> None:
        """Exactly at threshold should count as SUPPORTED (>=)."""
        verifier.entailment_threshold = 0.7
        verifier._run_nli = lambda premise, hypothesis: np.array([0.1, 0.2, 0.7])
        verdicts = verifier.verify([sample_claim], context="Context.")
        assert verdicts[0].verdict == Verdict.SUPPORTED

    def test_threshold_boundary_contradiction(
        self, verifier: NLIVerifier, sample_claim: Claim
    ) -> None:
        verifier.contradiction_threshold = 0.5
        verifier._run_nli = lambda premise, hypothesis: np.array([0.5, 0.3, 0.2])
        verdicts = verifier.verify([sample_claim], context="Context.")
        assert verdicts[0].verdict == Verdict.CONTRADICTED

    def test_entailment_takes_priority_when_both_high(
        self, verifier: NLIVerifier, sample_claim: Claim
    ) -> None:
        """If both entailment and contradiction exceed threshold, entailment wins (checked first)."""
        verifier._run_nli = lambda premise, hypothesis: np.array([0.75, 0.05, 0.80])
        verdicts = verifier.verify([sample_claim], context="Context.")
        assert verdicts[0].verdict == Verdict.SUPPORTED

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            NLIVerifier(entailment_threshold=1.5)
        with pytest.raises(ValueError):
            NLIVerifier(contradiction_threshold=-0.1)


# ── Consistency: best-chunk selection picks matching probs/evidence ─────────


class TestNLIVerifierChunkConsistency:
    """Regression tests for the chunk/probs consistency bug present in
    naive implementations that track entailment-max and contradiction-max
    independently (can select a chunk for evidence that doesn't match the
    probs used to derive the verdict)."""

    def test_strongest_signal_chunk_is_selected_entailment(
        self, verifier: NLIVerifier, sample_claim: Claim
    ) -> None:
        # Chunk 1: weak entailment. Chunk 2: strong entailment (the real evidence).
        # use_bidirectional=False so _verify_single_claim is used; the reverse NLI
        # pass in bidirectional mode would call _run_nli(premise=claim.text, ...)
        # which is not in the chunk-keyed responses dict.
        verifier.use_bidirectional = False
        responses = {
            "chunk one weak signal": np.array([0.1, 0.7, 0.2]),
            "chunk two strong signal": np.array([0.05, 0.05, 0.90]),
        }
        verifier._run_nli = lambda premise, hypothesis: responses[premise]
        verifier._chunk_context = lambda ctx: ["chunk one weak signal", "chunk two strong signal"]

        verdicts = verifier.verify([sample_claim], context="irrelevant — chunking mocked")
        assert verdicts[0].verdict == Verdict.SUPPORTED
        assert verdicts[0].nli_score == pytest.approx(0.90)

    def test_strongest_signal_chunk_is_selected_contradiction(
        self, verifier: NLIVerifier, sample_claim: Claim
    ) -> None:
        # use_bidirectional=False: same reasoning as the entailment test above.
        verifier.use_bidirectional = False
        responses = {
            "chunk one mild contra": np.array([0.4, 0.3, 0.3]),
            "chunk two strong contra": np.array([0.85, 0.1, 0.05]),
        }
        verifier._run_nli = lambda premise, hypothesis: responses[premise]
        verifier._chunk_context = lambda ctx: ["chunk one mild contra", "chunk two strong contra"]

        verdicts = verifier.verify([sample_claim], context="irrelevant — chunking mocked")
        assert verdicts[0].verdict == Verdict.CONTRADICTED
        assert verdicts[0].confidence == pytest.approx(0.85)

    def test_no_chunks_falls_back_to_unsupported(
        self, verifier: NLIVerifier, sample_claim: Claim
    ) -> None:
        verifier._chunk_context = lambda ctx: []
        verdicts = verifier.verify([sample_claim], context="x")
        assert verdicts[0].verdict == Verdict.UNSUPPORTED


# ── verify_batch Equivalence ──────────────────────────────────────────────────


class TestVerifyBatchEquivalence:
    def test_batch_matches_sequential(self, verifier: NLIVerifier) -> None:
        """verify_batch() must produce identical verdicts to verify()."""
        import unittest.mock as mock

        claims = [
            Claim(id="c1", text="Claim one.", source_span=(0, 10), source_text="Claim one."),
            Claim(id="c2", text="Claim two.", source_span=(11, 21), source_text="Claim two."),
        ]
        context = "Some context sentence one. Some context sentence two."

        verifier._chunk_context = lambda ctx: [context]
        verifier._run_nli = lambda premise, hypothesis: (
            np.array([0.05, 0.10, 0.85]) if "one" in hypothesis else np.array([0.70, 0.20, 0.10])
        )
        sequential = verifier.verify(claims, context=context)

        # For verify_batch: mock tokenizer + model + torch.softmax so that the
        # batch path returns the SAME two probability rows the sequential
        # mock above produced (claim "one" -> high entailment, claim "two"
        # -> high contradiction), one chunk per claim (since _chunk_context
        # is mocked to return a single chunk).
        expected_rows = np.array(
            [
                [0.05, 0.10, 0.85],
                [0.70, 0.20, 0.10],
            ]
        )

        fake_tokenizer = mock.MagicMock(
            return_value={"input_ids": _FakeTensor(), "attention_mask": _FakeTensor()}
        )
        fake_model = mock.MagicMock(return_value=mock.MagicMock(logits=expected_rows))
        verifier._tokenizer = fake_tokenizer
        verifier._model = fake_model

        with mock.patch("torch.no_grad"), mock.patch("torch.softmax") as mock_softmax:
            fake_softmax_result = mock.MagicMock()
            fake_softmax_result.cpu.return_value.numpy.return_value = expected_rows
            mock_softmax.return_value = fake_softmax_result

            batch = verifier.verify_batch(claims, context=context, batch_size=16)

        assert len(batch) == len(sequential)
        for b, s in zip(batch, sequential, strict=True):
            assert b.verdict == s.verdict
            assert b.nli_score == pytest.approx(s.nli_score)


class _FakeTensor:
    """Minimal fake tensor supporting .to() for device transfer in tests."""

    def __init__(self, batch_size: int = 1) -> None:
        self.batch_size = batch_size

    def to(self, device: str) -> _FakeTensor:
        return self


# ── Context Chunking ──────────────────────────────────────────────────────────


class TestChunkContext:
    def test_short_context_not_chunked(self) -> None:
        tok = _FakeTokenizer()
        chunks = chunk_context("short context", tok, max_tokens=512)
        assert chunks == ["short context"]

    def test_long_context_is_chunked(self) -> None:
        tok = _FakeTokenizer()
        long_text = " ".join(f"word{i}" for i in range(1000))
        chunks = chunk_context(long_text, tok, max_tokens=100, overlap_tokens=10)
        assert len(chunks) > 1

    def test_empty_context_returns_empty_list(self) -> None:
        tok = _FakeTokenizer()
        assert chunk_context("", tok) == []
        assert chunk_context("   ", tok) == []

    def test_overlap_exceeds_budget_raises(self) -> None:
        tok = _FakeTokenizer()
        with pytest.raises(ValueError):
            chunk_context("text", tok, max_tokens=100, overlap_tokens=60, reserved_tokens=50)

    def test_reserved_exceeds_max_raises(self) -> None:
        tok = _FakeTokenizer()
        with pytest.raises(ValueError):
            chunk_context("text", tok, max_tokens=50, reserved_tokens=60)

    def test_verifier_chunk_context_method(self, verifier: NLIVerifier) -> None:
        verifier.max_context_tokens = 150
        verifier.chunk_overlap_tokens = 5
        long_text = " ".join(f"word{i}" for i in range(200))
        chunks = verifier._chunk_context(long_text)
        assert len(chunks) > 1


# ── Evidence Extraction ────────────────────────────────────────────────────────


class TestExtractEvidenceSnippet:
    def test_finds_best_matching_sentence(self) -> None:
        chunk = "Paris is in France. It has many museums. The Louvre is famous."
        snippet = extract_evidence_snippet(chunk, "The Louvre museum is in Paris.")
        assert "louvre" in snippet.lower() or "paris" in snippet.lower()

    def test_empty_chunk_returns_empty(self) -> None:
        assert extract_evidence_snippet("", "some claim") == ""

    def test_truncates_to_max_length(self) -> None:
        long_sentence = "word " * 200 + "."
        snippet = extract_evidence_snippet(long_sentence, "word", max_length=50)
        assert len(snippet) <= 50

    def test_no_overlap_returns_first_sentence(self) -> None:
        chunk = "Cats are mammals. Dogs are mammals too."
        snippet = extract_evidence_snippet(chunk, "Quantum entanglement is strange.")
        assert snippet in ("Cats are mammals.", "Dogs are mammals too.")


# ── Text Utilities ────────────────────────────────────────────────────────────


class TestPreprocessText:
    def test_collapses_whitespace(self) -> None:
        assert preprocess_text("  Hello   world  \n") == "Hello world"

    def test_empty_string(self) -> None:
        assert preprocess_text("") == ""
        assert preprocess_text("   ") == ""


class TestSplitIntoSentencesUtil:
    def test_basic_split(self) -> None:
        sentences = split_into_sentences("One. Two. Three.")
        assert sentences == ["One.", "Two.", "Three."]

    def test_abbreviation_handling(self) -> None:
        sentences = split_into_sentences("Dr. Smith is here. He works.")
        assert len(sentences) == 2

    def test_empty_input(self) -> None:
        assert split_into_sentences("") == []
        assert split_into_sentences("   ") == []


# ── Lifecycle: load / unload ──────────────────────────────────────────────────


class TestNLIVerifierLifecycle:
    def test_not_loaded_initially(self) -> None:
        v = NLIVerifier()
        assert v._loaded is False

    def test_unload_when_not_loaded_is_noop(self) -> None:
        v = NLIVerifier()
        v.unload()  # Should not raise
        assert v._loaded is False

    def test_unload_clears_model_and_tokenizer(self, verifier: NLIVerifier) -> None:
        verifier._model = object()
        verifier.unload()
        assert verifier._model is None
        assert verifier._tokenizer is None
        assert verifier._loaded is False

    def test_label_order_default(self) -> None:
        v = NLIVerifier()
        assert v._label_index == {"contradiction": 0, "neutral": 1, "entailment": 2}
        assert EXPECTED_NLI_LABEL_ORDER == ("contradiction", "neutral", "entailment")

    def test_discover_label_order_from_config(self) -> None:
        import unittest.mock as mock

        v = NLIVerifier()
        v._model = mock.MagicMock()
        v._model.config.id2label = {0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}
        v._discover_label_order()
        assert v._label_index == {"entailment": 0, "neutral": 1, "contradiction": 2}

    def test_discover_label_order_falls_back_on_unknown_labels(self) -> None:
        import unittest.mock as mock

        v = NLIVerifier()
        v._model = mock.MagicMock()
        v._model.config.id2label = {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}
        original = dict(v._label_index)
        v._discover_label_order()
        assert v._label_index == original  # unchanged — fallback to default
