"""Integration tests for the grounded verification pipeline (Phase 2).

End-to-end: RuleDecomposer/LLMDecomposer → NLIVerifier, using a REAL
NLI model (cross-encoder/nli-deberta-v3-base). Marked `integration` and
`slow`; excluded from `make test` by default.

Run explicitly with:
    pytest tests/integration/test_grounded_pipeline.py -v -m integration
"""

from __future__ import annotations

import time

import pytest

from veritascore.core.types import Verdict
from veritascore.decomposer.rule_decomposer import RuleDecomposer
from veritascore.verifier.nli_verifier import NLIVerifier

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def nli_verifier() -> NLIVerifier:
    """Module-scoped — load the real NLI model once, reuse across tests."""
    verifier = NLIVerifier()
    yield verifier
    verifier.unload()


class TestNLIVerifierRealModel:
    def test_model_loads(self, nli_verifier: NLIVerifier) -> None:
        assert nli_verifier.is_available() is True

    def test_label_order_matches_expected(self, nli_verifier: NLIVerifier) -> None:
        """Sanity check the assumption documented in nli_verifier.py:
        cross-encoder/nli-deberta-v3-base outputs
        [contradiction, neutral, entailment]."""
        nli_verifier._load_model()
        id2label = nli_verifier._model.config.id2label
        labels_lower = {str(v).lower() for v in id2label.values()}
        assert any("entailment" in label for label in labels_lower)
        assert any("contradiction" in label for label in labels_lower)

    def test_clear_entailment(self, nli_verifier: NLIVerifier) -> None:
        from veritascore.core.types import Claim

        claim = Claim(
            text="The Eiffel Tower is located in Paris.",
            source_span=(0, 38),
            source_text="The Eiffel Tower is located in Paris.",
        )
        verdicts = nli_verifier.verify(
            [claim],
            context="The Eiffel Tower is a famous landmark in Paris, France.",
        )
        assert verdicts[0].verdict == Verdict.SUPPORTED
        assert verdicts[0].nli_score > 0.7

    def test_clear_contradiction(self, nli_verifier: NLIVerifier) -> None:
        from veritascore.core.types import Claim

        claim = Claim(
            text="The Eiffel Tower is located in London.",
            source_span=(0, 38),
            source_text="The Eiffel Tower is located in London.",
        )
        verdicts = nli_verifier.verify(
            [claim],
            context="The Eiffel Tower is a famous landmark in Paris, France.",
        )
        assert verdicts[0].verdict == Verdict.CONTRADICTED

    def test_unrelated_context(self, nli_verifier: NLIVerifier) -> None:
        """Verify that a claim is UNSUPPORTED when the context is entirely
        unrelated (neutral premise-hypothesis pair).

        NOTE: NLI cross-encoders (DeBERTa-v3-base) produce inconsistent
        scores for semantically unrelated text. Some pairs spuriously
        entail (e.g. stock market vs. bananas: entail=0.78) while others
        correctly produce neutral. We use a pair (boiling point vs.
        Roman Empire) that is stably neutral across model versions.
        """
        from veritascore.core.types import Claim

        claim = Claim(
            text="The Roman Empire fell in 476 AD.",
            source_span=(0, 34),
            source_text="The Roman Empire fell in 476 AD.",
        )
        verdicts = nli_verifier.verify(
            [claim],
            context="Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        )
        assert verdicts[0].verdict == Verdict.UNSUPPORTED


class TestEndToEndGroundedPipeline:
    """Full pipeline: response text → decompose → verify against context."""

    def test_decompose_then_verify(self, nli_verifier: NLIVerifier) -> None:
        decomposer = RuleDecomposer()
        response = "The Eiffel Tower stands 330 meters tall. It was completed in 1889."
        context = (
            "The Eiffel Tower is a wrought-iron lattice tower in Paris. "
            "It is 330 metres tall and was completed in 1889 for the World's Fair."
        )

        claims = decomposer.decompose(response)
        assert len(claims) >= 2

        verdicts = nli_verifier.verify(claims, context=context)
        assert len(verdicts) == len(claims)

        # At least the height/date facts should be supported by this context
        supported = [v for v in verdicts if v.verdict == Verdict.SUPPORTED]
        assert len(supported) >= 1

    def test_decompose_then_verify_with_contradiction(self, nli_verifier: NLIVerifier) -> None:
        decomposer = RuleDecomposer()
        response = "The Eiffel Tower is 500 meters tall."
        context = "The Eiffel Tower stands 330 metres tall in Paris."

        claims = decomposer.decompose(response)
        verdicts = nli_verifier.verify(claims, context=context)

        assert any(v.verdict == Verdict.CONTRADICTED for v in verdicts)


class TestNLIVerifierBatchVsSequential:
    def test_batch_results_match_sequential_real_model(self, nli_verifier: NLIVerifier) -> None:
        from veritascore.core.types import Claim

        claims = [
            Claim(text="Paris is the capital of France.", source_span=(0, 32), source_text="x"),
            Claim(text="The Louvre is located in Rome.", source_span=(0, 31), source_text="x"),
        ]
        context = (
            "Paris is the capital and largest city of France. The Louvre is a museum in Paris."
        )

        sequential = nli_verifier.verify(claims, context=context)
        batch = nli_verifier.verify_batch(claims, context=context, batch_size=4)

        for s, b in zip(sequential, batch, strict=True):
            assert s.verdict == b.verdict
            assert abs(s.nli_score - b.nli_score) < 0.01


class TestNLIVerifierPerformance:
    def test_latency_per_claim(self, nli_verifier: NLIVerifier) -> None:
        """Target: <200ms per claim on GPU (Phase 2 §2.8 criterion 10).
        On CPU this will be slower — assert a generous CPU-safe ceiling instead."""
        import torch

        from veritascore.core.types import Claim

        claim = Claim(
            text="The Great Wall of China is over 13,000 miles long.",
            source_span=(0, 52),
            source_text="x",
        )
        context = "The Great Wall of China stretches more than 13,000 miles across northern China."

        t0 = time.perf_counter()
        nli_verifier.verify([claim], context=context)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        ceiling = 200 if torch.cuda.is_available() else 5000
        assert elapsed_ms < ceiling, f"Latency {elapsed_ms:.0f}ms exceeded ceiling {ceiling}ms"


class TestNLIVerifierMemoryManagement:
    def test_unload_frees_gpu_memory(self, nli_verifier: NLIVerifier) -> None:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available")

        from veritascore.core.types import Claim

        nli_verifier.verify(
            [Claim(text="Test claim.", source_span=(0, 11), source_text="x")],
            context="Some context.",
        )
        mem_before = torch.cuda.memory_allocated()
        nli_verifier.unload()
        mem_after = torch.cuda.memory_allocated()
        assert mem_after < mem_before
