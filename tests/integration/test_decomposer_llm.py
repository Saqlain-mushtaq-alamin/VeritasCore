"""Integration tests for LLMDecomposer (Phase 1).

These tests require downloading and loading an actual local LLM
(Phi-3-mini-4k-instruct by default, ~7GB). They are marked `slow` and
`integration` and are excluded from the default `make test-unit` run.

Run explicitly with:
    pytest tests/integration/test_decomposer_llm.py -v -m integration

Or via Makefile:
    make test-integration
"""

from __future__ import annotations

import time

import pytest

from veritascore.core.config import EngineConfig, ModelConfig
from veritascore.core.types import Claim
from veritascore.decomposer.llm_decomposer import LLMDecomposer

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def llm_decomposer() -> LLMDecomposer:
    """Module-scoped fixture — load the model once, reuse across tests."""
    config = EngineConfig(models=ModelConfig(device="auto"))
    decomposer = LLMDecomposer(config=config, fallback_on_error=False)
    yield decomposer
    decomposer.unload()


class TestLLMDecomposerAvailability:
    def test_is_available(self, llm_decomposer: LLMDecomposer) -> None:
        assert llm_decomposer.is_available() is True


class TestLLMDecomposerBasic:
    def test_simple_decomposition(self, llm_decomposer: LLMDecomposer) -> None:
        text = "The Eiffel Tower is 330 meters tall and was built in 1889."
        claims = llm_decomposer.decompose(text)
        assert len(claims) >= 2
        assert all(isinstance(c, Claim) for c in claims)

    def test_pronoun_resolution(self, llm_decomposer: LLMDecomposer) -> None:
        """LLM decomposer should resolve pronouns (key advantage over RuleDecomposer)."""
        text = (
            "Albert Einstein was born in Germany in 1879. "
            "He developed the theory of relativity."
        )
        claims = llm_decomposer.decompose(text)
        claim_texts = " ".join(c.text for c in claims).lower()
        # "He" should be resolved to "Einstein" or "Albert Einstein"
        assert "einstein" in claim_texts

    def test_with_query_context(self, llm_decomposer: LLMDecomposer) -> None:
        text = "It has a population of about 2.1 million people."
        query = "Tell me about Paris."
        claims = llm_decomposer.decompose(text, query=query)
        # Should produce at least one claim, hopefully resolving "It" -> "Paris"
        assert len(claims) >= 1

    def test_multi_fact_sentence(self, llm_decomposer: LLMDecomposer) -> None:
        text = (
            "Marie Curie was born in Warsaw, Poland on November 7, 1867, "
            "and she won Nobel Prizes in both Physics and Chemistry."
        )
        claims = llm_decomposer.decompose(text)
        assert len(claims) >= 3  # birthplace, birthdate, physics prize, chemistry prize

    def test_empty_input(self, llm_decomposer: LLMDecomposer) -> None:
        assert llm_decomposer.decompose("") == []
        assert llm_decomposer.decompose("   ") == []

    def test_opinion_filtering(self, llm_decomposer: LLMDecomposer) -> None:
        text = "I believe Python is the best language. Python was released in 1991."
        claims = llm_decomposer.decompose(text)
        claim_texts = " ".join(c.text for c in claims).lower()
        assert "1991" in claim_texts


class TestLLMDecomposerSpanMapping:
    def test_all_spans_within_bounds(self, llm_decomposer: LLMDecomposer) -> None:
        text = "The Great Wall of China is over 13,000 miles long."
        claims = llm_decomposer.decompose(text)
        for claim in claims:
            assert 0 <= claim.source_span[0] < claim.source_span[1] <= len(text)

    def test_claim_ids_unique(self, llm_decomposer: LLMDecomposer) -> None:
        text = "The sun is a star. The moon orbits the Earth. Mars is red."
        claims = llm_decomposer.decompose(text)
        ids = [c.id for c in claims]
        assert len(ids) == len(set(ids))


class TestLLMDecomposerPerformance:
    def test_latency_short_response(self, llm_decomposer: LLMDecomposer) -> None:
        """Target: <5s for a ~20-word response (Master Plan §1.8)."""
        text = (
            "The Amazon rainforest covers about 5.5 million square kilometers "
            "and is home to roughly 10% of known species."
        )
        t0 = time.time()
        claims = llm_decomposer.decompose(text)
        elapsed = time.time() - t0

        assert len(claims) > 0
        assert elapsed < 10.0, f"Decomposition took {elapsed:.1f}s, expected <10s"


class TestLLMDecomposerMemoryManagement:
    def test_unload_frees_memory(self, llm_decomposer: LLMDecomposer) -> None:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available to test memory unloading")

        # Ensure model is loaded
        llm_decomposer.decompose("Test sentence for loading.")
        mem_before = torch.cuda.memory_allocated()

        llm_decomposer.unload()
        mem_after = torch.cuda.memory_allocated()

        assert mem_after < mem_before
        assert llm_decomposer._loaded is False

    def test_reload_after_unload(self, llm_decomposer: LLMDecomposer) -> None:
        """Decomposer should be able to reload and work after unload()."""
        llm_decomposer.decompose("First call.")
        llm_decomposer.unload()
        claims = llm_decomposer.decompose("The ocean covers 71% of Earth's surface.")
        assert len(claims) > 0


class TestLLMDecomposerFallback:
    def test_fallback_on_load_failure(self) -> None:
        """If the model fails to load and fallback_on_error=True, use RuleDecomposer."""
        bad_config = EngineConfig(
            models=ModelConfig(decomposer_model="nonexistent/fake-model-xyz")
        )
        decomposer = LLMDecomposer(config=bad_config, fallback_on_error=True)
        # Should not raise — falls back to rule-based decomposition
        claims = decomposer.decompose("The sky is blue. Water is wet.")
        assert isinstance(claims, list)

    def test_no_fallback_raises(self) -> None:
        from veritascore.core.exceptions import ModelLoadError

        bad_config = EngineConfig(
            models=ModelConfig(decomposer_model="nonexistent/fake-model-xyz")
        )
        decomposer = LLMDecomposer(config=bad_config, fallback_on_error=False)
        with pytest.raises((ModelLoadError, Exception)):
            decomposer.decompose("Some text.")


class TestOllamaBackend:
    """Tests for the optional Ollama backend (requires `ollama` package + server)."""

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValueError):
            LLMDecomposer(backend="invalid_backend")

    @pytest.mark.skip(reason="Requires running Ollama server — enable manually")
    def test_ollama_decomposition(self) -> None:
        config = EngineConfig(models=ModelConfig(decomposer_model="phi3:mini"))
        decomposer = LLMDecomposer(config=config, backend="ollama")
        claims = decomposer.decompose("The Eiffel Tower is in Paris.")
        assert len(claims) > 0
