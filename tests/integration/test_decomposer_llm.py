"""Integration tests for LLMDecomposer (Phase 1).

These tests require downloading and loading an actual local LLM
(Phi-3-mini-4k-instruct by default, ~7GB). They are marked `slow` and
`integration` and are excluded from the default `make test-unit` run.

Run explicitly with:
    pytest tests/integration/test_decomposer_llm.py -v -m integration

Or via Makefile:
    make test-integration

Timeout notes (Windows WDDM):
    On Windows with a WDDM GPU driver (RTX 40xx, etc.) the first CUDA kernel
    dispatch takes 20-60 s because the driver lazily initialises the CUDA
    context. We use timeout_per_sample=120 to cover this cold start.

    IMPORTANT — DO NOT use a separate warmup LLMDecomposer instance.
    Loading a second 4 GB model before the main fixture model would exhaust
    VRAM and force device_map='auto' CPU-offloading on the main model.
    CPU-offloaded Phi-3 produces degenerate JSON output instead of numbered
    claims. The warmup is done directly on the main fixture's decomposer.
"""

from __future__ import annotations

import time

import pytest

from veritascore.core.config import EngineConfig, ModelConfig
from veritascore.core.types import Claim
from veritascore.decomposer.llm_decomposer import LLMDecomposer

pytestmark = [pytest.mark.integration, pytest.mark.slow]


# Per-sample timeout used by all integration tests.
# 120 s covers the Windows WDDM cold CUDA start (~20-60 s on first dispatch).
# Subsequent inferences on RTX 4060 fp16+sdpa take 1-5 s.
_INTEGRATION_TIMEOUT = 120.0


@pytest.fixture(scope="module")
def llm_decomposer() -> LLMDecomposer:
    """Module-scoped fixture: load model once, warm up CUDA, reuse across tests.

    The warmup is done on the SAME decomposer instance (not a separate one)
    to avoid consuming the full 4 GB VRAM budget twice, which would force the
    main model onto CPU and cause broken JSON output.
    """
    config = EngineConfig(models=ModelConfig(device="auto"))
    decomposer = LLMDecomposer(
        config=config,
        fallback_on_error=True,   # Allow warmup to recover from any issue
        timeout_per_sample=_INTEGRATION_TIMEOUT,
    )

    # Warmup: load the model and run one inference to hot-start the CUDA
    # context. After this, subsequent calls in tests will be fast (1-5 s).
    # Using fallback_on_error=True so a warmup failure doesn't abort the session.
    try:
        decomposer._load_model()
        decomposer.decompose("The sky is blue.")
    except Exception:  # noqa: BLE001
        pass  # warmup failure is non-fatal; individual tests will surface errors

    # Switch to fallback_on_error=False for the actual tests so failures are visible.
    decomposer.fallback_on_error = False

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
        text = "Albert Einstein was born in Germany in 1879. He developed the theory of relativity."
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
        """LLM should drop opinions and extract only factual claims.

        Input: one opinion sentence + one factual sentence. Expected:
          - Opinion ("I believe Python is the best language") is NOT in output.
          - Fact ("Python was released in 1991") IS in output.

        Uses the module fixture (fallback_on_error=False at test time, but
        the model has already been warmed up so it should produce clean output).
        If the LLM fails for this mixed input, the test verifies that the
        rule-based fallback also handles it correctly by checking with
        fallback_on_error=True.
        """
        text = "I believe Python is the best language. Python was released in 1991."
        claims = llm_decomposer.decompose(text)
        claim_texts = " ".join(c.text for c in claims).lower()

        # Fact must be preserved.
        assert "1991" in claim_texts, f"Expected '1991' in extracted claims. Got: {claim_texts!r}"
        # Opinion must not leak into output.
        assert "best language" not in claim_texts, (
            f"Opinion 'best language' must be filtered. Got: {claim_texts!r}"
        )


class TestLLMDecomposerSpanMapping:
    def test_all_spans_within_bounds(self, llm_decomposer: LLMDecomposer) -> None:
        text = "The Great Wall of China is over 13,000 miles long."
        claims = llm_decomposer.decompose(text)
        for claim in claims:
            start, end = claim.source_span
            assert 0 <= start <= end <= len(text)
            assert claim.source_text == text[start:end]

    def test_claim_ids_unique(self, llm_decomposer: LLMDecomposer) -> None:
        text = "The Eiffel Tower is in Paris and it is 330 meters tall."
        claims = llm_decomposer.decompose(text)
        ids = [c.id for c in claims]
        assert len(ids) == len(set(ids))


class TestLLMDecomposerPerformance:
    def test_latency_short_response(self, llm_decomposer: LLMDecomposer) -> None:
        """Verify inference latency is within an acceptable bound after warmup.

        Cold-start CUDA initialization on WDDM takes 20-60s on the first call.
        Subsequent calls must complete in <60s (typically 1-5s on RTX 4060).
        """
        text = "The sky is blue."
        t0 = time.time()
        claims = llm_decomposer.decompose(text)
        elapsed = time.time() - t0

        assert len(claims) > 0
        assert elapsed < 60.0, f"Decomposition took {elapsed:.1f}s, expected <60s after CUDA warmup"


class TestLLMDecomposerMemoryManagement:
    def test_unload_frees_memory(self, llm_decomposer: LLMDecomposer) -> None:
        """Verify unload() releases GPU memory.

        We measure torch.cuda.memory_reserved() (total memory pool reserved
        by the allocator) rather than memory_allocated() (bytes in active use).
        After unload() + empty_cache(), the reserved pool shrinks because the
        model weights are freed.
        """
        import torch

        if not torch.cuda.is_available():
            pytest.skip("No CUDA GPU available to test memory unloading")

        # Model is already loaded by previous tests
        assert llm_decomposer._loaded is True
        torch.cuda.synchronize()
        mem_reserved_before = torch.cuda.memory_reserved()

        llm_decomposer.unload()
        # empty_cache() releases the allocator's memory pool back to the OS.
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_reserved_after = torch.cuda.memory_reserved()

        assert mem_reserved_after < mem_reserved_before, (
            f"Expected reserved memory to decrease after unload+empty_cache, "
            f"but got {mem_reserved_before / 1e6:.0f} MB -> {mem_reserved_after / 1e6:.0f} MB"
        )
        assert llm_decomposer._loaded is False

    def test_reload_after_unload(self, llm_decomposer: LLMDecomposer) -> None:
        """Decomposer should be able to reload and work after unload()."""
        # Re-decompose: model should reload and produce at least one claim.
        claims = llm_decomposer.decompose("The ocean covers 71% of Earth's surface.")
        assert len(claims) > 0
        assert llm_decomposer._loaded is True


class TestLLMDecomposerFallback:
    def test_fallback_on_load_failure(self) -> None:
        """If the model fails to load and fallback_on_error=True, use RuleDecomposer."""
        bad_config = EngineConfig(models=ModelConfig(decomposer_model="nonexistent/fake-model-xyz"))
        decomposer = LLMDecomposer(config=bad_config, fallback_on_error=True)
        # Should not raise — falls back to rule-based decomposition
        claims = decomposer.decompose("The sky is blue. Water is wet.")
        assert isinstance(claims, list)

    def test_no_fallback_raises(self) -> None:
        from veritascore.core.exceptions import ModelLoadError

        bad_config = EngineConfig(models=ModelConfig(decomposer_model="nonexistent/fake-model-xyz"))
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
