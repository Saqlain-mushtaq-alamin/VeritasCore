"""Integration tests for the ungrounded verification pipeline (Phase 3).

End-to-end: RuleDecomposer -> RetrievalVerifier, using a REAL NLI model and
(if API keys are configured) REAL web search. Marked `integration` and
`slow`; excluded from `make test` by default.

Tests requiring live API keys are additionally skipped automatically when
the relevant key is not configured (see skip conditions per test).

Run explicitly with:
    pytest tests/integration/test_ungrounded_pipeline.py -v -m integration
"""

from __future__ import annotations

import os

import pytest

from veritascore.core.config import EngineConfig, SearchConfig
from veritascore.core.types import Verdict
from veritascore.decomposer.rule_decomposer import RuleDecomposer
from veritascore.retriever.base import SearchResult
from veritascore.retriever.offline_retriever import OfflineRetriever
from veritascore.verifier.nli_verifier import NLIVerifier
from veritascore.verifier.retrieval_verifier import RetrievalVerifier

pytestmark = [pytest.mark.integration, pytest.mark.slow]

HAS_TAVILY_KEY = bool(os.environ.get("TAVILY_API_KEY"))
HAS_BRAVE_KEY = bool(os.environ.get("BRAVE_API_KEY"))


@pytest.fixture(scope="module")
def real_nli_verifier() -> NLIVerifier:
    """Module-scoped — load the real NLI model once, reuse across tests."""
    verifier = NLIVerifier()
    yield verifier
    verifier.unload()


class TestRetrievalVerifierWithRealNLIOfflineRetriever:
    """Exercises the real NLI model with a controlled (non-network)
    OfflineRetriever pre-seeded with known evidence — validates the NLI
    integration without depending on live search API availability."""

    def test_offline_retriever_with_seeded_cache(
        self, real_nli_verifier: NLIVerifier, tmp_path: object
    ) -> None:
        from veritascore.core.types import Claim

        config = EngineConfig(search=SearchConfig(cache_dir=str(tmp_path)))
        retriever = OfflineRetriever(config=config)

        # The claim text is used verbatim as the search query by
        # RetrievalVerifier._formulate_query (no prefix stripping applies
        # here), so the cache key must match the claim text exactly.
        claim_text = "Eiffel Tower height is 330 metres."
        retriever.cache.set(
            claim_text,
            [SearchResult(
                title="Eiffel Tower",
                url="https://en.wikipedia.org/wiki/Eiffel_Tower",
                snippet="The Eiffel Tower is 330 metres tall, located in Paris.",
                relevance_score=0.9,
            )],
        )

        verifier = RetrievalVerifier(
            config=config,
            retriever=retriever,
            nli_verifier=real_nli_verifier,
        )

        claim = Claim(
            text=claim_text,
            source_span=(0, len(claim_text)), source_text="x",
        )
        verdicts = verifier.verify([claim])
        assert verdicts[0].verdict == Verdict.SUPPORTED

    def test_no_cached_evidence_yields_unsupported(
        self, real_nli_verifier: NLIVerifier, tmp_path: object
    ) -> None:
        from veritascore.core.types import Claim

        config = EngineConfig(search=SearchConfig(cache_dir=str(tmp_path)))
        retriever = OfflineRetriever(config=config)
        verifier = RetrievalVerifier(
            config=config, retriever=retriever, nli_verifier=real_nli_verifier,
        )
        claim = Claim(text="Some obscure unverifiable claim.", source_span=(0, 30), source_text="x")
        verdicts = verifier.verify([claim])
        assert verdicts[0].verdict == Verdict.UNSUPPORTED


class TestEndToEndUngroundedPipeline:
    """Full pipeline: response text -> decompose -> retrieve+verify, all offline."""

    def test_decompose_then_retrieve_verify(
        self, real_nli_verifier: NLIVerifier, tmp_path: object
    ) -> None:
        config = EngineConfig(search=SearchConfig(cache_dir=str(tmp_path)))
        retriever = OfflineRetriever(config=config)
        retriever.cache.set(
            "The Eiffel Tower stands 330 meters tall.",
            [SearchResult(
                title="Eiffel Tower facts",
                url="https://example.com/eiffel",
                snippet="The tower stands 330 metres tall in Paris, France.",
                relevance_score=0.9,
            )],
        )

        decomposer = RuleDecomposer()
        verifier = RetrievalVerifier(
            config=config, retriever=retriever, nli_verifier=real_nli_verifier,
        )

        response = "The Eiffel Tower stands 330 meters tall."
        claims = decomposer.decompose(response)
        assert len(claims) >= 1

        verdicts = verifier.verify(claims)
        assert len(verdicts) == len(claims)
        assert all(v.verification_mode.value == "ungrounded" for v in verdicts)


@pytest.mark.skipif(not HAS_TAVILY_KEY, reason="TAVILY_API_KEY not configured")
class TestLiveTavilySearch:
    """Only runs when a real Tavily API key is present in the environment."""

    def test_live_search_and_verify(self, real_nli_verifier: NLIVerifier) -> None:
        from veritascore.core.types import Claim

        verifier = RetrievalVerifier(nli_verifier=real_nli_verifier)
        claim = Claim(
            text="The Eiffel Tower is located in Paris, France.",
            source_span=(0, 46), source_text="x",
        )
        verdicts = verifier.verify([claim])
        assert verdicts[0].verdict in (Verdict.SUPPORTED, Verdict.UNSUPPORTED)
        assert verdicts[0].evidence is not None


@pytest.mark.skipif(not HAS_BRAVE_KEY, reason="BRAVE_API_KEY not configured")
class TestLiveBraveSearch:
    """Only runs when a real Brave API key is present in the environment."""

    def test_live_search_and_verify(self, real_nli_verifier: NLIVerifier) -> None:
        from veritascore.core.config import EngineConfig, SearchConfig
        from veritascore.core.types import Claim

        config = EngineConfig(search=SearchConfig(provider="brave"))
        verifier = RetrievalVerifier(config=config, nli_verifier=real_nli_verifier)
        claim = Claim(
            text="The Eiffel Tower is located in Paris, France.",
            source_span=(0, 46), source_text="x",
        )
        verdicts = verifier.verify([claim])
        assert verdicts[0].verdict in (Verdict.SUPPORTED, Verdict.UNSUPPORTED)
