"""Unit tests for the retriever module (Phase 3): cache, rate limiting,
retry logic, and each retriever implementation with mocked HTTP calls.

No real network calls are made — httpx.AsyncClient is monkeypatched.
"""
# ruff: noqa: E501

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from veritascore.core.config import EngineConfig, SearchConfig
from veritascore.core.exceptions import RetrievalError
from veritascore.retriever.base import BaseRetriever, SearchResult
from veritascore.retriever.brave_retriever import BraveRetriever
from veritascore.retriever.cache import SearchCache
from veritascore.retriever.offline_retriever import OfflineRetriever
from veritascore.retriever.rate_limit import AsyncRateLimiter, retry_with_backoff
from veritascore.retriever.tavily_retriever import TavilyRetriever

# ── BaseRetriever Contract ───────────────────────────────────────────────────

class TestBaseRetriever:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseRetriever()  # type: ignore[abstract]

    def test_subclass_must_implement_search(self) -> None:
        class Incomplete(BaseRetriever):
            def is_available(self) -> bool:
                return True

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_must_implement_is_available(self) -> None:
        class Incomplete(BaseRetriever):
            async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
                return []

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]


class TestSearchResult:
    def test_defaults(self) -> None:
        r = SearchResult(title="T", url="http://x.com", snippet="snip")
        assert r.content is None
        assert r.relevance_score == 0.5

    def test_relevance_bounds(self) -> None:
        with pytest.raises(ValueError):
            SearchResult(title="T", url="x", snippet="s", relevance_score=1.5)


# ── SearchCache ───────────────────────────────────────────────────────────────

class TestSearchCache:
    @pytest.fixture
    def cache(self, tmp_path: Path) -> SearchCache:
        return SearchCache(cache_dir=str(tmp_path / "cache"), enabled=True)

    @pytest.fixture
    def sample_results(self) -> list[SearchResult]:
        return [
            SearchResult(title="A", url="http://a.com", snippet="snippet a"),
            SearchResult(title="B", url="http://b.com", snippet="snippet b"),
        ]

    def test_cache_miss_returns_none(self, cache: SearchCache) -> None:
        assert cache.get("nonexistent query") is None

    def test_set_then_get(self, cache: SearchCache, sample_results: list[SearchResult]) -> None:
        cache.set("eiffel tower height", sample_results)
        cached = cache.get("eiffel tower height")
        assert cached is not None
        assert len(cached) == 2
        assert cached[0].title == "A"

    def test_case_insensitive_key(self, cache: SearchCache, sample_results: list[SearchResult]) -> None:
        cache.set("Eiffel Tower", sample_results)
        assert cache.get("eiffel tower") is not None
        assert cache.get("  EIFFEL TOWER  ") is not None

    def test_disabled_cache_always_misses(self, tmp_path: Path, sample_results: list[SearchResult]) -> None:
        cache = SearchCache(cache_dir=str(tmp_path / "disabled"), enabled=False)
        cache.set("query", sample_results)
        assert cache.get("query") is None

    def test_clear(self, cache: SearchCache, sample_results: list[SearchResult]) -> None:
        cache.set("q1", sample_results)
        cache.set("q2", sample_results)
        count = cache.clear()
        assert count == 2
        assert cache.get("q1") is None

    def test_len(self, cache: SearchCache, sample_results: list[SearchResult]) -> None:
        assert len(cache) == 0
        cache.set("q1", sample_results)
        assert len(cache) == 1

    def test_corrupted_cache_file_treated_as_miss(self, cache: SearchCache) -> None:
        key = cache._key("broken query")
        path = cache.cache_dir / f"{key}.json"
        path.write_text("{not valid json")
        assert cache.get("broken query") is None

    def test_overwrite_existing_entry(self, cache: SearchCache, sample_results: list[SearchResult]) -> None:
        cache.set("q", sample_results)
        new_results = [SearchResult(title="C", url="http://c.com", snippet="c")]
        cache.set("q", new_results)
        cached = cache.get("q")
        assert cached is not None
        assert len(cached) == 1
        assert cached[0].title == "C"


# ── AsyncRateLimiter ──────────────────────────────────────────────────────────

class TestAsyncRateLimiter:
    def test_invalid_max_calls_raises(self) -> None:
        with pytest.raises(ValueError):
            AsyncRateLimiter(max_calls=0)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            AsyncRateLimiter(max_calls=5, period_seconds=0)

    @pytest.mark.asyncio
    async def test_allows_calls_under_limit(self) -> None:
        limiter = AsyncRateLimiter(max_calls=5, period_seconds=1.0)
        t0 = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5  # Should not have needed to sleep

    @pytest.mark.asyncio
    async def test_blocks_when_over_limit(self) -> None:
        limiter = AsyncRateLimiter(max_calls=2, period_seconds=0.3)
        t0 = time.monotonic()
        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()  # Should block until window clears
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.25  # Allow small scheduling slack


# ── retry_with_backoff ────────────────────────────────────────────────────────

class TestRetryWithBackoff:
    @pytest.mark.asyncio
    async def test_succeeds_first_try(self) -> None:
        calls = 0

        async def fn() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        result = await retry_with_backoff(fn, max_attempts=3)
        assert result == "ok"
        assert calls == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self) -> None:
        calls = 0

        async def fn() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ValueError("transient")
            return "ok"

        result = await retry_with_backoff(
            fn, max_attempts=5, base_delay=0.01, retryable_exceptions=(ValueError,)
        )
        assert result == "ok"
        assert calls == 3

    @pytest.mark.asyncio
    async def test_exhausts_attempts_and_raises(self) -> None:
        calls = 0

        async def fn() -> str:
            nonlocal calls
            calls += 1
            raise ValueError("always fails")

        with pytest.raises(ValueError):
            await retry_with_backoff(
                fn, max_attempts=3, base_delay=0.01, retryable_exceptions=(ValueError,)
            )
        assert calls == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception_propagates_immediately(self) -> None:
        calls = 0

        async def fn() -> str:
            nonlocal calls
            calls += 1
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            await retry_with_backoff(
                fn, max_attempts=3, base_delay=0.01, retryable_exceptions=(ValueError,)
            )
        assert calls == 1

    @pytest.mark.asyncio
    async def test_invalid_max_attempts_raises(self) -> None:
        async def fn() -> str:
            return "x"

        with pytest.raises(ValueError):
            await retry_with_backoff(fn, max_attempts=0)


# ── OfflineRetriever ──────────────────────────────────────────────────────────

class TestOfflineRetriever:
    @pytest.mark.asyncio
    async def test_returns_empty_on_miss(self, tmp_path: Path) -> None:
        config = EngineConfig(search=SearchConfig(cache_dir=str(tmp_path / "cache")))
        retriever = OfflineRetriever(config=config)
        results = await retriever.search("nonexistent query")
        assert results == []

    @pytest.mark.asyncio
    async def test_returns_cached_results(self, tmp_path: Path) -> None:
        config = EngineConfig(search=SearchConfig(cache_dir=str(tmp_path / "cache")))
        retriever = OfflineRetriever(config=config)
        retriever.cache.set("known query", [
            SearchResult(title="T", url="http://x.com", snippet="snip"),
        ])
        results = await retriever.search("known query")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_respects_max_results(self, tmp_path: Path) -> None:
        config = EngineConfig(search=SearchConfig(cache_dir=str(tmp_path / "cache")))
        retriever = OfflineRetriever(config=config)
        retriever.cache.set("q", [
            SearchResult(title=f"T{i}", url=f"http://x{i}.com", snippet="s") for i in range(5)
        ])
        results = await retriever.search("q", max_results=2)
        assert len(results) == 2

    def test_always_available(self) -> None:
        assert OfflineRetriever().is_available() is True


# ── TavilyRetriever (mocked HTTP) ────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, json_data: dict[str, Any], status_code: int = 200) -> None:
        self._json = json_data
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)


class TestTavilyRetriever:
    @pytest.fixture
    def config_with_key(self, tmp_path: Path) -> EngineConfig:
        return EngineConfig(
            search=SearchConfig(
                tavily_api_key="fake-key",
                cache_dir=str(tmp_path / "cache"),
                cache_enabled=True,
            )
        )

    def test_not_available_without_key(self, tmp_path: Path) -> None:
        config = EngineConfig(search=SearchConfig(cache_dir=str(tmp_path / "cache")))
        retriever = TavilyRetriever(config=config)
        assert retriever.is_available() is False

    def test_available_with_key(self, config_with_key: EngineConfig) -> None:
        retriever = TavilyRetriever(config=config_with_key)
        assert retriever.is_available() is True

    @pytest.mark.asyncio
    async def test_missing_key_raises_on_search(self, tmp_path: Path) -> None:
        config = EngineConfig(search=SearchConfig(cache_dir=str(tmp_path / "cache")))
        retriever = TavilyRetriever(config=config)
        with pytest.raises(RetrievalError):
            await retriever.search("test query")

    @pytest.mark.asyncio
    async def test_successful_search(
        self, config_with_key: EngineConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retriever = TavilyRetriever(config=config_with_key)

        async def fake_post(self: Any, url: str, **kwargs: Any) -> _FakeResponse:
            return _FakeResponse({
                "results": [
                    {"title": "Eiffel Tower", "url": "http://wiki.org/eiffel", "content": "330m tall", "score": 0.9},
                ]
            })

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        results = await retriever.search("Eiffel Tower height")
        assert len(results) == 1
        assert results[0].title == "Eiffel Tower"
        assert results[0].relevance_score == 0.9

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_http_call(
        self, config_with_key: EngineConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retriever = TavilyRetriever(config=config_with_key)
        retriever.cache.set("cached query", [
            SearchResult(title="Cached", url="http://x.com", snippet="s"),
        ])

        call_count = 0

        async def fake_post(self: Any, url: str, **kwargs: Any) -> _FakeResponse:
            nonlocal call_count
            call_count += 1
            return _FakeResponse({"results": []})

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        results = await retriever.search("cached query")
        assert call_count == 0
        assert results[0].title == "Cached"

    @pytest.mark.asyncio
    async def test_second_identical_search_uses_cache(
        self, config_with_key: EngineConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression test for G3 criterion 2: caching prevents duplicate API calls."""
        retriever = TavilyRetriever(config=config_with_key)
        call_count = 0

        async def fake_post(self: Any, url: str, **kwargs: Any) -> _FakeResponse:
            nonlocal call_count
            call_count += 1
            return _FakeResponse({
                "results": [{"title": "X", "url": "http://x.com", "content": "c", "score": 0.5}]
            })

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        await retriever.search("repeated query")
        await retriever.search("repeated query")
        await retriever.search("repeated query")
        assert call_count == 1  # Only the first call hits the network

    @pytest.mark.asyncio
    async def test_http_error_raises_retrieval_error(
        self, config_with_key: EngineConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retriever = TavilyRetriever(config=config_with_key, max_retries=1)

        async def fake_post(self: Any, url: str, **kwargs: Any) -> _FakeResponse:
            return _FakeResponse({}, status_code=500)

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        with pytest.raises(RetrievalError):
            await retriever.search("failing query")

    @pytest.mark.asyncio
    async def test_retries_on_transient_failure(
        self, config_with_key: EngineConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retriever = TavilyRetriever(config=config_with_key, max_retries=3)
        retriever.rate_limiter = AsyncRateLimiter(max_calls=100, period_seconds=1.0)
        attempt = 0

        async def fake_post(self: Any, url: str, **kwargs: Any) -> _FakeResponse:
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                return _FakeResponse({}, status_code=503)
            return _FakeResponse({"results": [{"title": "OK", "url": "http://x.com", "content": "c"}]})

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)
        results = await retriever.search("flaky query")
        assert attempt == 3
        assert len(results) == 1


# ── BraveRetriever (mocked HTTP) ─────────────────────────────────────────────

class TestBraveRetriever:
    @pytest.fixture
    def config_with_key(self, tmp_path: Path) -> EngineConfig:
        return EngineConfig(
            search=SearchConfig(
                brave_api_key="fake-brave-key",
                cache_dir=str(tmp_path / "cache"),
                cache_enabled=True,
            )
        )

    def test_not_available_without_key(self, tmp_path: Path) -> None:
        config = EngineConfig(search=SearchConfig(cache_dir=str(tmp_path / "cache")))
        retriever = BraveRetriever(config=config)
        assert retriever.is_available() is False

    @pytest.mark.asyncio
    async def test_missing_key_raises(self, tmp_path: Path) -> None:
        config = EngineConfig(search=SearchConfig(cache_dir=str(tmp_path / "cache")))
        retriever = BraveRetriever(config=config)
        with pytest.raises(RetrievalError):
            await retriever.search("test")

    @pytest.mark.asyncio
    async def test_successful_search(
        self, config_with_key: EngineConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retriever = BraveRetriever(config=config_with_key)

        async def fake_get(self: Any, url: str, **kwargs: Any) -> _FakeResponse:
            return _FakeResponse({
                "web": {"results": [
                    {"title": "Brave Result", "url": "http://brave.com/x", "description": "desc"},
                ]}
            })

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        results = await retriever.search("test query")
        assert len(results) == 1
        assert results[0].title == "Brave Result"
        assert results[0].relevance_score == 0.5  # Brave has no score; uses default

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_http_call(
        self, config_with_key: EngineConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        retriever = BraveRetriever(config=config_with_key)
        retriever.cache.set("cached", [SearchResult(title="C", url="http://x.com", snippet="s")])

        call_count = 0

        async def fake_get(self: Any, url: str, **kwargs: Any) -> _FakeResponse:
            nonlocal call_count
            call_count += 1
            return _FakeResponse({"web": {"results": []}})

        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
        await retriever.search("cached")
        assert call_count == 0


# ── Retriever Selection Logic (via RetrievalVerifier) ────────────────────────

class _DummyNLI:
    """Minimal stand-in for NLIVerifier — avoids loading a real model
    just to test retriever selection logic."""

    def _load_model(self) -> None:
        pass

    def is_available(self) -> bool:
        return True


class TestRetrieverSelection:
    def test_tavily_selected_when_key_present(self, tmp_path: Path) -> None:
        from veritascore.verifier.retrieval_verifier import RetrievalVerifier

        config = EngineConfig(
            search=SearchConfig(
                provider="tavily",
                tavily_api_key="key",
                cache_dir=str(tmp_path / "cache"),
            )
        )
        verifier = RetrievalVerifier(config=config, nli_verifier=_DummyNLI())  # type: ignore[arg-type]
        assert isinstance(verifier.retriever, TavilyRetriever)

    def test_brave_fallback_when_tavily_missing(self, tmp_path: Path) -> None:
        from veritascore.verifier.retrieval_verifier import RetrievalVerifier

        config = EngineConfig(
            search=SearchConfig(
                provider="tavily",
                brave_api_key="brave-key",
                cache_dir=str(tmp_path / "cache"),
            )
        )
        verifier = RetrievalVerifier(config=config, nli_verifier=_DummyNLI())  # type: ignore[arg-type]
        assert isinstance(verifier.retriever, BraveRetriever)

    def test_offline_fallback_when_no_keys(self, tmp_path: Path) -> None:
        from veritascore.verifier.retrieval_verifier import RetrievalVerifier

        config = EngineConfig(
            search=SearchConfig(provider="tavily", cache_dir=str(tmp_path / "cache"))
        )
        verifier = RetrievalVerifier(config=config, nli_verifier=_DummyNLI())  # type: ignore[arg-type]
        assert isinstance(verifier.retriever, OfflineRetriever)

    def test_brave_selected_when_provider_is_brave_and_key_present(self, tmp_path: Path) -> None:
        from veritascore.verifier.retrieval_verifier import RetrievalVerifier

        config = EngineConfig(
            search=SearchConfig(
                provider="brave",
                brave_api_key="brave-key",
                cache_dir=str(tmp_path / "cache"),
            )
        )
        verifier = RetrievalVerifier(config=config, nli_verifier=_DummyNLI())  # type: ignore[arg-type]
        assert isinstance(verifier.retriever, BraveRetriever)

    def test_brave_provider_falls_back_to_tavily_when_brave_key_missing(self, tmp_path: Path) -> None:
        from veritascore.verifier.retrieval_verifier import RetrievalVerifier

        config = EngineConfig(
            search=SearchConfig(
                provider="brave",
                tavily_api_key="tavily-key",
                cache_dir=str(tmp_path / "cache"),
            )
        )
        verifier = RetrievalVerifier(config=config, nli_verifier=_DummyNLI())  # type: ignore[arg-type]
        assert isinstance(verifier.retriever, TavilyRetriever)


class TestRetrievalVerifierEventLoopGuard:
    @pytest.mark.asyncio
    async def test_verify_raises_inside_running_loop(self, tmp_path: Path) -> None:
        """Calling the sync verify() from inside an already-running event
        loop must raise a clear error directing the caller to verify_async()."""
        from veritascore.verifier.retrieval_verifier import RetrievalVerifier

        config = EngineConfig(search=SearchConfig(cache_dir=str(tmp_path / "cache")))
        verifier = RetrievalVerifier(
            config=config,
            retriever=OfflineRetriever(config=config),
            nli_verifier=_DummyNLI(),  # type: ignore[arg-type]
        )
        from veritascore.core.types import Claim

        with pytest.raises(RuntimeError, match="verify_async"):
            verifier.verify([Claim(text="A claim.", source_span=(0, 8), source_text="A claim.")])


class TestRetrievalVerifierFailureHandling:
    @pytest.mark.asyncio
    async def test_retriever_exception_yields_unsupported_not_crash(self, tmp_path: Path) -> None:
        """If the retriever raises mid-search, that claim should resolve to
        UNSUPPORTED rather than propagating the exception and aborting the
        whole batch (G3 criterion 5: graceful handling, no crashes)."""
        from veritascore.core.types import Claim
        from veritascore.verifier.retrieval_verifier import RetrievalVerifier

        class _FailingRetriever(BaseRetriever):
            async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
                raise RuntimeError("simulated network failure")

            def is_available(self) -> bool:
                return True

        config = EngineConfig(search=SearchConfig(cache_dir=str(tmp_path / "cache")))
        verifier = RetrievalVerifier(
            config=config,
            retriever=_FailingRetriever(),
            nli_verifier=_DummyNLI(),  # type: ignore[arg-type]
        )
        claim = Claim(text="A claim that cannot be retrieved.", source_span=(0, 33), source_text="x")
        verdicts = await verifier.verify_async([claim])
        assert len(verdicts) == 1
        assert verdicts[0].verdict.value == "unsupported"
