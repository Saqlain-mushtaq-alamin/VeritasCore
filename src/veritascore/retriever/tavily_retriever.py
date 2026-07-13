"""Tavily Search API retriever.

Tavily's free tier allows 1000 requests/month. Caching (retriever/cache.py)
and rate limiting (retriever/rate_limit.py) are both wired in by default to
avoid exhausting this quota during repeated benchmark runs.
"""

from __future__ import annotations

import logging

import httpx

from veritascore.core.config import EngineConfig
from veritascore.core.exceptions import RetrievalError
from veritascore.retriever.base import BaseRetriever, SearchResult
from veritascore.retriever.cache import SearchCache
from veritascore.retriever.rate_limit import AsyncRateLimiter, retry_with_backoff

logger = logging.getLogger(__name__)

TAVILY_API_URL = "https://api.tavily.com/search"

# Tavily free tier: 1000 req/month ≈ generous per-second budget, but we
# self-impose a conservative cap to be a good API citizen and to smooth
# bursts during concurrent claim verification.
_DEFAULT_MAX_CALLS_PER_SECOND = 5


class TavilyRetriever(BaseRetriever):
    """Retrieve evidence using the Tavily Search API.

    Args:
        config: EngineConfig instance. Defaults to EngineConfig.default().
        cache: SearchCache instance. Defaults to one built from config.
        rate_limiter: AsyncRateLimiter instance. Defaults to a conservative
            5 calls/second limiter.
        max_retries: Number of attempts (including the first) on transient
            HTTP/network failures.

    Example:
        >>> retriever = TavilyRetriever()
        >>> results = await retriever.search("Eiffel Tower height", max_results=5)
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        cache: SearchCache | None = None,
        rate_limiter: AsyncRateLimiter | None = None,
        max_retries: int = 3,
    ) -> None:
        self.config = config or EngineConfig.default()
        self.api_key = self.config.search.tavily_api_key
        self.timeout = self.config.search.timeout_seconds
        self.max_retries = max_retries
        self.cache = cache or SearchCache(
            cache_dir=self.config.search.cache_dir,
            enabled=self.config.search.cache_enabled,
        )
        self.rate_limiter = rate_limiter or AsyncRateLimiter(
            max_calls=_DEFAULT_MAX_CALLS_PER_SECOND, period_seconds=1.0
        )

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search Tavily for evidence relevant to `query`.

        Checks the cache first; on a miss, calls the Tavily API (rate
        limited and retried on transient failures) and caches the result.

        Args:
            query: Search query string.
            max_results: Maximum number of results to request/return.

        Returns:
            List of SearchResult, possibly empty.

        Raises:
            RetrievalError: If no API key is configured, or all retry
                attempts fail.
        """
        cached = self.cache.get(query)
        if cached is not None:
            logger.debug("Cache hit for query: %s", query[:50])
            return cached[:max_results]

        if not self.api_key:
            raise RetrievalError("TAVILY_API_KEY not configured")

        async def _do_request() -> httpx.Response:
            await self.rate_limiter.acquire()
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    TAVILY_API_URL,
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max_results,
                        "include_raw_content": False,
                        "search_depth": "basic",
                    },
                )
                response.raise_for_status()
                return response

        try:
            response = await retry_with_backoff(
                _do_request,
                max_attempts=self.max_retries,
                retryable_exceptions=(httpx.TransportError, httpx.HTTPStatusError),
            )
            data = response.json()
        except httpx.HTTPStatusError as e:
            raise RetrievalError(f"Tavily API error: {e.response.status_code}") from e
        except Exception as e:
            raise RetrievalError(f"Tavily search failed: {e}") from e

        results = [
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                relevance_score=item.get("score", 0.5),
            )
            for item in data.get("results", [])
        ]

        self.cache.set(query, results)
        logger.info("Tavily search returned %d results for: %s", len(results), query[:50])
        return results

    def is_available(self) -> bool:
        """Return True if a Tavily API key is configured."""
        return bool(self.api_key)
