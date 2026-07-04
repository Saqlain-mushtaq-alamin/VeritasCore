"""Offline retriever — returns only pre-cached results, never calls the network.

Used as the final fallback when no search API key is configured (e.g.
fully offline deployments, CI environments, or VerificationMode.OFFLINE).
Always "available" in the is_available() sense, since it never fails due
to missing credentials — it simply returns empty results on a cache miss.
"""

from __future__ import annotations

import logging

from veritascore.core.config import EngineConfig
from veritascore.retriever.base import BaseRetriever, SearchResult
from veritascore.retriever.cache import SearchCache

logger = logging.getLogger(__name__)


class OfflineRetriever(BaseRetriever):
    """Retriever that only returns cached results. No network calls ever.

    Args:
        config: EngineConfig instance. Defaults to EngineConfig.default().

    Example:
        >>> retriever = OfflineRetriever()
        >>> results = await retriever.search("some query")  # [] if not cached
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig.default()
        self.cache = SearchCache(
            cache_dir=self.config.search.cache_dir,
            enabled=True,
        )

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Return cached results for `query`, or an empty list on a miss.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            Cached SearchResult list (possibly empty). Never raises.
        """
        cached = self.cache.get(query)
        if cached is not None:
            return cached[:max_results]
        logger.warning("Offline mode: no cached results for query: %s", query[:50])
        return []

    def is_available(self) -> bool:
        """Always True — offline mode never fails to be 'available',
        it just may return empty results."""
        return True
