"""Search result caching to minimize API calls.

Caching is mandatory, not optional: free-tier search APIs (Tavily: 1000
req/month, Brave: 2000 req/month) are easily exhausted by a single
benchmark run. A 500-sample benchmark with ~5 claims/sample is 2500
searches — caching turns repeated runs (e.g. iterating on thresholds)
into near-zero additional API usage.

File-based, keyed by a SHA-256 hash of the normalized query. Writes are
atomic (write-to-temp-then-rename) to avoid corrupting the cache if two
processes write concurrently.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

from veritascore.retriever.base import SearchResult

logger = logging.getLogger(__name__)


class SearchCache:
    """File-based cache for search results.

    Args:
        cache_dir: Directory to store cache files. Created on first use.
        enabled: If False, get() always returns None and set() is a no-op
            (useful for tests or explicitly disabling caching).

    Example:
        >>> cache = SearchCache(cache_dir=".cache/search")
        >>> cache.set("Eiffel Tower height", [result1, result2])
        >>> cache.get("Eiffel Tower height")
        [SearchResult(...), SearchResult(...)]
    """

    def __init__(self, cache_dir: str = ".cache/search", enabled: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, query: str) -> str:
        """Compute a stable cache key from a normalized query string."""
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def get(self, query: str) -> list[SearchResult] | None:
        """Retrieve cached results for a query, if present.

        Args:
            query: The search query string.

        Returns:
            List of SearchResult if cached, None on cache miss or read
            failure (corrupted cache file is treated as a miss, not an error).
        """
        if not self.enabled:
            return None

        path = self.cache_dir / f"{self._key(query)}.json"
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [SearchResult(**item) for item in data]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning("Corrupted cache entry for query (treating as miss): %s", e)
            return None

    def set(self, query: str, results: list[SearchResult]) -> None:
        """Store results for a query, overwriting any existing entry.

        Writes atomically (temp file + rename) so concurrent readers never
        see a partially-written cache file.

        Args:
            query: The search query string.
            results: List of SearchResult to cache.
        """
        if not self.enabled:
            return

        path = self.cache_dir / f"{self._key(query)}.json"
        data = [r.model_dump() for r in results]

        fd, tmp_path = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def clear(self) -> int:
        """Clear all cached results.

        Returns:
            Number of cache entries removed.
        """
        count = 0
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
                count += 1
        return count

    def __len__(self) -> int:
        """Return the number of cached entries."""
        if not self.cache_dir.exists():
            return 0
        return sum(1 for _ in self.cache_dir.glob("*.json"))
