"""Abstract base class and shared model for evidence retrievers.

All retriever implementations (Tavily, Brave, offline cache) satisfy this
interface. RetrievalVerifier (Phase 3) depends only on this contract, never
on a concrete implementation, so swapping search providers requires no
changes to verification logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single search result returned by an evidence retriever.

    Attributes:
        title: Page/document title as reported by the search provider.
        url: Source URL.
        snippet: Short text excerpt from the result (always present).
        content: Full page content if the provider returns it; usually None
            for "basic" search depth.
        relevance_score: Provider-reported relevance in [0.0, 1.0]. Defaults
            to 0.5 when the provider doesn't report a score (e.g. Brave).
    """

    title: str
    url: str
    snippet: str
    content: str | None = None
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.5)


class BaseRetriever(ABC):
    """Interface for retrieving evidence from external sources.

    Implementors must provide:
        - search(): Async evidence retrieval for a query
        - is_available(): Whether this retriever is configured/reachable

    The contract is:
        - Input: a search query string
        - Output: list[SearchResult], possibly empty (never raise on "no results")
        - RetrievalError on actual failures (network error, bad API key,
          rate limit exhaustion) — NOT on legitimate zero-result searches

    Example:
        >>> retriever = TavilyRetriever()
        >>> results = await retriever.search("Eiffel Tower height", max_results=5)
        >>> [r.url for r in results]
    """

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Search for evidence relevant to a query.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.

        Returns:
            List of SearchResult, possibly empty.

        Raises:
            RetrievalError: On network failure, missing credentials, or
                unrecoverable API errors (after retries are exhausted).
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether this retriever is configured and ready to use.

        Returns:
            True if search() can be called, False otherwise (e.g. missing
            API key). Does NOT guarantee network reachability.
        """
        ...
