"""Evidence retrieval module — web search for ungrounded verification.

Retrieves evidence from external search APIs to support Phase 3
(ungrounded verification), used when no source context is provided.

Public API:
    BaseRetriever     — abstract interface all retrievers implement
    SearchResult      — single search result model
    TavilyRetriever   — Tavily Search API (free tier: 1000 req/month)
    BraveRetriever    — Brave Search API (free tier: 2000 req/month)
    OfflineRetriever  — cached-only fallback, no network calls
    SearchCache       — file-based result cache shared by all retrievers
    AsyncRateLimiter  — token-bucket rate limiter for API calls
    retry_with_backoff — exponential-backoff retry helper

Example:
    >>> from veritascore.retriever import TavilyRetriever
    >>> retriever = TavilyRetriever()
    >>> results = await retriever.search("Eiffel Tower height")
"""

from veritascore.retriever.base import BaseRetriever, SearchResult
from veritascore.retriever.brave_retriever import BraveRetriever
from veritascore.retriever.cache import SearchCache
from veritascore.retriever.offline_retriever import OfflineRetriever
from veritascore.retriever.rate_limit import AsyncRateLimiter, retry_with_backoff
from veritascore.retriever.tavily_retriever import TavilyRetriever

__all__ = [
    "BaseRetriever",
    "SearchResult",
    "TavilyRetriever",
    "BraveRetriever",
    "OfflineRetriever",
    "SearchCache",
    "AsyncRateLimiter",
    "retry_with_backoff",
]
