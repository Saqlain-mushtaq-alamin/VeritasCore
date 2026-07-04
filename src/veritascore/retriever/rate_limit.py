"""Rate limiting and retry-with-backoff for search API calls.

Addresses Phase 3 Quality Gate G3 criterion 6 ("Rate limiting prevents API
quota exhaustion in benchmarks") and the "Retry logic" deliverable, neither
of which the reference implementation actually included.

Two independent concerns:
    - AsyncRateLimiter: caps the rate of outgoing requests (token bucket)
    - retry_with_backoff: retries a transient failure with exponential
      backoff + jitter, without exceeding a max attempt count
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncRateLimiter:
    """Simple async token-bucket rate limiter.

    Caps requests to `max_calls` per `period_seconds`. Callers `await
    limiter.acquire()` before making a request; it sleeps as needed to
    respect the limit.

    Args:
        max_calls: Maximum number of calls allowed per period.
        period_seconds: Length of the rolling window, in seconds.

    Example:
        >>> limiter = AsyncRateLimiter(max_calls=5, period_seconds=1.0)
        >>> for query in queries:
        ...     await limiter.acquire()
        ...     await do_search(query)
    """

    def __init__(self, max_calls: int, period_seconds: float = 1.0) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be positive")
        if period_seconds <= 0:
            raise ValueError("period_seconds must be positive")

        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self._call_times: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a call is permitted under the rate limit."""
        async with self._lock:
            now = time.monotonic()
            cutoff = now - self.period_seconds
            self._call_times = [t for t in self._call_times if t > cutoff]

            if len(self._call_times) >= self.max_calls:
                oldest = self._call_times[0]
                sleep_for = (oldest + self.period_seconds) - now
                if sleep_for > 0:
                    logger.debug("Rate limit reached; sleeping %.2fs", sleep_for)
                    await asyncio.sleep(sleep_for)
                now = time.monotonic()
                cutoff = now - self.period_seconds
                self._call_times = [t for t in self._call_times if t > cutoff]

            self._call_times.append(time.monotonic())


async def retry_with_backoff(
    fn: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> T:
    """Retry an async callable with exponential backoff and jitter.

    Args:
        fn: Zero-argument async callable to retry (wrap with a lambda/partial
            to bind arguments).
        max_attempts: Maximum number of attempts (including the first).
        base_delay: Base delay in seconds before the first retry.
        max_delay: Maximum delay cap, in seconds.
        retryable_exceptions: Exception types that should trigger a retry.
            Other exceptions propagate immediately.

    Returns:
        The return value of `fn()` on the first successful attempt.

    Raises:
        The last exception encountered, if all attempts are exhausted.

    Example:
        >>> result = await retry_with_backoff(
        ...     lambda: client.get(url),
        ...     max_attempts=3,
        ...     retryable_exceptions=(httpx.TransportError,),
        ... )
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    last_exception: Exception | None = None

    for attempt in range(max_attempts):
        try:
            return await fn()
        except retryable_exceptions as e:
            last_exception = e
            if attempt == max_attempts - 1:
                break
            delay = min(base_delay * (2**attempt), max_delay)
            jitter = random.uniform(0, delay * 0.1)
            sleep_for = delay + jitter
            logger.warning(
                "Attempt %d/%d failed (%s); retrying in %.2fs",
                attempt + 1, max_attempts, e, sleep_for,
            )
            await asyncio.sleep(sleep_for)

    assert last_exception is not None  # Reached only if all attempts failed
    raise last_exception
