"""Auto-routing logic for verification mode selection (FR9).

Priority:
    1. Manual override (any value that is not "auto")
    2. GROUNDED  — context was provided and is non-empty
    3. UNGROUNDED — internet is reachable AND a search API key is configured
    4. OFFLINE   — fallback when no connectivity or no search keys

The internet check uses a fast TCP probe to Google's public DNS resolver
(8.8.8.8:53, 2 s timeout). This is intentionally lightweight — the probe
just tests whether a TCP connection can be established, not whether the
search provider's API is up. A probe failure silently falls back to OFFLINE
without raising; the caller should treat OFFLINE mode as "best-effort."
"""

from __future__ import annotations

import logging
import socket

from veritascore.core.config import EngineConfig
from veritascore.core.types import VerificationMode

logger = logging.getLogger(__name__)

_INTERNET_PROBE_HOST = "8.8.8.8"
_INTERNET_PROBE_PORT = 53
_INTERNET_PROBE_TIMEOUT = 2.0


class ModeRouter:
    """Automatically select verification mode based on inputs and connectivity.

    Args:
        config: EngineConfig instance. Defaults to EngineConfig.default().

    Example:
        >>> router = ModeRouter()
        >>> mode = router.resolve(requested_mode="auto", has_context=True)
        >>> mode
        <VerificationMode.GROUNDED: 'grounded'>
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig.default()

    def resolve(
        self,
        requested_mode: str = "auto",
        has_context: bool = False,
        has_query: bool = False,
    ) -> VerificationMode:
        """Resolve the verification mode.

        Args:
            requested_mode: The mode string from the caller. If not "auto",
                converted directly to VerificationMode (raises ValueError for
                unknown strings).
            has_context: True if a non-empty context string was provided.
            has_query: True if an original query was provided (not currently
                used in routing decisions, reserved for future expansion).

        Returns:
            The resolved VerificationMode enum value.

        Raises:
            ValueError: If requested_mode is not a valid VerificationMode
                value and is not "auto".
        """
        if requested_mode != "auto":
            return VerificationMode(requested_mode)

        if has_context:
            logger.info("Mode resolved: GROUNDED (context provided)")
            return VerificationMode.GROUNDED

        if self._has_internet() and self._has_search_api():
            logger.info("Mode resolved: UNGROUNDED (internet + search API available)")
            return VerificationMode.UNGROUNDED

        logger.info("Mode resolved: OFFLINE (no internet or no search API key configured)")
        return VerificationMode.OFFLINE

    def _has_internet(self, timeout: float = _INTERNET_PROBE_TIMEOUT) -> bool:
        """Return True if a TCP connection to Google's public DNS can be established."""
        try:
            socket.create_connection(
                (_INTERNET_PROBE_HOST, _INTERNET_PROBE_PORT), timeout=timeout
            ).close()
            return True
        except OSError:
            return False

    def _has_search_api(self) -> bool:
        """Return True if at least one search API key is configured."""
        return bool(
            self.config.search.tavily_api_key or self.config.search.brave_api_key
        )
