"""Abstract base class for claim decomposers.

All decomposer implementations (LLM-based, rule-based, future fine-tuned)
must satisfy this interface. Downstream phases (2, 3, 4) depend only on
this contract — never on a concrete implementation.

Cross-phase contract (Master Plan §7):
    Phase 1 → Phases 2, 3, 4: list[Claim]
    Each Claim has: .text, .source_span, .source_text, .id
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from veritascore.core.types import Claim


class BaseDecomposer(ABC):
    """Interface for decomposing LLM responses into atomic factual claims.

    Implementors must provide:
        - decompose(): The core extraction logic
        - is_available(): Whether this decomposer is ready to run

    The contract is:
        - Input: raw LLM response string (and optional query for context)
        - Output: list[Claim] where each Claim has a valid source_span
          within the bounds of the input response_text.
        - Empty input → empty list (never raise on empty string)
        - DecompositionError on any failure during extraction

    Example:
        >>> decomposer = RuleDecomposer()
        >>> claims = decomposer.decompose(
        ...     "Einstein was born in 1879 in Germany.",
        ...     query="Tell me about Einstein.",
        ... )
        >>> for claim in claims:
        ...     print(claim.text, claim.source_span)
    """

    @abstractmethod
    def decompose(
        self,
        response_text: str,
        query: str | None = None,
    ) -> list[Claim]:
        """Decompose an LLM response into atomic factual claims.

        Each returned Claim must:
        - Be a single, independently verifiable factual assertion
        - Have a source_span (start, end) within [0, len(response_text)]
        - Have source_text = response_text[start:end] (approximately)
        - Be self-contained (no unresolved pronouns or references)

        Args:
            response_text: The full LLM response string to decompose.
            query: Optional original user query, used to give the decomposer
                context about what kind of claims are relevant.

        Returns:
            List of Claim objects. May be empty if no factual claims exist.

        Raises:
            DecompositionError: If decomposition fails for any reason.
                Callers should catch this and fall back to RuleDecomposer.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether this decomposer is ready to use.

        For LLM-based decomposers, this checks if the model can be loaded.
        For rule-based decomposers, this always returns True.

        Returns:
            True if decompose() can be called, False otherwise.
        """
        ...

    def decompose_batch(
        self,
        responses: list[str],
        queries: list[str | None] | None = None,
    ) -> list[list[Claim]]:
        """Decompose multiple responses. Default: sequential calls to decompose().

        Subclasses may override this for true batch inference.

        Args:
            responses: List of LLM response strings.
            queries: Optional list of queries, one per response.

        Returns:
            List of claim lists, one per input response.
        """
        if queries is None:
            queries = [None] * len(responses)
        return [self.decompose(resp, query) for resp, query in zip(responses, queries, strict=True)]
