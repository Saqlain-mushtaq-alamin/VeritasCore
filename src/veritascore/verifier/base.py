"""Abstract base class for claim verifiers.

All verifier implementations (NLI-based, retrieval-based, consistency-based)
must satisfy this interface. Downstream phases (5: fusion, 6: explainability)
depend only on this contract — never on a concrete implementation.

Cross-phase contract (Master Plan §7):
    Phase 2 → Phase 5: ClaimVerdict with .nli_score populated
    Phase 3 → Phase 5: ClaimVerdict with .retrieval_score populated
    Phase 4 → Phase 5: ClaimVerdict with .consistency_score populated
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from veritascore.core.types import Claim, ClaimVerdict


class BaseVerifier(ABC):
    """Interface for verifying claims against evidence.

    Implementors must provide:
        - verify(): The core verification logic
        - is_available(): Whether this verifier is ready to run

    The contract is:
        - Input: list[Claim] (from Phase 1) + optional context/query
        - Output: list[ClaimVerdict], one per input claim, in the same order
        - Each ClaimVerdict must have its corresponding score field
          populated (.nli_score for NLIVerifier, .retrieval_score for
          RetrievalVerifier, etc.)
        - VerificationError on failure (never silently drop claims)

    Example:
        >>> verifier = NLIVerifier()
        >>> verdicts = verifier.verify(
        ...     claims=[Claim(text="Paris is the capital of France.", ...)],
        ...     context="Paris is the capital and largest city of France.",
        ... )
        >>> verdicts[0].verdict
        <Verdict.SUPPORTED: 'supported'>
    """

    @abstractmethod
    def verify(
        self,
        claims: list[Claim],
        context: str | None = None,
        query: str | None = None,
    ) -> list[ClaimVerdict]:
        """Verify a list of claims against available evidence.

        Args:
            claims: Atomic claims to verify (from Phase 1 decomposer).
            context: Source context for grounded verification. Required
                for NLIVerifier; ignored by retrieval-based verifiers.
            query: Original user query, used for relevance filtering by
                some verifier implementations (e.g. retrieval-based).

        Returns:
            List of ClaimVerdict, one per input claim, same order as input.

        Raises:
            VerificationError: If verification fails for any reason.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether this verifier is ready to use.

        Returns:
            True if verify() can be called, False otherwise.
        """
        ...
