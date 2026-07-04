"""Abstract base class for fusion/confidence scorers.

All scorer implementations (heuristic fallback, trained LogisticRegression/
XGBoost fusion) must satisfy this interface. Downstream phases (6:
explainability, 7: API) depend only on this contract, never on a concrete
implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from veritascore.core.types import ClaimVerdict


class BaseScorer(ABC):
    """Interface for fusing verification signals into trust scores.

    Implementors must provide:
        - score_claim(): Per-claim calibrated confidence
        - score_response(): Response-level aggregate trust score
        - is_available(): Whether this scorer is ready to run

    The contract is:
        - Input: a ClaimVerdict (with whichever of .nli_score,
          .retrieval_score, .consistency_score happen to be populated —
          any subset may be None) or a list thereof
        - Output: a float confidence in [0.0, 1.0]
        - Missing signals must be handled gracefully (never raise on a
          None score field; treat as "no information," not as a crash)

    Example:
        >>> scorer = FusionScorer()
        >>> scorer.load()
        >>> confidence = scorer.score_claim(verdict)
        >>> trust_score = scorer.score_response(verdicts)
    """

    @abstractmethod
    def score_claim(self, verdict: ClaimVerdict) -> float:
        """Score a single claim verdict.

        Args:
            verdict: A ClaimVerdict with zero or more of .nli_score,
                .retrieval_score, .consistency_score populated.

        Returns:
            Calibrated confidence in [0.0, 1.0].
        """
        ...

    @abstractmethod
    def score_response(self, verdicts: list[ClaimVerdict]) -> float:
        """Compute an overall response-level trust score from claim verdicts.

        Args:
            verdicts: All claim verdicts for a single response. May be empty.

        Returns:
            Aggregate trust score in [0.0, 1.0]. Implementations should
            return a neutral score (e.g. 0.5) for an empty list rather
            than raising.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether this scorer is ready to use.

        Returns:
            True if score_claim()/score_response() can be called.
        """
        ...
