"""Semantic consistency checker — detects off-topic/non-responsive claims.

Catches answers that are factually correct but irrelevant to what the user
asked. A claim can be SUPPORTED by NLI (Phase 2) or retrieval (Phase 3) yet
be completely non-responsive to the query — this module is the only one in
the pipeline that checks relevance rather than truth.

Example:
    Query:    "What is the capital of France?"
    Response: "The Eiffel Tower is 330 meters tall."
    -> Factually true, but off-topic. SemanticConsistencyChecker flags this;
       NLI/retrieval verifiers would not, since they only check truth.

Uses sentence-transformers (default: all-MiniLM-L6-v2, ~80MB) for three
related signals:
    1. Query-claim relevance — is each claim on-topic?
    2. Claim-claim coherence — are the claims internally consistent with
       each other (not contradicting topic, not wildly scattered)?
    3. Response-query relevance — does the response as a whole address
       the query?

MiniLM is small enough to coexist with the NLI model in VRAM; unlike
LLMDecomposer/NLIVerifier, there is no memory-pressure reason to unload it
between phases, though unload() is still provided for completeness.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from veritascore.core.config import EngineConfig
from veritascore.core.exceptions import ModelLoadError, VerificationError
from veritascore.core.types import Claim

logger = logging.getLogger(__name__)

# Cosine similarity from normalized sentence-transformer embeddings is
# mathematically bounded to [-1, 1], but typical text pairs land in
# roughly [-0.3, 0.9] (see Phase 4 spec §4.9). We do NOT clamp to [0, 1]
# for response_relevance/coherence_score, since collapsing the negative
# range would silently hide genuinely anti-correlated (contradictory-topic)
# pairs as if they were merely "a bit irrelevant" (score 0.0). Per-claim
# relevance scores ARE clamped to [0, 1], matching the documented interface
# contract (".consistency_score: float — 0.0 (irrelevant) to 1.0 (highly
# relevant)") and ClaimVerdict.consistency_score's Pydantic ge=0/le=1 bounds.
_RAW_SIMILARITY_FLOOR = -1.0
_RAW_SIMILARITY_CEIL = 1.0


class ConsistencyResult:
    """Structured result of a consistency check, returned by check_consistency().

    Attributes:
        claim_scores: Mapping of claim.id -> relevance score in [0.0, 1.0].
        coherence_score: Average pairwise claim-claim similarity, in
            [-1.0, 1.0] (NOT clamped to [0,1] — see module docstring).
            1.0 for single-claim or empty claim lists (vacuously coherent).
        response_relevance: Query-response similarity, in [-1.0, 1.0]
            (NOT clamped to [0,1]).
        off_topic_claims: IDs of claims with relevance below the configured
            threshold.
    """

    __slots__ = ("claim_scores", "coherence_score", "response_relevance", "off_topic_claims")

    def __init__(
        self,
        claim_scores: dict[str, float],
        coherence_score: float,
        response_relevance: float,
        off_topic_claims: list[str],
    ) -> None:
        self.claim_scores = claim_scores
        self.coherence_score = coherence_score
        self.response_relevance = response_relevance
        self.off_topic_claims = off_topic_claims

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dict, matching the spec's documented
        return shape exactly (for callers that prefer dict access)."""
        return {
            "claim_scores": self.claim_scores,
            "coherence_score": self.coherence_score,
            "response_relevance": self.response_relevance,
            "off_topic_claims": self.off_topic_claims,
        }

    def __repr__(self) -> str:
        return (
            f"ConsistencyResult(n_claims={len(self.claim_scores)}, "
            f"coherence={self.coherence_score:.2f}, "
            f"response_relevance={self.response_relevance:.2f}, "
            f"n_off_topic={len(self.off_topic_claims)})"
        )


class SemanticConsistencyChecker:
    """Check semantic consistency between claims and the original query.

    Args:
        config: EngineConfig instance. Defaults to EngineConfig.default().
        relevance_threshold: Minimum query-claim cosine similarity for a
            claim to be considered on-topic. Claims scoring below this are
            added to `off_topic_claims`. Default 0.3, per the Phase 4 spec's
            own characterization as "conservative" — tune against
            tests/fixtures/consistency_samples.json for your domain.

    Example:
        >>> checker = SemanticConsistencyChecker()
        >>> result = checker.check_consistency(
        ...     claims=[Claim(text="Paris is the capital of France.", ...)],
        ...     query="What is the capital of France?",
        ...     response_text="Paris is the capital of France.",
        ... )
        >>> result.claim_scores
        {'c1': 0.78}
        >>> result.off_topic_claims
        []
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        relevance_threshold: float = 0.3,
    ) -> None:
        if not -1.0 <= relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be in [-1.0, 1.0]")

        self.config = config or EngineConfig.default()
        self.relevance_threshold = relevance_threshold
        self._model: Any = None
        self._loaded: bool = False

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Lazy-load the sentence-transformer embedding model (idempotent)."""
        if self._loaded:
            return

        from sentence_transformers import SentenceTransformer

        model_name = self.config.models.embedding_model
        logger.info("Loading embedding model: %s", model_name)
        t0 = time.time()

        try:
            self._model = SentenceTransformer(model_name)
            device = self._resolve_device()
            self._model = self._model.to(device)
        except Exception as e:
            raise ModelLoadError(f"Failed to load embedding model '{model_name}': {e}") from e

        elapsed = time.time() - t0
        logger.info("Embedding model loaded in %.2fs", elapsed)
        self._loaded = True

    def _resolve_device(self) -> str:
        """Determine the inference device. MiniLM is small enough that CPU
        is often fine too, but we respect the configured device like other
        modules for consistency."""
        import torch

        device = self.config.models.device
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "Configured device '%s' requested but CUDA unavailable; falling back to CPU.",
                device,
            )
            return "cpu"
        return device

    # ── Public Interface ──────────────────────────────────────────────────────

    def check_consistency(
        self,
        claims: list[Claim],
        query: str,
        response_text: str,
    ) -> ConsistencyResult:
        """Compute consistency scores for all claims against the query.

        Args:
            claims: Atomic claims to score (from Phase 1 decomposer).
            query: The original user query. Required — this is the entire
                point of the module (unlike NLIVerifier's context, there's
                no meaningful ungrounded/offline equivalent for "relevance
                to nothing").
            response_text: The full original LLM response text.

        Returns:
            A ConsistencyResult with per-claim scores, coherence, overall
            response relevance, and the list of flagged off-topic claim IDs.

        Raises:
            VerificationError: If query is None or empty/whitespace-only.
        """
        if query is None or not query.strip():
            raise VerificationError(
                "SemanticConsistencyChecker requires a non-empty query — "
                "relevance is meaningless without something to be relevant to."
            )

        self._load_model()

        if not claims:
            return ConsistencyResult(
                claim_scores={},
                coherence_score=1.0,
                response_relevance=self._pairwise_similarity(query, response_text)
                if response_text and response_text.strip()
                else 1.0,
                off_topic_claims=[],
            )

        texts = [query, response_text] + [c.text for c in claims]
        embeddings = self._model.encode(texts, normalize_embeddings=True)

        query_emb = embeddings[0]
        response_emb = embeddings[1]
        claim_embs = embeddings[2:]

        # 1. Query-claim relevance (clamped to [0,1] per the documented
        #    per-claim contract and ClaimVerdict.consistency_score bounds)
        claim_scores: dict[str, float] = {}
        off_topic: list[str] = []
        for i, claim in enumerate(claims):
            raw_sim = float(self._dot(query_emb, claim_embs[i]))
            clamped = max(0.0, min(1.0, raw_sim))
            claim_scores[claim.id] = clamped
            if raw_sim < self.relevance_threshold:
                off_topic.append(claim.id)

        # 2. Claim-claim coherence (average pairwise similarity, NOT clamped
        #    to [0,1] — see module docstring)
        if len(claims) > 1:
            pairwise_sims = []
            for i in range(len(claim_embs)):
                for j in range(i + 1, len(claim_embs)):
                    pairwise_sims.append(float(self._dot(claim_embs[i], claim_embs[j])))
            coherence = sum(pairwise_sims) / len(pairwise_sims)
        else:
            coherence = 1.0  # Vacuously coherent — nothing to disagree with

        # 3. Response-query relevance (NOT clamped to [0,1])
        response_relevance = float(self._dot(query_emb, response_emb))

        return ConsistencyResult(
            claim_scores=claim_scores,
            coherence_score=max(_RAW_SIMILARITY_FLOOR, min(_RAW_SIMILARITY_CEIL, coherence)),
            response_relevance=max(
                _RAW_SIMILARITY_FLOOR, min(_RAW_SIMILARITY_CEIL, response_relevance)
            ),
            off_topic_claims=off_topic,
        )

    def score_claim(self, claim: Claim, query: str) -> float:
        """Score a single claim's relevance to the query.

        Args:
            claim: The claim to score.
            query: The original user query.

        Returns:
            Cosine similarity in [0.0, 1.0] (clamped — negative similarity
            is reported as 0.0, i.e. "not relevant," matching the per-claim
            contract).

        Raises:
            VerificationError: If query is None or empty/whitespace-only.
        """
        if query is None or not query.strip():
            raise VerificationError("SemanticConsistencyChecker requires a non-empty query.")

        self._load_model()
        embs = self._model.encode([query, claim.text], normalize_embeddings=True)
        raw_sim = float(self._dot(embs[0], embs[1]))
        return max(0.0, min(1.0, raw_sim))

    @staticmethod
    def _dot(a: Any, b: Any) -> float:
        """Dot product of two normalized embedding vectors == cosine similarity."""
        import numpy as np

        return float(np.dot(a, b))

    def _pairwise_similarity(self, text_a: str, text_b: str) -> float:
        """Helper for the empty-claims response_relevance fallback path."""
        embs = self._model.encode([text_a, text_b], normalize_embeddings=True)
        return max(
            _RAW_SIMILARITY_FLOOR,
            min(_RAW_SIMILARITY_CEIL, float(self._dot(embs[0], embs[1]))),
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if the embedding model can be loaded successfully."""
        try:
            self._load_model()
            return True
        except ModelLoadError:
            return False

    def unload(self) -> None:
        """Unload the embedding model from memory.

        MiniLM is tiny (~80MB) and can coexist with the NLI model in VRAM
        (Phase 4 spec §4.9), so calling this is rarely necessary — provided
        mainly for symmetry with other modules and explicit test teardown.
        """
        if not self._loaded:
            return

        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        logger.info("Embedding model unloaded")
