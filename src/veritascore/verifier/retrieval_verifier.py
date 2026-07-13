"""Retrieval-based claim verification for ungrounded mode.

Used when no source context is provided. Pipeline per claim:
    1. Formulate a search query from the claim text
    2. Retrieve top-K evidence snippets from the web (Tavily/Brave/offline)
    3. Run NLI between the claim and each evidence snippet
    4. Aggregate signals (weighted by source relevance) into a verdict

Reuses the same NLIVerifier from Phase 2 — never loads a second copy of
the NLI model. Claims are verified concurrently (each claim's retrieval +
NLI work is independent), bounded by a semaphore and the retriever's own
rate limiter.
"""

from __future__ import annotations

import asyncio
import logging
from typing import NamedTuple

from veritascore.core.config import EngineConfig
from veritascore.core.types import Claim, ClaimVerdict, Verdict, VerificationMode
from veritascore.retriever.base import BaseRetriever, SearchResult
from veritascore.retriever.brave_retriever import BraveRetriever
from veritascore.retriever.offline_retriever import OfflineRetriever
from veritascore.retriever.tavily_retriever import TavilyRetriever
from veritascore.verifier.base import BaseVerifier
from veritascore.verifier.nli_verifier import NLIVerifier

logger = logging.getLogger(__name__)


class _EvidenceNLIResult(NamedTuple):
    """NLI outcome for a single (evidence, claim) pair, with provenance."""

    probs: object  # numpy array [contradiction, neutral, entailment]
    evidence: str
    url: str
    title: str
    relevance: float


# Prefixes that make a claim a worse search query than its bare assertion
_REMOVE_PREFIXES = (
    "it is ",
    "there is ",
    "there are ",
    "this is ",
    "the fact that ",
    "according to ",
)
_MAX_QUERY_LENGTH = 200


class RetrievalVerifier(BaseVerifier):
    """Verify claims by retrieving web evidence and running NLI.

    Args:
        config: EngineConfig instance. Defaults to EngineConfig.default().
        retriever: BaseRetriever instance. Auto-selected from config if
            not provided (Tavily -> Brave -> Offline, based on configured
            keys).
        nli_verifier: NLIVerifier instance to reuse. Defaults to a new
            instance built from config — pass an existing one to avoid
            loading the NLI model twice.
        max_evidence_per_claim: Max search results retrieved per claim.
        agreement_threshold: Minimum weighted entailment/contradiction
            score required to assign SUPPORTED/CONTRADICTED rather than
            UNSUPPORTED.
        max_concurrent_claims: Maximum claims verified concurrently
            (bounds memory/connection usage during batch verification).

    Example:
        >>> verifier = RetrievalVerifier()
        >>> verdicts = verifier.verify(claims)  # No context — ungrounded mode
        >>> verifier.nli_verifier.unload()
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        retriever: BaseRetriever | None = None,
        nli_verifier: NLIVerifier | None = None,
        max_evidence_per_claim: int = 5,
        agreement_threshold: float = 0.6,
        max_concurrent_claims: int = 8,
    ) -> None:
        if not 0.0 <= agreement_threshold <= 1.0:
            raise ValueError("agreement_threshold must be in [0, 1]")

        self.config = config or EngineConfig.default()
        self.retriever = retriever or self._create_retriever()
        self.nli_verifier = nli_verifier or NLIVerifier(config=self.config)
        self.max_evidence = max_evidence_per_claim
        self.agreement_threshold = agreement_threshold
        self.max_concurrent_claims = max_concurrent_claims

    def _create_retriever(self) -> BaseRetriever:
        """Auto-select a retriever based on config and available API keys.

        Selection order: configured provider (if its key is present) ->
        the other provider (if its key is present) -> OfflineRetriever.
        """
        provider = self.config.search.provider

        if provider == "tavily":
            r: BaseRetriever = TavilyRetriever(config=self.config)
            if r.is_available():
                return r
            brave = BraveRetriever(config=self.config)
            if brave.is_available():
                logger.info("Tavily key missing; falling back to Brave.")
                return brave
        elif provider == "brave":
            r = BraveRetriever(config=self.config)
            if r.is_available():
                return r
            tavily = TavilyRetriever(config=self.config)
            if tavily.is_available():
                logger.info("Brave key missing; falling back to Tavily.")
                return tavily

        logger.warning("No search API key available. Using OfflineRetriever.")
        return OfflineRetriever(config=self.config)

    # ── Public Interface ──────────────────────────────────────────────────────

    def verify(
        self,
        claims: list[Claim],
        context: str | None = None,
        query: str | None = None,
    ) -> list[ClaimVerdict]:
        """Verify claims using web retrieval + NLI.

        Args:
            claims: Atomic claims to verify.
            context: Ignored by RetrievalVerifier (this IS the ungrounded
                path — context is what's absent).
            query: Optional original user query (currently unused in query
                formulation, reserved for future domain-aware search).

        Returns:
            List of ClaimVerdict, one per claim, in input order.

        Raises:
            RuntimeError: If called from within a running event loop. Use
                `await verify_async(...)` directly in that case.
        """
        if not claims:
            return []

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError(
                "verify() cannot be called from within a running event loop; "
                "use 'await RetrievalVerifier.verify_async(...)' instead."
            )

        return asyncio.run(self.verify_async(claims, query))

    async def verify_async(
        self,
        claims: list[Claim],
        query: str | None = None,
    ) -> list[ClaimVerdict]:
        """Async verification pipeline — verifies claims concurrently.

        Args:
            claims: Atomic claims to verify.
            query: Optional original user query.

        Returns:
            List of ClaimVerdict, one per claim, in input order.
        """
        if not claims:
            return []

        self.nli_verifier._load_model()

        semaphore = asyncio.Semaphore(self.max_concurrent_claims)

        async def _verify_one(claim: Claim) -> ClaimVerdict:
            async with semaphore:
                search_query = self._formulate_query(claim, query)
                try:
                    results = await self.retriever.search(
                        search_query, max_results=self.max_evidence
                    )
                except Exception as e:
                    logger.error("Retrieval failed for claim %s: %s", claim.id, e)
                    results = []
                return self._verify_against_evidence(claim, results)

        return list(await asyncio.gather(*(_verify_one(c) for c in claims)))

    # ── Query Formulation ──────────────────────────────────────────────────────

    def _formulate_query(self, claim: Claim, original_query: str | None) -> str:
        """Convert a claim into an effective search query.

        Strips low-value sentence-initial filler ("It is ", "There is ",
        etc.) that adds noise without adding search signal, and caps
        length to stay within typical search API limits.

        Args:
            claim: The claim to formulate a query for.
            original_query: The original user query (currently unused;
                reserved for future query-aware reformulation).

        Returns:
            A search query string derived from claim.text.
        """
        text = claim.text.strip()
        lower = text.lower()

        for prefix in _REMOVE_PREFIXES:
            if lower.startswith(prefix):
                text = text[len(prefix) :]
                break

        if len(text) > _MAX_QUERY_LENGTH:
            text = text[:_MAX_QUERY_LENGTH]

        return text

    # ── Evidence Aggregation ──────────────────────────────────────────────────

    def _verify_against_evidence(self, claim: Claim, results: list[SearchResult]) -> ClaimVerdict:
        """Run NLI between the claim and each evidence snippet, then aggregate.

        Aggregation strategy: relevance-weighted average entailment/
        contradiction across all evidence, combined with the single
        strongest-signal source (selected consistently — verdict, score,
        and evidence are always derived from the SAME source; see the
        Phase 2 chunk-consistency fix for the same class of bug).

        Args:
            claim: The claim being verified.
            results: Search results retrieved for this claim.

        Returns:
            A ClaimVerdict with verification_mode=UNGROUNDED.
        """
        import numpy as np

        if not results:
            return ClaimVerdict(
                claim=claim,
                verdict=Verdict.UNSUPPORTED,
                confidence=0.5,
                retrieval_score=0.0,
                nli_score=0.0,
                evidence=None,
                reason="No evidence retrieved — claim is unsupported by available sources.",
                verification_mode=VerificationMode.UNGROUNDED,
            )

        nli_results: list[_EvidenceNLIResult] = []
        for result in results:
            evidence_text = result.snippet or result.content or ""
            if not evidence_text.strip():
                continue

            probs = self.nli_verifier._run_nli(premise=evidence_text, hypothesis=claim.text)
            nli_results.append(
                _EvidenceNLIResult(
                    probs=probs,
                    evidence=evidence_text,
                    url=result.url,
                    title=result.title,
                    relevance=result.relevance_score,
                )
            )

        if not nli_results:
            return ClaimVerdict(
                claim=claim,
                verdict=Verdict.UNSUPPORTED,
                confidence=0.5,
                retrieval_score=0.0,
                nli_score=0.0,
                evidence=None,
                reason="Retrieved evidence was empty.",
                verification_mode=VerificationMode.UNGROUNDED,
            )

        entail_scores = [float(r.probs[2]) * r.relevance for r in nli_results]  # type: ignore[index]
        contra_scores = [float(r.probs[0]) * r.relevance for r in nli_results]  # type: ignore[index]

        avg_entail = float(np.mean(entail_scores))
        avg_contra = float(np.mean(contra_scores))
        max_entail = max(entail_scores)
        max_contra = max(contra_scores)

        # Select ONE best source consistently: whichever (entailment,
        # contradiction)-weighted signal is strongest overall. All reported
        # fields (verdict basis, evidence, url) come from this same source.
        best_idx = max(
            range(len(nli_results)),
            key=lambda i: max(entail_scores[i], contra_scores[i]),
        )
        best = nli_results[best_idx]
        best_probs = best.probs

        n_support = sum(1 for e in entail_scores if e > 0.5)
        n_contradict = sum(1 for c in contra_scores if c > 0.5)

        if max_entail >= self.agreement_threshold and avg_entail > avg_contra:
            verdict = Verdict.SUPPORTED
            confidence = max_entail
            reason = f"Supported by {n_support}/{len(nli_results)} sources (best: {max_entail:.2f})"
        elif max_contra >= self.agreement_threshold and avg_contra > avg_entail:
            verdict = Verdict.CONTRADICTED
            confidence = max_contra
            reason = (
                f"Contradicted by {n_contradict}/{len(nli_results)} sources "
                f"(best: {max_contra:.2f})"
            )
        else:
            verdict = Verdict.UNSUPPORTED
            confidence = 1.0 - max(max_entail, max_contra)
            reason = (
                f"Insufficient evidence agreement "
                f"(entail: {avg_entail:.2f}, contradict: {avg_contra:.2f})"
            )

        evidence_text = best.evidence[:300]
        if best.url:
            evidence_text += f" [Source: {best.url}]"

        return ClaimVerdict(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            retrieval_score=max_entail,
            nli_score=float(best_probs[2]),  # type: ignore[index]
            evidence=evidence_text,
            reason=reason,
            verification_mode=VerificationMode.UNGROUNDED,
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if both the retriever and the NLI model are usable."""
        return self.retriever.is_available() and self.nli_verifier.is_available()
