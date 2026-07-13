"""VeritasCore engine — main orchestrator for the verification pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any

from veritascore.core.config import EngineConfig
from veritascore.core.types import ClaimVerdict, Verdict, VerificationMode, VerificationReport
from veritascore.decomposer.base import BaseDecomposer
from veritascore.decomposer.llm_decomposer import LLMDecomposer
from veritascore.decomposer.rule_decomposer import RuleDecomposer
from veritascore.explainer.evidence_linker import EvidenceLinker
from veritascore.explainer.span_mapper import SpanMapper
from veritascore.profiles.registry import ProfileRegistry
from veritascore.router.mode_router import ModeRouter
from veritascore.scorer.fusion import FusionScorer
from veritascore.verifier.consistency import SemanticConsistencyChecker
from veritascore.verifier.nli_verifier import NLIVerifier
from veritascore.verifier.retrieval_verifier import RetrievalVerifier

logger = logging.getLogger(__name__)


class VeritasCoreEngine:
    """Main entry point for VeritasCore verification.

    Orchestrates: decomposition → verification → consistency → fusion → report.
    All ML models are loaded lazily on first verify() call.

    Example:
        >>> engine = VeritasCoreEngine()
        >>> report = engine.verify(
        ...     response="The Eiffel Tower is 350m tall.",
        ...     query="How tall is the Eiffel Tower?",
        ...     context="The Eiffel Tower is 330 metres tall.",
        ... )
        >>> print(report.overall_trust_score)
    """

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig.default()

        # Lazy-loaded ML components
        self._decomposer: BaseDecomposer | None = None
        self._nli_verifier: NLIVerifier | None = None
        self._retrieval_verifier: RetrievalVerifier | None = None
        self._consistency_checker: SemanticConsistencyChecker | None = None
        self._scorer: FusionScorer | None = None

        # Lightweight non-ML components
        self._span_mapper = SpanMapper()
        self._evidence_linker = EvidenceLinker()
        self._profile_registry = ProfileRegistry()
        self._mode_router = ModeRouter(config=self.config)

    # ── Public API ────────────────────────────────────────────────────────────

    def verify(
        self,
        response: str,
        query: str | None = None,
        context: str | None = None,
        mode: str = "auto",
        domain: str = "general",
    ) -> VerificationReport:
        """Verify an LLM response and return a structured report.

        Args:
            response: The LLM-generated text to verify.
            query: The original user query (optional; required for consistency check).
            context: Source document for grounded verification (optional).
            mode: "auto" (default), "grounded", "ungrounded", or "offline".
            domain: Domain profile name: "general", "medical", "legal", "finance".

        Returns:
            VerificationReport with per-claim verdicts and overall trust score.
        """
        start_time = time.perf_counter()

        profile = self._profile_registry.get_or_default(domain)

        has_context = bool(context and context.strip())
        resolved_mode = self._mode_router.resolve(
            requested_mode=mode,
            has_context=has_context,
            has_query=query is not None,
        )

        decomposer = self._get_decomposer()
        claims = decomposer.decompose(response, query=query)

        if not claims:
            return self._empty_report(response, query, resolved_mode, domain, start_time)

        verdicts = self._run_verification(claims, context, query, resolved_mode, profile)

        if query and query.strip():
            self._apply_consistency_scores(verdicts, claims, query, response)

        scorer = self._get_scorer()
        for v in verdicts:
            v.confidence = scorer.score_claim(v)

        overall_score = scorer.score_response(verdicts)
        overall_verdict = self._aggregate_verdict(verdicts)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return VerificationReport(
            query=query,
            response_text=response,
            claims=verdicts,
            overall_trust_score=overall_score,
            overall_verdict=overall_verdict,
            verification_mode=resolved_mode,
            domain_profile=domain,
            processing_time_ms=elapsed_ms,
            metadata={
                "num_claims": len(claims),
                "resolved_mode": resolved_mode.value,
                "profile_thresholds": profile.thresholds.model_dump(),
            },
        )

    def unload(self) -> None:
        """Unload all ML models from memory, freeing GPU VRAM."""
        if self._decomposer is not None and hasattr(self._decomposer, "unload"):
            self._decomposer.unload()
            self._decomposer = None
        if self._nli_verifier is not None:
            self._nli_verifier.unload()
            self._nli_verifier = None
        if self._retrieval_verifier is not None:
            self._retrieval_verifier = None
        if self._consistency_checker is not None:
            self._consistency_checker.unload()
            self._consistency_checker = None
        self._scorer = None
        logger.info("VeritasCoreEngine: all models unloaded")

    # ── Internal Pipeline Steps ───────────────────────────────────────────────

    def _run_verification(
        self,
        claims: list[Any],
        context: str | None,
        query: str | None,
        mode: VerificationMode,
        profile: Any,
    ) -> list[ClaimVerdict]:
        """Dispatch to the appropriate verifier based on resolved mode."""
        if mode == VerificationMode.GROUNDED:
            verifier = self._get_nli_verifier()
            profile.apply_to_nli_verifier(verifier)
            return verifier.verify(claims, context=context, query=query)
        # UNGROUNDED and OFFLINE both use RetrievalVerifier;
        # in OFFLINE mode the OfflineRetriever returns only cached results.
        return self._get_retrieval_verifier().verify(claims, query=query)

    def _apply_consistency_scores(
        self,
        verdicts: list[ClaimVerdict],
        claims: list[Any],
        query: str,
        response: str,
    ) -> None:
        """Run Phase 4 and write consistency scores into verdicts in-place.

        Bug fix vs. spec: check_consistency() returns a ConsistencyResult
        object (not a dict), so we access .claim_scores as an attribute.
        """
        checker = self._get_consistency_checker()
        result = checker.check_consistency(claims, query=query, response_text=response)
        for v in verdicts:
            v.consistency_score = result.claim_scores.get(v.claim.id, 0.5)

    def _aggregate_verdict(self, verdicts: list[ClaimVerdict]) -> Verdict:
        """Derive overall response verdict from individual claim verdicts.

        Any CONTRADICTED → CONTRADICTED.
        All SUPPORTED → SUPPORTED.
        Otherwise (any UNSUPPORTED, or mixed) → UNSUPPORTED.
        """
        if not verdicts:
            return Verdict.UNSUPPORTED
        if any(v.verdict == Verdict.CONTRADICTED for v in verdicts):
            return Verdict.CONTRADICTED
        if all(v.verdict == Verdict.SUPPORTED for v in verdicts):
            return Verdict.SUPPORTED
        return Verdict.UNSUPPORTED

    def _empty_report(
        self,
        response: str,
        query: str | None,
        mode: VerificationMode,
        domain: str,
        start_time: float,
    ) -> VerificationReport:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return VerificationReport(
            query=query,
            response_text=response,
            claims=[],
            overall_trust_score=0.5,
            overall_verdict=Verdict.UNSUPPORTED,
            verification_mode=mode,
            domain_profile=domain,
            processing_time_ms=elapsed_ms,
            metadata={"num_claims": 0},
        )

    # ── Lazy Component Getters ────────────────────────────────────────────────

    def _get_decomposer(self) -> BaseDecomposer:
        if self._decomposer is None:
            try:
                self._decomposer = LLMDecomposer(config=self.config, fallback_on_error=True)
            except Exception:
                logger.warning("LLMDecomposer failed; using RuleDecomposer fallback")
                self._decomposer = RuleDecomposer()
        return self._decomposer

    def _get_nli_verifier(self) -> NLIVerifier:
        if self._nli_verifier is None:
            self._nli_verifier = NLIVerifier(config=self.config)
        return self._nli_verifier

    def _get_retrieval_verifier(self) -> RetrievalVerifier:
        if self._retrieval_verifier is None:
            self._retrieval_verifier = RetrievalVerifier(
                config=self.config,
                nli_verifier=self._get_nli_verifier(),
            )
        return self._retrieval_verifier

    def _get_consistency_checker(self) -> SemanticConsistencyChecker:
        if self._consistency_checker is None:
            self._consistency_checker = SemanticConsistencyChecker(config=self.config)
        return self._consistency_checker

    def _get_scorer(self) -> FusionScorer:
        if self._scorer is None:
            self._scorer = FusionScorer()
            self._scorer.load()
        return self._scorer
