"""Link evidence to verdicts for full traceability (G6 criteria 1, 2, 8)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from veritascore.core.types import ClaimVerdict, Verdict

_SOURCE_URL_PATTERN = re.compile(r"\[Source:\s*(https?://[^\]]+)\]")


@dataclass
class EvidenceChain:
    """Complete traceability chain for a single claim verdict.

    Attributes:
        claim_id: ID of the Claim.
        claim_text: The atomic claim text.
        source_span: Character offsets (start, end) in the original response.
        verdict: Verdict value string.
        confidence: Calibrated confidence in [0.0, 1.0].
        evidence_snippet: Short evidence text (may be None for unsupported claims).
        evidence_source: URL or 'provided context' or None.
        reason: Human-readable explanation of the verdict.
        verification_mode: Verification mode value string.
        signals: Raw signal values from Phases 2-4.
    """

    claim_id: str
    claim_text: str
    source_span: tuple[int, int]
    verdict: str
    confidence: float
    evidence_snippet: str | None
    evidence_source: str | None
    reason: str
    verification_mode: str
    signals: dict[str, float | None] = field(default_factory=dict)


class EvidenceLinker:
    """Construct complete evidence chains for every verdict.

    Example:
        >>> linker = EvidenceLinker()
        >>> chains = linker.build_chains(verdicts)
        >>> issues = linker.validate_traceability(verdicts)
        >>> assert issues == [], issues
    """

    def build_chains(self, verdicts: list[ClaimVerdict]) -> list[EvidenceChain]:
        """Build an EvidenceChain for each ClaimVerdict.

        Args:
            verdicts: All ClaimVerdict objects for a response.

        Returns:
            List of EvidenceChain, one per verdict, in the same order.
        """
        return [self._build_one(v) for v in verdicts]

    def _build_one(self, v: ClaimVerdict) -> EvidenceChain:
        source = self._extract_source(v.evidence, v.verification_mode.value)
        return EvidenceChain(
            claim_id=v.claim.id,
            claim_text=v.claim.text,
            source_span=v.claim.source_span,
            verdict=v.verdict.value,
            confidence=v.confidence,
            evidence_snippet=v.evidence,
            evidence_source=source,
            reason=v.reason,
            verification_mode=v.verification_mode.value,
            signals={
                "nli_score": v.nli_score,
                "retrieval_score": v.retrieval_score,
                "consistency_score": v.consistency_score,
            },
        )

    @staticmethod
    def _extract_source(evidence: str | None, mode: str) -> str | None:
        """Extract or infer the evidence source.

        For grounded mode the source is the caller-provided context.
        For ungrounded mode, RetrievalVerifier appends a [Source: url]
        annotation — extract it if present.
        """
        if not evidence:
            return None
        if mode == "grounded":
            return "provided context"
        m = _SOURCE_URL_PATTERN.search(evidence)
        if m:
            return m.group(1)
        return "web search"

    def validate_traceability(self, verdicts: list[ClaimVerdict]) -> list[str]:
        """Validate that all claims have complete traceability.

        Implements Quality Gate G6 criteria 1 and 2:
            Criterion 1: Every non-supported (flagged) claim has a non-empty reason.
            Criterion 2: Every CONTRADICTED claim has non-empty evidence.

        Args:
            verdicts: All ClaimVerdict objects to validate.

        Returns:
            List of issue strings. Empty = fully traceable.
        """
        issues: list[str] = []
        for v in verdicts:
            cid = v.claim.id

            if not v.reason or not v.reason.strip():
                issues.append(f"Claim {cid}: missing reason")

            if v.verdict == Verdict.CONTRADICTED and (not v.evidence or not v.evidence.strip()):
                issues.append(f"Claim {cid}: verdict is CONTRADICTED but evidence is missing")

        return issues

    def format_chain_text(self, chain: EvidenceChain) -> str:
        """Format an EvidenceChain as a human-readable multi-line string.

        Format:
            Claim [id]: "text"
              Verdict:  SUPPORTED (confidence: 0.91)
              Mode:     grounded
              Reason:   Entailed by context
              Evidence: "snippet..." (Source: provided context)
              Signals:  nli=0.91, retrieval=None, consistency=0.78
        """
        lines = [
            f'Claim [{chain.claim_id}]: "{chain.claim_text}"',
            f"  Verdict:  {chain.verdict.upper()} (confidence: {chain.confidence:.2f})",
            f"  Mode:     {chain.verification_mode}",
            f"  Reason:   {chain.reason}",
        ]
        if chain.evidence_snippet:
            snippet = chain.evidence_snippet[:120]
            src = f" (Source: {chain.evidence_source})" if chain.evidence_source else ""
            lines.append(f'  Evidence: "{snippet}"{src}')
        else:
            lines.append("  Evidence: None")
        sigs = chain.signals
        sig_str = (
            f"nli={sigs.get('nli_score')}, "
            f"retrieval={sigs.get('retrieval_score')}, "
            f"consistency={sigs.get('consistency_score')}"
        )
        lines.append(f"  Signals:  {sig_str}")
        return "\n".join(lines)
