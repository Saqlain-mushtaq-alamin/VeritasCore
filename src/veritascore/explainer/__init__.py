"""Explainability module — span mapping and evidence chain construction.

Every ClaimVerdict in VeritasCore is fully traceable: this module provides
the tools to surface that traceability to end-users and downstream systems.

Public API:
    HighlightedSpan   — a character span with verdict metadata
    SpanMapper        — maps claims to spans; renders annotated text
    EvidenceChain     — complete traceability record for one claim verdict
    EvidenceLinker    — builds chains; validates traceability completeness
"""

from veritascore.explainer.evidence_linker import EvidenceChain, EvidenceLinker
from veritascore.explainer.span_mapper import HighlightedSpan, SpanMapper

__all__ = [
    "HighlightedSpan",
    "SpanMapper",
    "EvidenceChain",
    "EvidenceLinker",
]
