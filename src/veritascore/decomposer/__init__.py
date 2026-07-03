"""Claim decomposition module.

Decomposes arbitrary LLM response text into atomic, independently-checkable
factual claims. This is Phase 1 of the VeritasCore pipeline — the foundational
input stage for all downstream verification modules (Phases 2-5).

Public API:
    BaseDecomposer  — abstract interface all decomposers implement
    LLMDecomposer   — primary decomposer using a local LLM (Phi-3-mini default)
    RuleDecomposer  — lightweight, GPU-free fallback decomposer

Example:
    >>> from veritascore.decomposer import LLMDecomposer, RuleDecomposer
    >>>
    >>> # Primary path (requires model download)
    >>> decomposer = LLMDecomposer(fallback_on_error=True)
    >>> claims = decomposer.decompose("Paris is the capital of France.")
    >>>
    >>> # CPU-only / no-model fallback
    >>> decomposer = RuleDecomposer()
    >>> claims = decomposer.decompose("Paris is the capital of France.")
"""

from veritascore.decomposer.base import BaseDecomposer
from veritascore.decomposer.llm_decomposer import LLMDecomposer
from veritascore.decomposer.rule_decomposer import RuleDecomposer

__all__ = ["BaseDecomposer", "LLMDecomposer", "RuleDecomposer"]
