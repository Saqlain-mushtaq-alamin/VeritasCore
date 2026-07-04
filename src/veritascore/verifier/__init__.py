"""Claim verification module — grounded (NLI-based) verification.

Verifies atomic claims (from Phase 1) against evidence. This module
implements Phase 2: grounded verification using Natural Language
Inference (NLI) against caller-supplied context.

Public API:
    BaseVerifier  — abstract interface all verifiers implement
    NLIVerifier   — NLI cross-encoder verification against provided context

Example:
    >>> from veritascore.verifier import NLIVerifier
    >>> verifier = NLIVerifier()
    >>> verdicts = verifier.verify(claims, context="The Eiffel Tower is 330m tall.")
    >>> verifier.unload()
"""

from veritascore.verifier.base import BaseVerifier
from veritascore.verifier.nli_verifier import NLIVerifier

__all__ = ["BaseVerifier", "NLIVerifier"]
