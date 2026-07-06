"""VeritasCore — Post-hoc LLM verification engine.

A model-agnostic, post-hoc verification layer that audits LLM outputs
for factual accuracy. Decomposes responses into atomic claims, verifies
each against available evidence, and produces structured trust reports.

Example:
    >>> from veritascore import VeritasCoreEngine
    >>> engine = VeritasCoreEngine()
    >>> report = engine.verify(
    ...     response="The Eiffel Tower is 350m tall.",
    ...     query="How tall is the Eiffel Tower?",
    ...     context="The Eiffel Tower is 330 metres tall.",
    ... )
    >>> print(f"Trust Score: {report.overall_trust_score:.2f}")
    >>> print(f"Verdict: {report.overall_verdict.value}")
"""

__version__ = "0.1.0"

from veritascore.core.engine import VeritasCoreEngine

__all__ = ["__version__", "VeritasCoreEngine"]
