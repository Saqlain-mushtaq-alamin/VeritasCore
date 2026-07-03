"""Rule-based claim decomposition — lightweight fallback requiring no GPU.

This decomposer uses only regex and heuristics (no ML models). It is:
    - Always available (CPU-only, no model download required)
    - Deterministic (same input → same output, every time)
    - Fast (<50ms per response)
    - The fallback when LLMDecomposer fails or is unavailable

Quality trade-off: lower atomic accuracy than LLMDecomposer (~70-80% vs >90%),
but sufficient as a fallback and useful for unit testing without GPU.

Algorithm:
    1. Split text into sentences (punctuation-aware, abbreviation-safe)
    2. Split compound sentences on coordinating conjunctions
    3. Filter non-factual sentences (questions, opinions, commands)
    4. Map each remaining sentence to a source span
    5. Produce Claim objects
"""

from __future__ import annotations

import logging
import re

from veritascore.core.exceptions import DecompositionError
from veritascore.core.types import Claim
from veritascore.decomposer.base import BaseDecomposer
from veritascore.decomposer.span_mapper import (
    SentenceSpan,
    split_into_sentences,
)

logger = logging.getLogger(__name__)


# ── Non-factual Heuristics ────────────────────────────────────────────────────

# Sentence-initial words/phrases that indicate non-factual content
_OPINION_PREFIXES: frozenset[str] = frozenset([
    "i think", "i believe", "i feel", "i suspect",
    "in my opinion", "in my view", "personally,",
    "it seems", "it appears", "it looks like",
    "perhaps", "maybe", "possibly", "probably",
    "reportedly", "allegedly", "supposedly",
    "note that", "please note", "please remember",
    "consider ", "remember that",
])

# Patterns for sentences that are questions
_QUESTION_PATTERN = re.compile(
    r'^\s*(?:who|what|when|where|why|how|is|are|was|were|do|does|did'
    r'|can|could|would|should|will|shall|have|has|had)\b.*\?\s*$',
    re.IGNORECASE,
)

# Conjunctions used to join independent clauses (require comma before them)
_CONJUNCTION_SPLITS = [
    r',\s+and\s+',
    r',\s+but\s+',
    r',\s+while\s+',
    r',\s+whereas\s+',
    r',\s+yet\s+',
    r';\s+',
]

# Minimum character length for a sentence to be considered a claim
_MIN_CLAIM_LENGTH = 12

# Maximum compound splits per sentence (prevents over-splitting)
_MAX_COMPOUND_SPLITS = 4


# Hedge/opinion words that, when present, mark a sentence as non-factual
# even if not at the very start (e.g. "X is probably the biggest...")
_HEDGE_WORDS: frozenset[str] = frozenset([
    "probably", "perhaps", "possibly", "maybe", "arguably",
    "reportedly", "allegedly", "supposedly", "presumably",
])


def _is_factual(text: str) -> bool:
    """Return True if the sentence looks like a factual claim.

    Filters out:
        - Empty / very short strings
        - Questions
        - Opinion/hedging language (prefix or embedded hedge words)
        - Commands and imperatives
        - List headers ("The following...", "For example:")

    Args:
        text: Sentence text to evaluate.

    Returns:
        True if this looks like a factual claim worth keeping.
    """
    stripped = text.strip()
    if len(stripped) < _MIN_CLAIM_LENGTH:
        return False

    lower = stripped.lower()

    # Filter questions
    if _QUESTION_PATTERN.match(stripped):
        return False

    # Filter trailing question marks
    if stripped.endswith("?"):
        return False

    # Filter opinion/hedging prefixes
    for prefix in _OPINION_PREFIXES:
        if lower.startswith(prefix):
            return False

    # Filter embedded hedge words (word-boundary match, anywhere in sentence)
    words = set(re.findall(r"\b\w+\b", lower))
    if words & _HEDGE_WORDS:
        return False

    # Filter list headers / meta-commentary
    meta_patterns = [
        r'^the following\b',
        r'^here (is|are)\b',
        r'^for example[,:]',
        r'^such as[,:]',
        r'^including\b',
        r'^\(.*\)$',          # Pure parenthetical
        r'^note:',
        r'^important:',
    ]
    return all(not re.match(pat, lower) for pat in meta_patterns)


def _split_compound_sentence(
    sent: SentenceSpan,
) -> list[SentenceSpan]:
    """Try to split a compound sentence into simpler factual parts.

    Only splits when the conjunction joins two reasonably long clauses
    (both sides ≥ 15 characters), preventing over-splitting of phrases
    like "cats and dogs".

    Args:
        sent: The sentence to attempt splitting.

    Returns:
        List of SentenceSpan objects. Either [sent] (no split) or 2+ parts.
    """
    parts: list[SentenceSpan] = [sent]

    for conj_pattern in _CONJUNCTION_SPLITS:
        new_parts: list[SentenceSpan] = []
        did_split = False

        for part in parts:
            if did_split and len(parts) >= _MAX_COMPOUND_SPLITS:
                new_parts.append(part)
                continue

            match = re.search(conj_pattern, part.text, re.IGNORECASE)
            if match:
                left = part.text[: match.start()].strip()
                right = part.text[match.end() :].strip()

                # Only split if both sides are substantial
                if len(left) >= 15 and len(right) >= 15:
                    left_start = part.start
                    left_end = part.start + match.start()
                    right_start = part.start + match.end()
                    right_end = part.end

                    new_parts.append(SentenceSpan(left_start, left_end, left))
                    new_parts.append(SentenceSpan(right_start, right_end, right))
                    did_split = True
                    continue

            new_parts.append(part)

        parts = new_parts

    return parts


class RuleDecomposer(BaseDecomposer):
    """Decompose LLM responses using rule-based sentence splitting and filtering.

    No ML model required. Always available. Deterministic.

    This decomposer is used:
        1. As the fallback when LLMDecomposer fails or is unavailable
        2. In unit tests (no GPU / model download needed)
        3. In CPU-only deployment environments
        4. For rapid prototyping and debugging

    Quality characteristics:
        - Atomic accuracy: ~75% (vs >90% for LLMDecomposer)
        - Does NOT resolve pronouns (limitation vs LLM)
        - Does NOT detect all compound facts within a single clause
        - Speed: <5ms per response

    Example:
        >>> decomposer = RuleDecomposer()
        >>> claims = decomposer.decompose(
        ...     "The Eiffel Tower is 330m tall. It was built in 1889."
        ... )
        >>> [c.text for c in claims]
        ['The Eiffel Tower is 330m tall.', 'It was built in 1889.']
    """

    def decompose(
        self,
        response_text: str,
        query: str | None = None,
    ) -> list[Claim]:
        """Decompose a response into factual claims using rules only.

        Args:
            response_text: The LLM response to decompose.
            query: Ignored (rule decomposer has no query-awareness).

        Returns:
            List of Claim objects with source spans.

        Raises:
            DecompositionError: On unexpected processing failure.
        """
        if not response_text or not response_text.strip():
            return []

        try:
            # Step 1: Split into sentences with offsets
            sentences = split_into_sentences(response_text)

            # Step 2: Attempt compound sentence splitting
            atomic_parts: list[SentenceSpan] = []
            for sent in sentences:
                splits = _split_compound_sentence(sent)
                atomic_parts.extend(splits)

            # Step 3: Filter non-factual sentences
            factual_parts = [p for p in atomic_parts if _is_factual(p.text)]

            # Step 4: Produce Claim objects
            claims: list[Claim] = []
            for i, part in enumerate(factual_parts):
                # Clamp span to valid bounds
                start = max(0, part.start)
                end = min(len(response_text), part.end)

                claims.append(Claim(
                    id=f"r{i + 1:03d}",
                    text=part.text.strip(),
                    source_span=(start, end),
                    source_text=response_text[start:end].strip(),
                ))

            logger.debug(
                "RuleDecomposer: %d sentences → %d atomic parts → %d claims",
                len(sentences),
                len(atomic_parts),
                len(claims),
            )
            return claims

        except Exception as e:
            raise DecompositionError(
                f"Rule-based decomposition failed: {e}"
            ) from e

    def is_available(self) -> bool:
        """Always True — no model required."""
        return True
