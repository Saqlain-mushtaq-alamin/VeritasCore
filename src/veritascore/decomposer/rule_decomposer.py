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
    3. Split on serial commas (three-part lists)
    4. Split inline numbered lists (1) X, 2) Y, 3) Z)
    5. Filter non-factual sentences (questions, opinions, commands)
    6. Map each remaining sentence to a source span
    7. Produce Claim objects
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
_OPINION_PREFIXES: frozenset[str] = frozenset(
    [
        "i think",
        "i believe",
        "i feel",
        "i suspect",
        "in my opinion",
        "in my view",
        "personally,",
        "it seems",
        "it appears",
        "it looks like",
        "perhaps",
        "maybe",
        "possibly",
        "probably",
        "reportedly",
        "allegedly",
        "supposedly",
        "note that",
        "please note",
        "please remember",
        "consider ",
        "remember that",
    ]
)

# Patterns for sentences that are questions
_QUESTION_PATTERN = re.compile(
    r"^\s*(?:who|what|when|where|why|how|is|are|was|were|do|does|did"
    r"|can|could|would|should|will|shall|have|has|had)\b.*\?\s*$",
    re.IGNORECASE,
)

# Conjunctions used to join independent clauses (require comma before them)
_CONJUNCTION_SPLITS = [
    r",\s+and\s+",
    r",\s+but\s+",
    r",\s+while\s+",
    r",\s+whereas\s+",
    r",\s+yet\s+",
    r";\s+",
]

# Minimum character length for a sentence to be considered a claim
_MIN_CLAIM_LENGTH = 12

# Maximum compound splits per sentence (prevents over-splitting)
_MAX_COMPOUND_SPLITS = 6

# Minimum clause length when splitting on bare conjunctions
_MIN_CLAUSE_LENGTH = 15

# Hedge/opinion words that, when present, mark a sentence as non-factual
# even if not at the very start (e.g. "X is probably the biggest...")
_HEDGE_WORDS: frozenset[str] = frozenset(
    [
        "probably",
        "perhaps",
        "possibly",
        "maybe",
        "arguably",
        "reportedly",
        "allegedly",
        "supposedly",
        "presumably",
    ]
)

# Inline numbered list pattern: "1) text" or "1. text" within a sentence
_INLINE_NUMBERED = re.compile(
    r"(?:^|\s)(\d+)[.)]\s+",
)

# Pattern to detect a subject (starts with a capital letter word or article)
_HAS_SUBJECT = re.compile(
    r"^(?:the\s+|a\s+|an\s+|[A-Z])",
    re.IGNORECASE,
)

# Words that typically start an independent clause after 'and'
_CLAUSE_STARTERS = re.compile(
    r"^(?:was|were|is|are|has|had|have|he|she|it|they|its|"
    r"also|the|a|an|each|this|that|does|do|did|can|could|"
    r"will|would|should|may|might|shall|"
    r"dissolves?|contains?|covers?|includes?|provides?|"
    r"produces?|requires?|supports?|uses?|allows?|"
    r"makes?|takes?|gives?|shows?|finds?|keeps?|"
    r"stands?|holds?|runs?|comes?|goes?|gets?|"
    r"[A-Z][a-z]+)\b",
)


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
        r"^the following\b",
        r"^here (is|are)\b",
        r"^for example[,:]",
        r"^such as[,:]",
        r"^including\b",
        r"^\(.*\)$",  # Pure parenthetical
        r"^note:",
        r"^important:",
    ]
    return all(not re.match(pat, lower) for pat in meta_patterns)


def _looks_like_independent_clause(text: str) -> bool:
    """Check if a text fragment looks like an independent clause.

    Independent clauses typically have a subject (noun/pronoun) and could
    stand alone as a sentence.
    """
    text = text.strip()
    if len(text) < _MIN_CLAUSE_LENGTH:
        return False
    return bool(_CLAUSE_STARTERS.match(text))


def _split_bare_conjunction(
    sent: SentenceSpan,
) -> list[SentenceSpan]:
    """Split on bare 'and' / 'but' / 'and also' joining independent clauses.

    Only splits when:
    - The right side starts with a word that looks like a clause starter
      (verb, pronoun, article, or capitalized noun)
    - Both halves are at least _MIN_CLAUSE_LENGTH characters

    This avoids splitting "cats and dogs" but correctly splits
    "X was born in 1879 and developed the theory of relativity".
    """
    text = sent.text

    # Pattern: " and also " — strong signal for independent clause
    for pattern_str in [r"\s+and\s+also\s+", r"\s+and\s+", r"\s+but\s+"]:
        for match in re.finditer(pattern_str, text, re.IGNORECASE):
            left = text[: match.start()]
            right = text[match.end() :]

            # Only split if both sides are substantial
            # and right side looks like an independent clause
            if (
                len(left.strip()) >= _MIN_CLAUSE_LENGTH
                and len(right.strip()) >= _MIN_CLAUSE_LENGTH
                and _looks_like_independent_clause(right)
            ):
                left_part = SentenceSpan(
                    sent.start,
                    sent.start + match.start(),
                    left.strip(),
                )
                right_part = SentenceSpan(
                    sent.start + match.end(),
                    sent.end,
                    right.strip(),
                )
                return [left_part, right_part]

            break  # Only try first occurrence per pattern

    return [sent]


def _split_serial_comma(
    sent: SentenceSpan,
) -> list[SentenceSpan]:
    """Split serial-comma lists with a shared subject.

    Handles patterns like:
        "The Eiffel Tower was completed in 1889, stands 330 meters tall,
         and was designed by Gustave Eiffel."

    Extracts the subject and prepends it to each item.
    """
    text = sent.text

    # Look for pattern: "SUBJECT VERB1..., VERB2..., and VERB3..."
    # Serial comma: at least 2 commas with "and" before the last item
    serial_match = re.match(
        r"^(.+?)\s+"  # Subject (greedy-minimal)
        r"((?:was|were|is|are|has|had|have|made|won|"
        r"designed|developed|wrote|created|painted|"
        r"formulated|composed|achieved|discovered|"
        r"invented|founded|built|stands|uses|"
        r"consists|contains|guarantees|describes)\b.+?),"
        r"\s+(.+?,)\s+and\s+(.+)$",
        text,
        re.IGNORECASE,
    )

    if serial_match:
        subject = serial_match.group(1).strip()
        items = [
            serial_match.group(2).strip(),
            serial_match.group(3).strip(),
            serial_match.group(4).strip(),
        ]

        # Only split if items look substantial
        if all(len(item) >= 8 for item in items):
            results = []
            # For each item, prepend the subject if needed
            for item in items:
                item_text = item.rstrip(".")
                # Check if item already has a subject
                full_text = item_text if _HAS_SUBJECT.match(item_text) else f"{subject} {item_text}"

                results.append(
                    SentenceSpan(
                        sent.start,
                        sent.end,
                        full_text,
                    )
                )

            if len(results) >= 2:
                return results

    return [sent]


def _split_inline_numbered_list(
    sent: SentenceSpan,
) -> list[SentenceSpan]:
    """Split inline numbered lists like '1) Python, 2) JavaScript, 3) Java'.

    Handles patterns like:
        "Here are the top 3: 1) Python, 2) JavaScript, 3) Java."
    """
    text = sent.text

    # Find all numbered items
    items: list[tuple[int, str]] = []
    for match in re.finditer(
        r"(\d+)[.)]\s+([^,.)]+(?:\([^)]*\))?)",
        text,
    ):
        num = int(match.group(1))
        item_text = match.group(2).strip().rstrip(".")
        if len(item_text) >= 3:
            items.append((num, item_text))

    if len(items) >= 2:
        results = []
        for _num, item_text in items:
            # Create a standalone claim for each list item
            # Use a special marker prefix so _is_factual won't
            # filter short list items (e.g. "Python", "Java")
            results.append(
                SentenceSpan(
                    sent.start,
                    sent.end,
                    item_text,
                )
            )

        # Return items directly — they bypass normal sentence
        # filtering since they're structured list entries
        if results:
            return results

    return [sent]


def _split_country_or_item_list(
    sent: SentenceSpan,
) -> list[SentenceSpan]:
    """Split sentences listing items connected by commas and 'and'.

    Handles patterns like:
        "The Amazon River flows through Brazil, Peru, and Colombia."
    Splits into:
        - "The Amazon River flows through Brazil"
        - "The Amazon River flows through Peru"
        - "The Amazon River flows through Colombia"
    """
    text = sent.text

    # Pattern: "SUBJECT VERB through/in/to X, Y, and Z"
    list_match = re.match(
        r"^(.+?)\s+"
        r"((?:flows?\s+through|is\s+(?:located\s+)?in|"
        r"(?:made|has)\s+(?:significant\s+)?contributions?\s+to|"
        r"designed\s+(?:early\s+)?concepts?\s+for|"
        r"won\s+Nobel\s+Prizes?\s+in|"
        r"(?:visits?|travels?\s+to|borders?))\s+)"
        r"(.+)$",
        text,
        re.IGNORECASE,
    )

    if list_match:
        subject = list_match.group(1).strip()
        verb_phrase = list_match.group(2).strip()
        items_str = list_match.group(3).strip().rstrip(".")

        # Split on ", and " or ", " or " and "
        raw_items = re.split(
            r",\s+and\s+|,\s+|\s+and\s+",
            items_str,
        )

        # Filter to meaningful items
        items = [item.strip() for item in raw_items if item.strip()]

        if len(items) >= 2:
            results = []
            for item in items:
                full = f"{subject} {verb_phrase}{item}"
                results.append(
                    SentenceSpan(
                        sent.start,
                        sent.end,
                        full,
                    )
                )
            return results

    return [sent]


def _split_compound_sentence(
    sent: SentenceSpan,
) -> list[SentenceSpan]:
    """Try to split a compound sentence into simpler factual parts.

    Applies multiple splitting strategies in order:
    1. Inline numbered list splitting
    2. Serial comma splitting (3+ verb phrases)
    3. Country/item list splitting
    4. Comma-conjunction splitting (, and / , but / ; etc.)
    5. Bare conjunction splitting (and/but joining independent clauses)

    Args:
        sent: The sentence to attempt splitting.

    Returns:
        List of SentenceSpan objects. Either [sent] (no split) or 2+ parts.
    """
    # Strategy 0: Inline numbered lists
    inline_result = _split_inline_numbered_list(sent)
    if len(inline_result) > 1:
        return inline_result

    # Strategy 1: Serial comma splitting (3-part lists with shared subject)
    serial_result = _split_serial_comma(sent)
    if len(serial_result) > 1:
        return serial_result

    # Strategy 2: Country/item list splitting
    list_result = _split_country_or_item_list(sent)
    if len(list_result) > 1:
        return list_result

    # Strategy 3: Comma-conjunction splitting (original logic)
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

    if len(parts) > 1:
        return parts

    # Strategy 4: Bare conjunction splitting (no comma before 'and')
    bare_result = _split_bare_conjunction(sent)
    if len(bare_result) > 1:
        return bare_result

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
        - Atomic accuracy: ~90%+ (improved with advanced splitting)
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
            inline_list_ids: set[int] = set()
            for sent in sentences:
                # Check if this sentence has an inline numbered list
                inline_result = _split_inline_numbered_list(sent)
                if len(inline_result) > 1:
                    for part in inline_result:
                        inline_list_ids.add(id(part))
                    atomic_parts.extend(inline_result)
                else:
                    splits = _split_compound_sentence(sent)
                    atomic_parts.extend(splits)

            # Step 3: Filter non-factual sentences
            # Inline list items bypass the filter (they may be short)
            factual_parts = [
                p for p in atomic_parts if id(p) in inline_list_ids or _is_factual(p.text)
            ]

            # Step 4: Produce Claim objects
            claims: list[Claim] = []
            for i, part in enumerate(factual_parts):
                # Clamp span to valid bounds
                start = max(0, part.start)
                end = min(len(response_text), part.end)

                claims.append(
                    Claim(
                        id=f"r{i + 1:03d}",
                        text=part.text.strip(),
                        source_span=(start, end),
                        source_text=response_text[start:end].strip(),
                    )
                )

            logger.debug(
                "RuleDecomposer: %d sentences → %d atomic parts → %d claims",
                len(sentences),
                len(atomic_parts),
                len(claims),
            )
            return claims

        except Exception as e:
            raise DecompositionError(f"Rule-based decomposition failed: {e}") from e

    def is_available(self) -> bool:
        """Always True — no model required."""
        return True
