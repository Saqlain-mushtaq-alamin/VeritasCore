"""Span mapping utilities — map extracted claim texts to source offsets.

The LLM decomposer produces claim *text* that may differ from the exact
wording in the source response (pronoun resolution, paraphrasing). This
module handles the alignment from claim text → (start, end) character
offsets in the original response.

Strategies (in order of priority):
    1. Exact substring match (case-insensitive)
    2. Longest common subsequence sentence match (token overlap)
    3. Sliding window phrase match
    4. Fallback: whole-response span

All functions are pure (no model dependencies) and can be unit-tested
without any ML infrastructure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class SentenceSpan:
    """A sentence with its character offsets in the source text."""

    start: int
    end: int
    text: str

    def __len__(self) -> int:
        return self.end - self.start


# Abbreviations that should not trigger sentence splits
_ABBREVS = frozenset([
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr",
    "vs", "etc", "e.g", "i.e", "u.s", "u.k", "u.n",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug",
    "sep", "oct", "nov", "dec",
])

_SENTENCE_SPLIT = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])',
)


def split_into_sentences(text: str) -> list[SentenceSpan]:
    """Split text into sentences, preserving character offsets.

    Uses a regex-based splitter that avoids splitting on known abbreviations.
    Falls back to the whole text as a single sentence if no splits found.

    Args:
        text: Input text to split.

    Returns:
        List of SentenceSpan objects with start/end offsets.

    Example:
        >>> spans = split_into_sentences("Paris is in France. Rome is in Italy.")
        >>> [(s.start, s.end) for s in spans]
        [(0, 19), (20, 37)]
    """
    if not text.strip():
        return []

    # Temporarily mask known abbreviations to prevent false splits
    masked = text
    placeholder_map: dict[str, str] = {}
    for abbr in _ABBREVS:
        for variant in [abbr.capitalize() + ".", abbr.upper() + "."]:
            if variant in masked:
                placeholder = f"__ABBR_{hash(variant) & 0xFFFF}__"
                placeholder_map[placeholder] = variant
                masked = masked.replace(variant, placeholder)

    # Find split positions in the masked text
    splits: list[int] = [0]
    for m in _SENTENCE_SPLIT.finditer(masked):
        splits.append(m.end())
    splits.append(len(masked))

    # Reconstruct sentences with original offsets
    sentences: list[SentenceSpan] = []
    for i in range(len(splits) - 1):
        start = splits[i]
        end = splits[i + 1]
        sent_masked = masked[start:end].strip()

        # Restore abbreviations to find the real position
        sent_original = sent_masked
        for placeholder, original in placeholder_map.items():
            sent_original = sent_original.replace(placeholder, original)

        # Find this sentence in the original text
        true_start = text.find(sent_original[:20], max(0, start - 5))
        if true_start == -1:
            true_start = start
        true_end = true_start + len(sent_original)

        if sent_original.strip():
            sentences.append(SentenceSpan(
                start=true_start,
                end=min(true_end, len(text)),
                text=sent_original.strip(),
            ))

    # Fallback: treat whole text as one sentence
    if not sentences:
        sentences.append(SentenceSpan(0, len(text), text.strip()))

    return sentences


def _token_overlap(a: str, b: str) -> float:
    """Compute Jaccard-like token overlap between two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Overlap ratio in [0.0, 1.0].
    """
    tokens_a = set(re.findall(r'\b\w+\b', a.lower()))
    tokens_b = set(re.findall(r'\b\w+\b', b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def _sliding_window_match(
    claim: str,
    text: str,
    window_size: int = 8,
) -> tuple[int, int] | None:
    """Try to find a short phrase from claim as a substring of text.

    Takes the longest N-gram from the claim that appears in the text.

    Args:
        claim: Claim text to match.
        text: Source text to search in.
        window_size: Maximum number of words in the phrase to match.

    Returns:
        (start, end) if a match is found, None otherwise.
    """
    claim_words = claim.split()
    text_lower = text.lower()

    # Try decreasing window sizes
    for size in range(min(window_size, len(claim_words)), 2, -1):
        for i in range(len(claim_words) - size + 1):
            phrase = " ".join(claim_words[i: i + size]).lower()
            idx = text_lower.find(phrase)
            if idx != -1:
                return (idx, idx + len(phrase))

    return None


def find_best_span(
    claim_text: str,
    response_text: str,
    sentences: list[SentenceSpan] | None = None,
) -> tuple[tuple[int, int], str]:
    """Find the best character span in response_text for a given claim.

    Tries three strategies in order:
        1. Exact case-insensitive substring match
        2. Best-matching sentence by token overlap (threshold: 0.25)
        3. Sliding window phrase match
        4. Fallback: entire response span

    Args:
        claim_text: The atomic claim text (may be paraphrased).
        response_text: The original LLM response string.
        sentences: Pre-computed sentence splits (avoids re-splitting).
            If None, computed from response_text.

    Returns:
        Tuple of ((start, end), source_text_substring).

    Example:
        >>> span, src = find_best_span(
        ...     "The Eiffel Tower is 330 meters tall.",
        ...     "The Eiffel Tower stands 330 meters tall in Paris.",
        ... )
    """
    if not response_text:
        return (0, 0), ""

    if sentences is None:
        sentences = split_into_sentences(response_text)

    # ── Strategy 1: Exact substring (case-insensitive) ────────────────────────
    idx = response_text.lower().find(claim_text.lower())
    if idx != -1:
        end = idx + len(claim_text)
        return (idx, end), response_text[idx:end]

    # ── Strategy 2: Best sentence by token overlap ────────────────────────────
    if sentences:
        best_score = 0.25  # Minimum threshold to avoid garbage matches
        best_span: tuple[int, int] | None = None
        best_text: str | None = None

        for sent in sentences:
            score = _token_overlap(claim_text, sent.text)
            if score > best_score:
                best_score = score
                best_span = (sent.start, sent.end)
                best_text = sent.text

        if best_span is not None and best_text is not None:
            return best_span, best_text

    # ── Strategy 3: Sliding window phrase match ───────────────────────────────
    window_match = _sliding_window_match(claim_text, response_text)
    if window_match is not None:
        start, end = window_match
        return (start, end), response_text[start:end]

    # ── Strategy 4: Fallback — whole response ─────────────────────────────────
    return (0, len(response_text)), response_text


def map_claims_to_spans(
    claim_texts: list[str],
    response_text: str,
) -> list[tuple[tuple[int, int], str]]:
    """Map a list of claim texts to spans in the response.

    Pre-computes sentence splits once for efficiency, then maps each claim.

    Args:
        claim_texts: List of atomic claim strings.
        response_text: The original LLM response.

    Returns:
        List of ((start, end), source_text) tuples, one per claim.

    Example:
        >>> spans = map_claims_to_spans(
        ...     ["Paris is in France.", "Rome is in Italy."],
        ...     "Paris is in France. Rome is in Italy.",
        ... )
    """
    sentences = split_into_sentences(response_text)
    return [
        find_best_span(claim, response_text, sentences)
        for claim in claim_texts
    ]
