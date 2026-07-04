"""Shared utilities for verifier modules.

Includes text preprocessing, sentence splitting, and the token-aware
context chunking algorithm used by NLIVerifier to handle documents that
exceed the NLI model's max sequence length.
"""

from __future__ import annotations

import re
from typing import Any, Protocol


def preprocess_text(text: str) -> str:
    """Normalize whitespace in text before feeding to an NLI model.

    Args:
        text: Raw input text.

    Returns:
        Text with collapsed whitespace and stripped ends.

    Example:
        >>> preprocess_text("  Hello   world  \\n")
        'Hello world'
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Common abbreviations that should not trigger a sentence split
_ABBREVS = ["Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.", "vs.", "etc.", "e.g.", "i.e."]


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, guarding against common abbreviations.

    Args:
        text: Input text (e.g. a context chunk).

    Returns:
        List of sentence strings (stripped, non-empty).

    Example:
        >>> split_into_sentences("Dr. Smith works here. He is a doctor.")
        ['Dr. Smith works here.', 'He is a doctor.']
    """
    if not text or not text.strip():
        return []

    temp = text
    for abbr in _ABBREVS:
        temp = temp.replace(abbr, abbr.replace(".", "<DOT>"))

    sentences = re.split(r"(?<=[.!?])\s+", temp)
    return [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]


class _Tokenizer(Protocol):
    """Structural type for the subset of tokenizer API this module needs."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...
    def decode(self, ids: list[int], skip_special_tokens: bool = ...) -> str: ...


def chunk_context(
    context: str,
    tokenizer: Any,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    reserved_tokens: int = 50,
) -> list[str]:
    """Split a context document into overlapping, token-bounded chunks.

    NLI models have a maximum sequence length covering BOTH premise and
    hypothesis. This function reserves `reserved_tokens` for the hypothesis
    (claim) + special tokens, and chunks the premise (context) to fit the
    remaining budget, with overlap to avoid losing claims that straddle a
    chunk boundary.

    Args:
        context: The full context/document text to chunk.
        tokenizer: A HuggingFace-style tokenizer with .encode()/.decode().
        max_tokens: Maximum total sequence length for the NLI model.
        overlap_tokens: Number of tokens to overlap between consecutive chunks.
        reserved_tokens: Tokens reserved for hypothesis + special tokens.

    Returns:
        List of chunk strings. Returns [context] unchanged if it already
        fits within budget.

    Raises:
        ValueError: If overlap_tokens >= the usable chunk budget (would
            cause an infinite loop / zero-progress stride).

    Example:
        >>> chunks = chunk_context(long_document, tokenizer, max_tokens=512)
        >>> all(len(tokenizer.encode(c)) <= 512 - 50 for c in chunks)
        True
    """
    if not context or not context.strip():
        return []

    budget = max_tokens - reserved_tokens
    if budget <= 0:
        raise ValueError(
            f"max_tokens ({max_tokens}) must exceed reserved_tokens ({reserved_tokens})"
        )
    if overlap_tokens >= budget:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be smaller than the "
            f"usable chunk budget ({budget}); otherwise chunking cannot progress."
        )

    tokens = tokenizer.encode(context, add_special_tokens=False)

    if len(tokens) <= budget:
        return [context.strip()]

    chunks: list[str] = []
    stride = budget - overlap_tokens

    for start in range(0, len(tokens), stride):
        chunk_tokens = tokens[start : start + budget]
        if not chunk_tokens:
            break
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunk_text = chunk_text.strip()
        if chunk_text:
            chunks.append(chunk_text)

        if start + budget >= len(tokens):
            break

    return chunks if chunks else [context.strip()]


def extract_evidence_snippet(
    chunk: str,
    claim_text: str,
    max_length: int = 300,
) -> str:
    """Extract the most relevant sentence from a context chunk as evidence.

    Uses simple word-overlap scoring (Jaccard-like, no embeddings) to find
    the sentence in `chunk` most relevant to `claim_text`.

    Args:
        chunk: The context chunk that was matched to the claim.
        claim_text: The claim text to find supporting evidence for.
        max_length: Maximum character length of the returned snippet.

    Returns:
        The best-matching sentence, truncated to max_length. Falls back to
        a truncated prefix of the chunk if no sentences are found.

    Example:
        >>> extract_evidence_snippet(
        ...     "Paris is in France. It has many museums.",
        ...     "Paris is the capital of France.",
        ... )
        'Paris is in France.'
    """
    if not chunk:
        return ""

    sentences = split_into_sentences(chunk)
    if not sentences:
        return chunk[:max_length]

    claim_words = set(re.findall(r"\b\w+\b", claim_text.lower()))
    if not claim_words:
        return sentences[0][:max_length]

    best_sentence = sentences[0]
    best_overlap = -1

    for sentence in sentences:
        sent_words = set(re.findall(r"\b\w+\b", sentence.lower()))
        overlap = len(claim_words & sent_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_sentence = sentence

    return best_sentence[:max_length]
