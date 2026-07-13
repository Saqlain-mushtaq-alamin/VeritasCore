"""Unit tests for the claim decomposition module (Phase 1).

Covers:
    - RuleDecomposer (full coverage, no GPU needed)
    - Span mapping utilities (span_mapper.py)
    - Prompt template construction (prompts.py)
    - BaseDecomposer abstract contract

LLMDecomposer tests requiring an actual model load live in
tests/integration/test_decomposer_llm.py (marked as integration/slow).
"""

from __future__ import annotations

import pytest

from veritascore.core.exceptions import DecompositionError
from veritascore.core.types import Claim
from veritascore.decomposer.base import BaseDecomposer
from veritascore.decomposer.prompts import (
    DECOMPOSITION_VERSION,
    build_decomposition_prompt,
    get_system_prompt,
    list_versions,
)
from veritascore.decomposer.rule_decomposer import RuleDecomposer
from veritascore.decomposer.span_mapper import (
    find_best_span,
    map_claims_to_spans,
    split_into_sentences,
)

# ── BaseDecomposer Contract ───────────────────────────────────────────────────


class TestBaseDecomposer:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseDecomposer()  # type: ignore[abstract]

    def test_subclass_must_implement_decompose(self) -> None:
        class Incomplete(BaseDecomposer):
            def is_available(self) -> bool:
                return True

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_must_implement_is_available(self) -> None:
        class Incomplete(BaseDecomposer):
            def decompose(self, response_text: str, query: str | None = None) -> list[Claim]:
                return []

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_decompose_batch_default_impl(self) -> None:
        decomposer = RuleDecomposer()
        responses = ["Paris is the capital of France.", "Rome is the capital of Italy."]
        results = decomposer.decompose_batch(responses)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_decompose_batch_with_queries(self) -> None:
        decomposer = RuleDecomposer()
        responses = ["Fact one is true.", "Fact two is also true."]
        queries = ["Query A?", "Query B?"]
        results = decomposer.decompose_batch(responses, queries)
        assert len(results) == 2


# ── RuleDecomposer ────────────────────────────────────────────────────────────


class TestRuleDecomposer:
    """Tests for the rule-based fallback decomposer."""

    @pytest.fixture
    def decomposer(self) -> RuleDecomposer:
        return RuleDecomposer()

    def test_empty_input(self, decomposer: RuleDecomposer) -> None:
        assert decomposer.decompose("") == []
        assert decomposer.decompose("   ") == []
        assert decomposer.decompose("\n\t  \n") == []

    def test_single_sentence(self, decomposer: RuleDecomposer) -> None:
        text = "The Eiffel Tower was built in 1889."
        claims = decomposer.decompose(text)
        assert len(claims) >= 1
        assert all(isinstance(c, Claim) for c in claims)

    def test_multiple_sentences(self, decomposer: RuleDecomposer) -> None:
        text = (
            "Paris is the capital of France. "
            "It has a population of 2.1 million. "
            "The Seine river flows through the city."
        )
        claims = decomposer.decompose(text)
        assert len(claims) >= 3

    def test_filters_questions(self, decomposer: RuleDecomposer) -> None:
        text = "What is the capital of France? Paris is the capital of France."
        claims = decomposer.decompose(text)
        claim_texts = [c.text.lower() for c in claims]
        assert not any("what is" in t for t in claim_texts)
        assert any("paris" in t for t in claim_texts)

    def test_filters_opinions(self, decomposer: RuleDecomposer) -> None:
        text = "I think Python is a great language. Python was created by Guido van Rossum in 1991."
        claims = decomposer.decompose(text)
        claim_texts = [c.text.lower() for c in claims]
        assert not any("i think" in t for t in claim_texts)
        assert any("guido" in t for t in claim_texts)

    def test_filters_commands(self, decomposer: RuleDecomposer) -> None:
        text = "Please remember to water the plants. The plant is a ficus tree."
        claims = decomposer.decompose(text)
        claim_texts = [c.text.lower() for c in claims]
        assert not any("please remember" in t for t in claim_texts)

    def test_compound_splitting(self, decomposer: RuleDecomposer) -> None:
        text = "Einstein was born in Germany, and he later moved to the United States."
        claims = decomposer.decompose(text)
        assert len(claims) >= 2

    def test_compound_splitting_short_clauses_not_split(self, decomposer: RuleDecomposer) -> None:
        """Short clauses like 'cats and dogs' should not be split."""
        text = "I like cats and dogs as pets, which is common."
        claims = decomposer.decompose(text)
        # Should not produce a nonsensical split on "and dogs"
        for c in claims:
            assert len(c.text) > 5

    def test_span_mapping_within_bounds(self, decomposer: RuleDecomposer) -> None:
        text = "The Earth orbits the Sun. Water boils at 100 degrees Celsius."
        claims = decomposer.decompose(text)
        for claim in claims:
            assert claim.source_span[0] >= 0
            assert claim.source_span[1] <= len(text)
            assert claim.source_span[0] < claim.source_span[1]

    def test_claim_ids_unique(self, decomposer: RuleDecomposer) -> None:
        text = "Fact one is true. Fact two is also true. Fact three is true too."
        claims = decomposer.decompose(text)
        ids = [c.id for c in claims]
        assert len(ids) == len(set(ids))

    def test_is_available(self, decomposer: RuleDecomposer) -> None:
        assert decomposer.is_available() is True

    def test_single_word_input(self, decomposer: RuleDecomposer) -> None:
        # Should not crash; likely filtered as too short
        claims = decomposer.decompose("Hello")
        assert isinstance(claims, list)

    def test_very_long_input(self, decomposer: RuleDecomposer) -> None:
        text = "The sky is blue. " * 200  # 200 repeated sentences
        claims = decomposer.decompose(text)
        assert isinstance(claims, list)
        assert len(claims) > 0

    def test_numbers_and_dates_preserved(self, decomposer: RuleDecomposer) -> None:
        text = "The Eiffel Tower is 330 meters tall and was completed in 1889."
        claims = decomposer.decompose(text)
        all_text = " ".join(c.text for c in claims)
        assert "330" in all_text
        assert "1889" in all_text

    def test_abbreviations_not_falsely_split(self, decomposer: RuleDecomposer) -> None:
        text = "Dr. Smith works at the U.S. embassy. He has a PhD in chemistry."
        claims = decomposer.decompose(text)
        # Should produce sentences, not split mid-abbreviation
        assert all(len(c.text) > 5 for c in claims)

    def test_source_text_matches_span(self, decomposer: RuleDecomposer) -> None:
        text = "Mount Everest is the tallest mountain on Earth."
        claims = decomposer.decompose(text)
        for claim in claims:
            start, end = claim.source_span
            assert claim.source_text == text[start:end].strip()

    def test_decompose_raises_decomposition_error_on_internal_failure(
        self, decomposer: RuleDecomposer, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken_split(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(
            "veritascore.decomposer.rule_decomposer.split_into_sentences", broken_split
        )
        with pytest.raises(DecompositionError):
            decomposer.decompose("Some text that will fail.")


# ── Span Mapper ───────────────────────────────────────────────────────────────


class TestSplitIntoSentences:
    def test_empty_text(self) -> None:
        assert split_into_sentences("") == []
        assert split_into_sentences("   ") == []

    def test_single_sentence(self) -> None:
        spans = split_into_sentences("Paris is the capital of France.")
        assert len(spans) == 1
        assert spans[0].text == "Paris is the capital of France."

    def test_multiple_sentences(self) -> None:
        text = "Paris is in France. Rome is in Italy. Berlin is in Germany."
        spans = split_into_sentences(text)
        assert len(spans) == 3

    def test_offsets_within_bounds(self) -> None:
        text = "Fact one. Fact two. Fact three."
        spans = split_into_sentences(text)
        for span in spans:
            assert 0 <= span.start < span.end <= len(text)

    def test_abbreviation_handling(self) -> None:
        text = "Dr. Smith is a doctor. He works at the hospital."
        spans = split_into_sentences(text)
        # Should split into 2 sentences, not 3 (not splitting on "Dr.")
        assert len(spans) == 2

    def test_no_terminal_punctuation(self) -> None:
        text = "This text has no ending punctuation"
        spans = split_into_sentences(text)
        assert len(spans) == 1
        assert spans[0].text == text


class TestFindBestSpan:
    def test_exact_match(self) -> None:
        response = "The Eiffel Tower is 330 meters tall and located in Paris."
        claim = "The Eiffel Tower is 330 meters tall"
        span, text = find_best_span(claim, response)
        assert response[span[0] : span[1]].lower() == claim.lower()

    def test_case_insensitive_match(self) -> None:
        response = "the eiffel tower is in paris."
        claim = "The Eiffel Tower is in Paris."
        span, text = find_best_span(claim, response)
        assert span[0] == 0

    def test_paraphrased_claim_uses_sentence_overlap(self) -> None:
        response = "Marie Curie discovered radium and polonium in 1898."
        claim = "Marie Curie discovered radium."  # Subset paraphrase
        span, text = find_best_span(claim, response)
        assert "radium" in text.lower() or "curie" in text.lower()

    def test_no_match_falls_back_to_full_response(self) -> None:
        response = "Cats are mammals."
        claim = "Quantum computers use qubits for computation."
        span, text = find_best_span(claim, response)
        # Should fall back gracefully (not crash), span within bounds
        assert 0 <= span[0] <= span[1] <= len(response)

    def test_empty_response(self) -> None:
        span, text = find_best_span("Some claim", "")
        assert span == (0, 0)
        assert text == ""


class TestMapClaimsToSpans:
    def test_multiple_claims(self) -> None:
        response = "Paris is in France. Rome is in Italy."
        claims = ["Paris is in France.", "Rome is in Italy."]
        results = map_claims_to_spans(claims, response)
        assert len(results) == 2
        for span, _text in results:
            assert span[0] >= 0
            assert span[1] <= len(response)

    def test_empty_claims_list(self) -> None:
        results = map_claims_to_spans([], "Some response text.")
        assert results == []


# ── Prompts ───────────────────────────────────────────────────────────────────


class TestPrompts:
    def test_prompt_without_query(self) -> None:
        prompt = build_decomposition_prompt("Some response text.")
        assert "Some response text." in prompt
        assert "Context —" not in prompt

    def test_prompt_with_query(self) -> None:
        prompt = build_decomposition_prompt("Some response.", query="What is X?")
        assert "Some response." in prompt
        assert "What is X?" in prompt
        assert "Context —" in prompt

    def test_get_system_prompt_default_version(self) -> None:
        prompt = get_system_prompt()
        assert len(prompt) > 100
        assert "atomic" in prompt.lower()

    def test_get_system_prompt_specific_version(self) -> None:
        prompt_v1 = get_system_prompt("v1")
        prompt_v2 = get_system_prompt("v2")
        assert prompt_v1 != prompt_v2

    def test_unknown_version_raises(self) -> None:
        with pytest.raises(KeyError):
            get_system_prompt("v999")
        with pytest.raises(KeyError):
            build_decomposition_prompt("text", version="v999")

    def test_list_versions(self) -> None:
        versions = list_versions()
        assert "v1" in versions
        assert "v2" in versions
        assert DECOMPOSITION_VERSION in versions

    def test_prompt_strips_response_text(self) -> None:
        prompt = build_decomposition_prompt("   text with whitespace   ")
        assert "text with whitespace" in prompt
