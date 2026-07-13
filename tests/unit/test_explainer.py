"""Unit tests for Phase 6 explainability (span mapping + evidence linking).
No real models required.
"""
# ruff: noqa: E501

from __future__ import annotations

import pytest

from veritascore.core.types import Claim, ClaimVerdict, Verdict, VerificationMode
from veritascore.explainer import EvidenceLinker, HighlightedSpan, SpanMapper


def make_claim(cid: str, text: str, start: int = 0, src: str | None = None) -> Claim:
    end = start + len(text)
    return Claim(id=cid, text=text, source_span=(start, end), source_text=src or text)


def make_verdict(
    cid: str,
    text: str,
    verdict: Verdict = Verdict.SUPPORTED,
    start: int = 0,
    evidence: str | None = "Some evidence.",
    reason: str = "Test reason.",
    nli_score: float | None = 0.85,
    retrieval_score: float | None = None,
    consistency_score: float | None = None,
    mode: VerificationMode = VerificationMode.GROUNDED,
) -> ClaimVerdict:
    claim = make_claim(cid, text, start)
    return ClaimVerdict(
        claim=claim,
        verdict=verdict,
        confidence=0.85,
        nli_score=nli_score,
        retrieval_score=retrieval_score,
        consistency_score=consistency_score,
        evidence=evidence,
        reason=reason,
        verification_mode=mode,
    )


RESPONSE = "The Eiffel Tower is 330 meters tall. It was completed in 1889."


class TestSpanMapper:
    @pytest.fixture
    def mapper(self) -> SpanMapper:
        return SpanMapper()

    def test_empty_verdicts_returns_empty(self, mapper: SpanMapper) -> None:
        assert mapper.map_spans(RESPONSE, []) == []

    def test_empty_response_returns_empty(self, mapper: SpanMapper) -> None:
        v = make_verdict("c1", "Some claim.")
        assert mapper.map_spans("", [v]) == []

    def test_exact_span_validated(self, mapper: SpanMapper) -> None:
        text = "The Eiffel Tower is 330 meters tall."
        v = make_verdict("c1", text, start=0)
        spans = mapper.map_spans(RESPONSE, [v])
        assert len(spans) == 1
        assert spans[0].text == text
        assert spans[0].claim_id == "c1"

    def test_span_refined_by_substring_search(self, mapper: SpanMapper) -> None:
        src = "1889"
        claim = Claim(id="c1", text="year", source_span=(0, 5), source_text=src)
        v = ClaimVerdict(
            claim=claim,
            verdict=Verdict.SUPPORTED,
            confidence=0.9,
            reason="test",
            verification_mode=VerificationMode.GROUNDED,
        )
        spans = mapper.map_spans(RESPONSE, [v])
        assert any(s.text == src for s in spans)

    def test_spans_sorted_by_start(self, mapper: SpanMapper) -> None:
        v1 = make_verdict("c1", "It was completed in 1889.", start=37)
        v2 = make_verdict("c2", "The Eiffel Tower is 330 meters tall.", start=0)
        spans = mapper.map_spans(RESPONSE, [v1, v2])
        assert spans[0].start <= spans[1].start

    def test_verdict_metadata_preserved(self, mapper: SpanMapper) -> None:
        v = make_verdict("c1", "The Eiffel Tower is 330 meters tall.", verdict=Verdict.CONTRADICTED)
        spans = mapper.map_spans(RESPONSE, [v])
        assert spans[0].verdict == Verdict.CONTRADICTED
        assert spans[0].confidence == pytest.approx(0.85)

    def test_zero_length_span_skipped(self, mapper: SpanMapper) -> None:
        claim = Claim(id="c1", text="", source_span=(5, 5), source_text="")
        v = ClaimVerdict(
            claim=claim,
            verdict=Verdict.SUPPORTED,
            confidence=0.8,
            reason="test",
            verification_mode=VerificationMode.GROUNDED,
        )
        assert mapper.map_spans(RESPONSE, [v]) == []


class TestSpanMapperOverlapResolution:
    @pytest.fixture
    def mapper(self) -> SpanMapper:
        return SpanMapper()

    def test_non_overlapping_kept(self, mapper: SpanMapper) -> None:
        s1 = HighlightedSpan(0, 10, "span one..", "c1", Verdict.SUPPORTED, 0.8)
        s2 = HighlightedSpan(10, 20, "span two..", "c2", Verdict.CONTRADICTED, 0.9)
        assert len(mapper._resolve_overlaps([s1, s2])) == 2

    def test_higher_severity_wins_overlap(self, mapper: SpanMapper) -> None:
        sup = HighlightedSpan(0, 20, "overlap text here..", "c1", Verdict.SUPPORTED, 0.8)
        con = HighlightedSpan(5, 25, "overlap text here..", "c2", Verdict.CONTRADICTED, 0.9)
        result = mapper._resolve_overlaps([sup, con])
        assert len(result) == 1
        assert result[0].verdict == Verdict.CONTRADICTED

    def test_equal_severity_keeps_earlier(self, mapper: SpanMapper) -> None:
        s1 = HighlightedSpan(0, 20, "span one overlap..", "c1", Verdict.SUPPORTED, 0.8)
        s2 = HighlightedSpan(5, 25, "span two overlap..", "c2", Verdict.SUPPORTED, 0.7)
        result = mapper._resolve_overlaps([s1, s2])
        assert len(result) == 1
        assert result[0].claim_id == "c1"

    def test_single_span_unchanged(self, mapper: SpanMapper) -> None:
        s = HighlightedSpan(0, 10, "text......", "c1", Verdict.SUPPORTED, 0.8)
        assert mapper._resolve_overlaps([s]) == [s]

    def test_empty_list(self, mapper: SpanMapper) -> None:
        assert mapper._resolve_overlaps([]) == []

    def test_three_way_overlap_highest_severity_wins(self, mapper: SpanMapper) -> None:
        supported = HighlightedSpan(0, 30, "x" * 30, "c1", Verdict.SUPPORTED, 0.8)
        unsupported = HighlightedSpan(5, 30, "y" * 25, "c2", Verdict.UNSUPPORTED, 0.7)
        contradicted = HighlightedSpan(10, 30, "z" * 20, "c3", Verdict.CONTRADICTED, 0.9)
        result = mapper._resolve_overlaps([supported, unsupported, contradicted])
        assert len(result) == 1
        assert result[0].verdict == Verdict.CONTRADICTED


class TestRenderAnnotatedText:
    @pytest.fixture
    def mapper(self) -> SpanMapper:
        return SpanMapper()

    def test_no_spans_returns_original(self, mapper: SpanMapper) -> None:
        assert mapper.render_annotated_text(RESPONSE, []) == RESPONSE

    def test_contradicted_annotated(self, mapper: SpanMapper) -> None:
        spans = [HighlightedSpan(0, 36, RESPONSE[:36], "c1", Verdict.CONTRADICTED, 0.9)]
        assert "[CONTRADICTED:" in mapper.render_annotated_text(RESPONSE, spans)

    def test_supported_not_annotated_by_default(self, mapper: SpanMapper) -> None:
        spans = [HighlightedSpan(0, 36, RESPONSE[:36], "c1", Verdict.SUPPORTED, 0.9)]
        result = mapper.render_annotated_text(RESPONSE, spans)
        assert "[SUPPORTED:" not in result

    def test_supported_annotated_when_flag_set(self, mapper: SpanMapper) -> None:
        spans = [HighlightedSpan(0, 36, RESPONSE[:36], "c1", Verdict.SUPPORTED, 0.9)]
        assert "[SUPPORTED:" in mapper.render_annotated_text(
            RESPONSE, spans, include_supported=True
        )

    def test_multiple_spans_all_annotated(self, mapper: SpanMapper) -> None:
        spans = [
            HighlightedSpan(0, 5, RESPONSE[:5], "c1", Verdict.CONTRADICTED, 0.9),
            HighlightedSpan(37, 41, RESPONSE[37:41], "c2", Verdict.UNSUPPORTED, 0.6),
        ]
        result = mapper.render_annotated_text(RESPONSE, spans)
        assert "[CONTRADICTED:" in result
        assert "[UNSUPPORTED:" in result

    def test_get_verdict_summary(self, mapper: SpanMapper) -> None:
        spans = [
            HighlightedSpan(0, 10, "x" * 10, "c1", Verdict.SUPPORTED, 0.9),
            HighlightedSpan(10, 20, "y" * 10, "c2", Verdict.CONTRADICTED, 0.7),
            HighlightedSpan(20, 30, "z" * 10, "c3", Verdict.UNSUPPORTED, 0.5),
        ]
        summary = mapper.get_verdict_summary(spans)
        assert summary == {"supported": 1, "contradicted": 1, "unsupported": 1}


class TestEvidenceLinker:
    @pytest.fixture
    def linker(self) -> EvidenceLinker:
        return EvidenceLinker()

    def test_build_chain_basic(self, linker: EvidenceLinker) -> None:
        v = make_verdict("c1", "Paris is the capital of France.", evidence="Paris is in France.")
        chains = linker.build_chains([v])
        assert len(chains) == 1
        chain = chains[0]
        assert chain.claim_id == "c1"
        assert chain.verdict == "supported"
        assert chain.confidence == pytest.approx(0.85)

    def test_build_chain_signals_populated(self, linker: EvidenceLinker) -> None:
        v = make_verdict("c1", "Claim.", nli_score=0.9, retrieval_score=0.7, consistency_score=0.8)
        chain = linker.build_chains([v])[0]
        assert chain.signals["nli_score"] == pytest.approx(0.9)
        assert chain.signals["retrieval_score"] == pytest.approx(0.7)
        assert chain.signals["consistency_score"] == pytest.approx(0.8)

    def test_missing_signals_are_none_in_chain(self, linker: EvidenceLinker) -> None:
        v = make_verdict("c1", "Claim.", nli_score=None, retrieval_score=None)
        chain = linker.build_chains([v])[0]
        assert chain.signals["nli_score"] is None
        assert chain.signals["retrieval_score"] is None

    def test_source_grounded_mode(self, linker: EvidenceLinker) -> None:
        v = make_verdict("c1", "Claim.", evidence="evidence", mode=VerificationMode.GROUNDED)
        chain = linker.build_chains([v])[0]
        assert chain.evidence_source == "provided context"

    def test_source_ungrounded_with_url(self, linker: EvidenceLinker) -> None:
        evidence = "Some text. [Source: https://example.com/page]"
        v = make_verdict("c1", "Claim.", evidence=evidence, mode=VerificationMode.UNGROUNDED)
        chain = linker.build_chains([v])[0]
        assert chain.evidence_source == "https://example.com/page"

    def test_source_ungrounded_no_url(self, linker: EvidenceLinker) -> None:
        v = make_verdict(
            "c1", "Claim.", evidence="evidence without url", mode=VerificationMode.UNGROUNDED
        )
        assert linker.build_chains([v])[0].evidence_source == "web search"

    def test_source_no_evidence(self, linker: EvidenceLinker) -> None:
        v = make_verdict("c1", "Claim.", evidence=None)
        assert linker.build_chains([v])[0].evidence_source is None

    def test_build_chains_empty(self, linker: EvidenceLinker) -> None:
        assert linker.build_chains([]) == []

    def test_validate_traceability_passes(self, linker: EvidenceLinker) -> None:
        v = make_verdict(
            "c1",
            "Claim.",
            verdict=Verdict.CONTRADICTED,
            evidence="Contradicting text.",
            reason="Context contradicts claim.",
        )
        assert linker.validate_traceability([v]) == []

    def test_validate_missing_reason_flagged(self, linker: EvidenceLinker) -> None:
        v = make_verdict("c1", "Claim.", reason="")
        issues = linker.validate_traceability([v])
        assert any("missing reason" in i for i in issues)

    def test_validate_contradicted_no_evidence_flagged(self, linker: EvidenceLinker) -> None:
        v = make_verdict(
            "c1",
            "Claim.",
            verdict=Verdict.CONTRADICTED,
            evidence=None,
            reason="Context says otherwise.",
        )
        issues = linker.validate_traceability([v])
        assert any("CONTRADICTED" in i and "evidence is missing" in i for i in issues)

    def test_validate_supported_no_evidence_ok(self, linker: EvidenceLinker) -> None:
        v = make_verdict(
            "c1", "Claim.", verdict=Verdict.SUPPORTED, evidence=None, reason="Context confirms."
        )
        assert linker.validate_traceability([v]) == []

    def test_format_chain_text(self, linker: EvidenceLinker) -> None:
        v = make_verdict(
            "c1",
            "Paris is in France.",
            verdict=Verdict.SUPPORTED,
            evidence="Paris is the capital of France.",
        )
        chain = linker.build_chains([v])[0]
        text = linker.format_chain_text(chain)
        assert "c1" in text
        assert "SUPPORTED" in text
        assert "Signals:" in text

    def test_format_chain_no_evidence(self, linker: EvidenceLinker) -> None:
        v = make_verdict(
            "c1", "Claim.", verdict=Verdict.UNSUPPORTED, evidence=None, reason="No evidence found."
        )
        chain = linker.build_chains([v])[0]
        assert "Evidence: None" in linker.format_chain_text(chain)

    def test_multiple_verdicts_same_count_chains(self, linker: EvidenceLinker) -> None:
        verdicts = [make_verdict(f"c{i}", f"Claim {i}.") for i in range(3)]
        chains = linker.build_chains(verdicts)
        assert len(chains) == 3
