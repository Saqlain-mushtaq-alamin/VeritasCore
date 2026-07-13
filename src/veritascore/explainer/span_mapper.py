"""Map claims back to exact text spans with highlighting metadata."""

from __future__ import annotations

from dataclasses import dataclass

from veritascore.core.types import ClaimVerdict, Verdict

_SEVERITY: dict[Verdict, int] = {
    Verdict.SUPPORTED: 1,
    Verdict.UNSUPPORTED: 2,
    Verdict.CONTRADICTED: 3,
}


@dataclass
class HighlightedSpan:
    """A text span with verdict highlighting metadata.

    Attributes:
        start: Inclusive start character offset in the response text.
        end: Exclusive end character offset.
        text: The exact text slice response_text[start:end].
        claim_id: ID of the Claim this span corresponds to.
        verdict: The verdict for this claim.
        confidence: Calibrated confidence in [0, 1].
    """

    start: int
    end: int
    text: str
    claim_id: str
    verdict: Verdict
    confidence: float


class SpanMapper:
    """Map claim verdicts to highlighted character spans in the original text.

    Example:
        >>> mapper = SpanMapper()
        >>> spans = mapper.map_spans(response_text, verdicts)
        >>> annotated = mapper.render_annotated_text(response_text, spans)
    """

    def map_spans(
        self,
        response_text: str,
        verdicts: list[ClaimVerdict],
    ) -> list[HighlightedSpan]:
        """Create highlighted spans for all claim verdicts.

        Args:
            response_text: The full original LLM response string.
            verdicts: List of ClaimVerdict objects from the pipeline.

        Returns:
            Non-overlapping spans sorted by start position.
        """
        if not response_text or not verdicts:
            return []

        spans: list[HighlightedSpan] = []
        for v in verdicts:
            start, end = self._refine_span(response_text, v.claim.source_span, v.claim.source_text)
            if start >= end:
                continue
            spans.append(
                HighlightedSpan(
                    start=start,
                    end=end,
                    text=response_text[start:end],
                    claim_id=v.claim.id,
                    verdict=v.verdict,
                    confidence=v.confidence,
                )
            )

        spans = self._resolve_overlaps(spans)
        return sorted(spans, key=lambda s: s.start)

    def _refine_span(
        self,
        text: str,
        span: tuple[int, int],
        source_text: str,
    ) -> tuple[int, int]:
        """Refine a stored span to the exact source text location.

        Tries three strategies in order:
            1. Validate the stored span directly.
            2. Case-insensitive substring search.
            3. Return clamped original span.
        """
        start, end = span
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))

        if source_text and text[start:end].strip() == source_text.strip():
            return start, end

        needle = source_text.strip()
        if needle:
            idx = text.lower().find(needle.lower())
            if idx != -1:
                return idx, idx + len(needle)

        return start, end

    def _resolve_overlaps(self, spans: list[HighlightedSpan]) -> list[HighlightedSpan]:
        """Remove overlapping spans, keeping the higher-severity verdict.

        When two spans overlap, the one with the higher severity (CONTRADICTED
        > UNSUPPORTED > SUPPORTED) wins. Ties go to the span starting earlier.

        Args:
            spans: Arbitrary list of HighlightedSpan objects.

        Returns:
            Non-overlapping subset, may be shorter than input.
        """
        if len(spans) <= 1:
            return list(spans)

        ordered = sorted(
            spans,
            key=lambda s: (s.start, -_SEVERITY.get(s.verdict, 0)),
        )

        result: list[HighlightedSpan] = [ordered[0]]

        for current in ordered[1:]:
            prev = result[-1]
            if current.start >= prev.end:
                result.append(current)
            else:
                prev_sev = _SEVERITY.get(prev.verdict, 0)
                curr_sev = _SEVERITY.get(current.verdict, 0)
                if curr_sev > prev_sev:
                    result[-1] = current

        return result

    def render_annotated_text(
        self,
        response_text: str,
        spans: list[HighlightedSpan],
        include_supported: bool = False,
    ) -> str:
        """Render response text with inline verdict annotations.

        Inserts in reverse order to preserve earlier offsets.
        Format: Normal text [CONTRADICTED: "flagged span text"] more text.

        Args:
            response_text: The full original LLM response.
            spans: Pre-computed spans from map_spans(), sorted by start.
            include_supported: If True, also annotate SUPPORTED claims.

        Returns:
            Annotated string, or response_text unchanged if spans is empty.
        """
        if not spans:
            return response_text

        to_annotate = [s for s in spans if include_supported or s.verdict != Verdict.SUPPORTED]
        to_annotate_sorted = sorted(to_annotate, key=lambda s: s.start, reverse=True)

        result = response_text
        for span in to_annotate_sorted:
            label = span.verdict.value.upper()
            annotation = f'[{label}: "{span.text}"]'
            result = result[: span.start] + annotation + result[span.end :]

        return result

    def get_verdict_summary(self, spans: list[HighlightedSpan]) -> dict[str, int]:
        """Count spans by verdict type.

        Returns:
            Dict with keys 'supported', 'contradicted', 'unsupported'.
        """
        return {v.value: sum(1 for s in spans if s.verdict == v) for v in Verdict}
