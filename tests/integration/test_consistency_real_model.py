"""Integration tests for SemanticConsistencyChecker using the REAL
sentence-transformers embedding model (default: all-MiniLM-L6-v2).

Marked `integration` and `slow`; excluded from `make test` by default.

Run explicitly with:
    pytest tests/integration/test_consistency_real_model.py -v -m integration
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from veritascore.core.types import Claim
from veritascore.verifier.consistency import SemanticConsistencyChecker

pytestmark = [pytest.mark.integration, pytest.mark.slow]

FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "consistency_samples.json"


@pytest.fixture(scope="module")
def checker() -> SemanticConsistencyChecker:
    """Module-scoped — load the real embedding model once, reuse across tests."""
    c = SemanticConsistencyChecker()
    yield c
    c.unload()


def _make_claim(text: str, idx: int) -> Claim:
    return Claim(id=f"c{idx}", text=text, source_span=(0, len(text)), source_text=text)


class TestRealModelLoading:
    def test_model_loads_quickly(self, checker: SemanticConsistencyChecker) -> None:
        """G4 criterion 4: embedding model loads in <30s.

        On Windows, loading sentence-transformers in a fresh Python process
        takes 10-15s because torch/transformers module import time alone is
        8-10s. The 30s ceiling catches real problems (e.g. re-downloading a
        model that should already be cached) without being a flaky hardware
        benchmark. The real G4 quality gates are accuracy criteria 1, 2, 6.
        """
        fresh = SemanticConsistencyChecker()
        t0 = time.time()
        fresh._load_model()
        elapsed = time.time() - t0
        assert elapsed < 30.0, f"Model load took {elapsed:.2f}s, expected <30s"
        fresh.unload()

    def test_is_available(self, checker: SemanticConsistencyChecker) -> None:
        assert checker.is_available() is True


class TestRealModelBasicScoring:
    def test_on_topic_high_score(self, checker: SemanticConsistencyChecker) -> None:
        claim = _make_claim("Paris is the capital of France.", 1)
        result = checker.check_consistency(
            [claim], "What is the capital of France?", "Paris is the capital of France."
        )
        assert result.claim_scores[claim.id] > 0.5

    def test_off_topic_low_score(self, checker: SemanticConsistencyChecker) -> None:
        claim = _make_claim("Python was created by Guido van Rossum.", 1)
        result = checker.check_consistency(
            [claim],
            "What is the capital of France?",
            "Python was created by Guido van Rossum.",
        )
        assert result.claim_scores[claim.id] < 0.3


class TestPerformance:
    def test_batch_encoding_50_claims_under_one_second(
        self, checker: SemanticConsistencyChecker
    ) -> None:
        """G4 criterion 5: batch encoding processes 50 claims in <1s."""
        checker._load_model()
        claims = [
            _make_claim(f"This is test claim number {i} about various topics.", i)
            for i in range(50)
        ]
        t0 = time.time()
        checker.check_consistency(claims, "A general query about topics.", "A response.")
        elapsed = time.time() - t0
        assert elapsed < 1.0, f"Batch encoding took {elapsed:.2f}s, expected <1s"


class TestCuratedFixtureFullEvaluation:
    """The actual Quality Gate G4 evaluation against the real model and
    the full curated fixture — see docs/phase4_evaluation_log.md for
    results recorded from a real hardware run."""

    def test_on_topic_claims_score_above_threshold_90pct(
        self, checker: SemanticConsistencyChecker
    ) -> None:
        """G4 criterion 1: on-topic claims score >0.5 on >90% of samples."""
        with open(FIXTURES_PATH) as f:
            data = json.load(f)

        total = 0
        passed = 0
        for sample in data["samples"]:
            query = sample["query"]
            for claim_text in sample["on_topic_claims"]:
                claim = _make_claim(claim_text, total)
                score = checker.score_claim(claim, query)
                total += 1
                if score > 0.5:
                    passed += 1

        pass_rate = passed / total
        assert pass_rate >= 0.90, f"On-topic pass rate {pass_rate:.1%} below 90% target"

    def test_off_topic_claims_score_below_threshold_80pct(
        self, checker: SemanticConsistencyChecker
    ) -> None:
        """G4 criterion 2: off-topic claims score <0.3 on >80% of samples."""
        with open(FIXTURES_PATH) as f:
            data = json.load(f)

        total = 0
        passed = 0
        for sample in data["samples"]:
            query = sample["query"]
            for claim_text in sample["off_topic_claims"]:
                claim = _make_claim(claim_text, total)
                score = checker.score_claim(claim, query)
                total += 1
                if score < 0.3:
                    passed += 1

        pass_rate = passed / total
        assert pass_rate > 0.80, f"Off-topic pass rate {pass_rate:.1%} below 80% target"

    def test_off_topic_claims_flagged_in_off_topic_list_80pct(
        self, checker: SemanticConsistencyChecker
    ) -> None:
        """G4 criterion 6: off_topic_claims list correctly flags >80% of
        planted off-topic items, evaluated via the full check_consistency()
        path (not just score_claim) to also exercise batch encoding."""
        with open(FIXTURES_PATH) as f:
            data = json.load(f)

        total_off_topic = 0
        correctly_flagged = 0

        for sample in data["samples"]:
            query = sample["query"]
            on_topic_claims = [_make_claim(t, i) for i, t in enumerate(sample["on_topic_claims"])]
            off_topic_claims = [
                _make_claim(t, 100 + i) for i, t in enumerate(sample["off_topic_claims"])
            ]
            all_claims = on_topic_claims + off_topic_claims
            response_text = " ".join(c.text for c in all_claims)

            result = checker.check_consistency(all_claims, query, response_text)

            for claim in off_topic_claims:
                total_off_topic += 1
                if claim.id in result.off_topic_claims:
                    correctly_flagged += 1

        detection_rate = correctly_flagged / total_off_topic
        assert detection_rate > 0.80, (
            f"Off-topic detection rate {detection_rate:.1%} below 80% target"
        )
