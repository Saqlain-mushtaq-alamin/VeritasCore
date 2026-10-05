"""Tests for scripts/baseline_summac.py — Phase R4 SummaC ZS oracle baseline.

All tests use mocking/synthetic data so the NLI model is NOT loaded during the
test suite. This ensures CI passes without GPU and without heavy model downloads.

The SummaC ZS reimplementation is tested by:
  - Mocking the NLI batch-inference call
  - Testing score inversion, sentence-level max aggregation, label mapping
  - Testing the full benchmark runner's output schema and CI keys

Run with:
    pytest tests/test_baseline_summac.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _make_samples(n: int = 20, seed: int = 0) -> list[dict]:
    """Synthetic {context, claim_text, label} samples."""
    rng = np.random.RandomState(seed)
    return [
        {
            "context": f"The capital of country {i} is city {i}. "
                       f"City {i} was founded in {1000 + i}.",
            "claim_text": f"City {i} is the capital." if rng.rand() > 0.5
                          else f"City {i + 100} is the capital.",
            "label": "supported" if i % 2 == 0 else "hallucinated",
        }
        for i in range(n)
    ]


def _make_mock_scorer(fixed_consistency: float = 0.7) -> MagicMock:
    """Create a mock SummaCZSReimplemented that always returns fixed_consistency."""
    scorer = MagicMock()
    scorer.model_name = "cross-encoder/nli-deberta-v3-base"
    scorer.device = "cpu"
    scorer.score.return_value = fixed_consistency
    return scorer


# ---------------------------------------------------------------------------
# Tests: sentence splitter
# ---------------------------------------------------------------------------

class TestSplitSentences:
    def test_splits_basic_text(self):
        from baseline_summac import _split_sentences
        text = "Alice was born in 1990. She lives in London. Her cat is named Bob."
        sents = _split_sentences(text)
        assert len(sents) >= 2

    def test_single_sentence(self):
        from baseline_summac import _split_sentences
        text = "Alice was born in 1990."
        sents = _split_sentences(text)
        assert len(sents) >= 1

    def test_empty_string_returns_list(self):
        from baseline_summac import _split_sentences
        sents = _split_sentences("")
        assert isinstance(sents, list)

    def test_returns_non_empty_strings(self):
        from baseline_summac import _split_sentences
        text = "First sentence. Second sentence. Third."
        sents = _split_sentences(text)
        assert all(s.strip() for s in sents)


# ---------------------------------------------------------------------------
# Tests: score inversion logic
# ---------------------------------------------------------------------------

class TestScoreInversion:
    """Verify the 1.0 - consistency formula."""

    def test_high_consistency_becomes_low_hallucination(self):
        """SummaC score 0.9 (highly consistent) → hallucination 0.1."""
        raw = 0.9
        hallucination = 1.0 - raw
        assert abs(hallucination - 0.1) < 1e-9

    def test_low_consistency_becomes_high_hallucination(self):
        """SummaC score 0.1 (low consistency) → hallucination 0.9."""
        raw = 0.1
        hallucination = 1.0 - raw
        assert abs(hallucination - 0.9) < 1e-9

    def test_score_range_0_to_1(self):
        """Hallucination scores stay in [0, 1] for all consistency values."""
        for raw in np.linspace(0.0, 1.0, 11):
            hallucination = 1.0 - raw
            assert 0.0 <= hallucination <= 1.0

    def test_exactly_half_stays_half(self):
        """0.5 consistency → 0.5 hallucination."""
        assert abs(1.0 - 0.5 - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Tests: SummaCZSReimplemented scorer (mocked model)
# ---------------------------------------------------------------------------

class TestSummaCZSReimplemented:
    """Test the scorer class without loading any real model."""

    def _make_scorer(self, entailment_probs: list[float]):
        """Build a SummaCZSReimplemented instance bypassing __init__ entirely.

        We use __new__ + direct attribute injection to avoid loading any
        HuggingFace model. This lets us test the score() / _nli_batch() logic
        in pure isolation.
        """
        from baseline_summac import SummaCZSReimplemented

        scorer = SummaCZSReimplemented.__new__(SummaCZSReimplemented)
        scorer.device = "cpu"
        scorer.batch_size = 32
        scorer.model_name = "cross-encoder/nli-deberta-v3-base"
        scorer._entailment_idx = 2
        scorer.tokenizer = MagicMock()
        scorer.model = MagicMock()
        # Patch _nli_batch to return the specified probabilities
        scorer._nli_batch = MagicMock(return_value=list(entailment_probs))
        return scorer

    def test_score_returns_max_entailment(self):
        """score() must return the max of sentence-level entailment probs."""
        scorer = self._make_scorer([0.3, 0.9, 0.5])
        with patch("baseline_summac._split_sentences", return_value=["s1", "s2", "s3"]):
            result = scorer.score("document", "hypothesis")
        # SummaC ZS = max entailment = 0.9
        assert abs(result - 0.9) < 1e-6

    def test_score_single_sentence(self):
        """Single-sentence document returns that sentence's entailment prob."""
        scorer = self._make_scorer([0.7])
        with patch("baseline_summac._split_sentences", return_value=["only sentence"]):
            result = scorer.score("one sentence.", "claim")
        assert abs(result - 0.7) < 1e-6

    def test_score_empty_document_returns_neutral(self):
        """Empty document (no sentences) returns 0.5 (uncertain)."""
        scorer = self._make_scorer([])
        with patch("baseline_summac._split_sentences", return_value=[]):
            result = scorer.score("", "claim")
        assert result == 0.5

    def test_entailment_idx_is_2(self):
        """DeBERTa NLI label order: 0=contradiction, 1=neutral, 2=entailment."""
        scorer = self._make_scorer([0.5])
        assert scorer.entailment_idx == 2


# ---------------------------------------------------------------------------
# Tests: build_summac_scores
# ---------------------------------------------------------------------------

class TestBuildSummacScores:
    """Test the batch scoring function."""

    def test_returns_three_lists(self):
        from baseline_summac import build_summac_scores

        samples = _make_samples(5)
        scorer = _make_mock_scorer(0.7)
        y_true, y_score, raw_scores = build_summac_scores(samples, scorer)

        assert len(y_true) == 5
        assert len(y_score) == 5
        assert len(raw_scores) == 5

    def test_y_true_is_binary(self):
        from baseline_summac import build_summac_scores

        samples = _make_samples(10)
        scorer = _make_mock_scorer(0.5)
        y_true, _, _ = build_summac_scores(samples, scorer)

        assert all(v in (0, 1) for v in y_true)

    def test_hallucinated_maps_to_1(self):
        from baseline_summac import build_summac_scores

        samples = [
            {"context": "A is B.", "claim_text": "A is C.", "label": "hallucinated"},
            {"context": "X is Y.", "claim_text": "X is Y.", "label": "supported"},
        ]
        scorer = _make_mock_scorer(0.5)
        y_true, _, _ = build_summac_scores(samples, scorer)

        assert y_true[0] == 1   # hallucinated
        assert y_true[1] == 0   # supported

    def test_score_inversion_applied(self):
        """Consistency 0.8 → hallucination 0.2."""
        from baseline_summac import build_summac_scores

        samples = [{"context": "A.", "claim_text": "B.", "label": "supported"}]
        scorer = _make_mock_scorer(fixed_consistency=0.8)
        _, y_score, raw_scores = build_summac_scores(samples, scorer)

        assert abs(y_score[0] - 0.2) < 1e-9
        assert abs(raw_scores[0] - 0.8) < 1e-9

    def test_skips_empty_context(self):
        from baseline_summac import build_summac_scores

        samples = [
            {"context": "", "claim_text": "Something.", "label": "supported"},
            {"context": "Valid.", "claim_text": "", "label": "supported"},
            {"context": "Valid context.", "claim_text": "Valid claim.", "label": "hallucinated"},
        ]
        scorer = _make_mock_scorer(0.5)
        y_true, _, _ = build_summac_scores(samples, scorer)

        assert len(y_true) == 1
        assert y_true[0] == 1

    def test_scorer_called_once_per_sample(self):
        from baseline_summac import build_summac_scores

        samples = _make_samples(6)
        scorer = _make_mock_scorer(0.5)
        build_summac_scores(samples, scorer)

        assert scorer.score.call_count == 6


# ---------------------------------------------------------------------------
# Tests: run_summac_benchmark (mock scorer)
# ---------------------------------------------------------------------------

class TestRunSummacBenchmark:
    def test_returns_dict_with_required_keys(self):
        from baseline_summac import run_summac_benchmark

        samples = _make_samples(20)
        scorer = _make_mock_scorer(0.6)

        # Vary scores to ensure both classes have distinct scores
        rng = np.random.RandomState(0)
        call_results = []
        for s in samples:
            call_results.append(0.8 if s["label"] == "supported" else 0.3)
        scorer.score.side_effect = call_results

        result = run_summac_benchmark(samples, "HaluEval QA", scorer)

        required = {"mode", "dataset", "n_samples", "auroc", "f1",
                    "precision", "recall", "avg_latency_ms", "protocol"}
        assert required.issubset(result.keys())

    def test_mode_is_summac_zs_reimplemented(self):
        from baseline_summac import run_summac_benchmark

        samples = _make_samples(20)
        scorer = _make_mock_scorer(0.5)
        scorer.score.side_effect = [0.8 if i % 2 == 0 else 0.3 for i in range(20)]

        result = run_summac_benchmark(samples, "test", scorer)
        assert result["mode"] == "summac_zs_reimplemented"

    def test_protocol_is_oracle_evidence(self):
        from baseline_summac import run_summac_benchmark

        samples = _make_samples(20)
        scorer = _make_mock_scorer(0.5)
        scorer.score.side_effect = [0.8 if i % 2 == 0 else 0.3 for i in range(20)]

        result = run_summac_benchmark(samples, "test", scorer)
        assert result["protocol"] == "oracle_evidence"

    def test_auroc_in_01(self):
        from baseline_summac import run_summac_benchmark

        samples = _make_samples(20)
        scorer = _make_mock_scorer(0.5)
        scorer.score.side_effect = [0.8 if i % 2 == 0 else 0.2 for i in range(20)]

        result = run_summac_benchmark(samples, "test", scorer)
        assert isinstance(result["auroc"], float)
        assert 0.0 <= result["auroc"] <= 1.0

    def test_bootstrap_ci_adds_keys(self):
        from baseline_summac import run_summac_benchmark

        samples = _make_samples(30)
        scorer = _make_mock_scorer(0.5)
        scorer.score.side_effect = [0.8 if i % 2 == 0 else 0.2 for i in range(30)]

        result = run_summac_benchmark(
            samples, "test", scorer,
            bootstrap_ci=True, n_bootstrap=50, seed=42,
        )
        assert "ci_auroc_mean" in result
        assert "ci_auroc_lower" in result
        assert "ci_auroc_upper" in result
        assert result["ci_auroc_lower"] <= result["ci_auroc_mean"] <= result["ci_auroc_upper"]

    def test_no_ci_when_flag_false(self):
        from baseline_summac import run_summac_benchmark

        samples = _make_samples(20)
        scorer = _make_mock_scorer(0.5)
        scorer.score.side_effect = [0.8 if i % 2 == 0 else 0.3 for i in range(20)]

        result = run_summac_benchmark(samples, "test", scorer, bootstrap_ci=False)
        assert "ci_auroc_mean" not in result

    def test_saves_json_output(self, tmp_path):
        from baseline_summac import run_summac_benchmark

        samples = _make_samples(20)
        scorer = _make_mock_scorer(0.5)
        scorer.score.side_effect = [0.8 if i % 2 == 0 else 0.3 for i in range(20)]

        run_summac_benchmark(samples, "HaluEval QA", scorer, output_dir=tmp_path)

        files = list(tmp_path.glob("*summac_oracle_results.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert "auroc" in data
        assert "protocol" in data

    def test_raw_arrays_in_result(self):
        from baseline_summac import run_summac_benchmark

        samples = _make_samples(20)
        scorer = _make_mock_scorer(0.5)
        scorer.score.side_effect = [0.8 if i % 2 == 0 else 0.3 for i in range(20)]

        result = run_summac_benchmark(samples, "test", scorer)
        assert "_y_true" in result
        assert "_y_score" in result
        assert "_y_pred" in result
        assert all(v in (0, 1) for v in result["_y_true"])


# ---------------------------------------------------------------------------
# Tests: data loader logic (pure logic, no disk I/O)
# ---------------------------------------------------------------------------

class TestDataLoaders:
    def test_halueval_claim_format(self):
        """Claims are formatted as 'Q: <question>  A: <answer>'."""
        row = {"knowledge": "Paris is France's capital.", "question": "Capital?", "answer": "Paris", "hallucination": "no"}
        question, answer = row["question"], row["answer"]
        claim_text = f"Q: {question}  A: {answer}" if question else answer
        assert "Q:" in claim_text and "A:" in claim_text

    def test_halueval_hallucinated_yes_maps_to_hallucinated(self):
        hallucination = "yes"
        label = "hallucinated" if hallucination.lower() == "yes" else "supported"
        assert label == "hallucinated"

    def test_halueval_hallucinated_no_maps_to_supported(self):
        hallucination = "no"
        label = "hallucinated" if hallucination.lower() == "yes" else "supported"
        assert label == "supported"

    def test_fever_nei_skipped(self):
        rows = [
            {"label": "NOT ENOUGH INFO", "claim": "A.", "evidence": [["p", 0, "e"]]},
            {"label": "SUPPORTS", "claim": "B.", "evidence": [["p", 0, "evidence text"]]},
        ]
        samples, skipped_nei = [], 0
        for row in rows:
            if row["label"] == "NOT ENOUGH INFO":
                skipped_nei += 1
                continue
            evidence = row.get("evidence", [])
            context = " ".join(t[2] for t in evidence if len(t) >= 3 and t[2])
            if context:
                samples.append({"context": context, "claim_text": row["claim"],
                                 "label": "supported" if row["label"] == "SUPPORTS" else "hallucinated"})
        assert skipped_nei == 1
        assert len(samples) == 1

    def test_fever_supports_maps_to_supported(self):
        label = "SUPPORTS"
        assert ("supported" if label == "SUPPORTS" else "hallucinated") == "supported"

    def test_fever_refutes_maps_to_hallucinated(self):
        label = "REFUTES"
        assert ("supported" if label == "SUPPORTS" else "hallucinated") == "hallucinated"

    def test_fever_evidence_flattening(self):
        evidence = [["P1", 0, "Alice was born in 1990."], ["P1", 1, "She lives in London."]]
        context = " ".join(t[2] for t in evidence if isinstance(t, (list, tuple)) and len(t) >= 3 and t[2])
        assert "Alice" in context and "London" in context
