"""Unit tests for scripts/benchmark_truthfulqa.py.

Tests cover:
  - load_truthfulqa (real disk data or mock)
  - run_benchmark with a mocked verifier
  - metric calculation (AUROC, F1, precision, recall)
  - CI generation when requested

Run with:
    pytest tests/test_benchmark_truthfulqa.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Make scripts/ importable
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from benchmark_truthfulqa import load_truthfulqa, run_benchmark  # noqa: E402
from veritascore.core.types import Verdict  # noqa: E402


class TestLoadTruthfulQA:
    def test_load_truthfulqa_samples(self) -> None:
        """Test loading from real disk if available, else skip."""
        data_path = Path(__file__).parent.parent / "data" / "datasets" / "truthfulqa" / "multiple_choice"
        if not data_path.exists():
            pytest.skip("TruthfulQA dataset not found on disk")

        samples = load_truthfulqa(n=10)
        assert len(samples) == 10
        for s in samples:
            assert "context" in s
            assert "claim_text" in s
            assert s["label"] in ("hallucinated", "supported")
            assert "question_idx" in s
            assert "choice_idx" in s

    def test_load_truthfulqa_missing_raises(self, tmp_path: Path) -> None:
        with patch("benchmark_truthfulqa.DATA_DIR", tmp_path):
            with pytest.raises(FileNotFoundError, match="TruthfulQA not found"):
                load_truthfulqa(n=10)


class TestRunBenchmarkTruthfulQA:
    def _make_fake_samples(self, n: int = 20) -> list[dict[str, Any]]:
        samples = []
        for i in range(n):
            samples.append({
                "context": f"What is fact {i}?",
                "claim_text": f"This is answer {i}.",
                "label": "hallucinated" if i % 2 == 0 else "supported",
                "question_idx": i // 2,
                "choice_idx": i % 2,
                "raw_label": 0 if i % 2 == 0 else 1,
            })
        return samples

    def test_run_benchmark_mocked_verifier(self) -> None:
        samples = self._make_fake_samples(20)

        mock_verifier = MagicMock()

        def _mock_verify(claims, context=None):
            # Return contradictory for hallucinated, supported for supported
            is_hallu = "0" in claims[0].text or "2" in claims[0].text or "4" in claims[0].text
            mock_verdict = MagicMock()
            if is_hallu:
                mock_verdict.nli_score = 0.1
                mock_verdict.contradiction_score = 0.9
                mock_verdict.reverse_entailment_score = 0.1
                mock_verdict.verdict = Verdict.CONTRADICTED
            else:
                mock_verdict.nli_score = 0.9
                mock_verdict.contradiction_score = 0.05
                mock_verdict.reverse_entailment_score = 0.8
                mock_verdict.verdict = Verdict.SUPPORTED
            return [mock_verdict]

        mock_verifier.verify.side_effect = _mock_verify

        with patch("veritascore.verifier.nli_verifier.NLIVerifier", return_value=mock_verifier):
            result = run_benchmark(samples, bootstrap_ci=False)

        assert result["dataset"] == "TruthfulQA (MC)"
        assert result["n_samples"] == 20
        assert result["auroc"] is not None
        assert 0.0 <= result["auroc"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0
        assert "avg_latency_ms" in result

    def test_run_benchmark_with_bootstrap_ci(self) -> None:
        samples = self._make_fake_samples(30)

        mock_verifier = MagicMock()

        def _mock_verify(claims, context=None):
            idx = int(claims[0].id.replace("tqa_", ""))
            is_hallu = idx % 2 == 0
            mock_verdict = MagicMock()
            mock_verdict.nli_score = 0.1 if is_hallu else 0.8
            mock_verdict.contradiction_score = 0.8 if is_hallu else 0.1
            mock_verdict.reverse_entailment_score = 0.1 if is_hallu else 0.7
            mock_verdict.verdict = Verdict.CONTRADICTED if is_hallu else Verdict.SUPPORTED
            return [mock_verdict]

        mock_verifier.verify.side_effect = _mock_verify

        with patch("veritascore.verifier.nli_verifier.NLIVerifier", return_value=mock_verifier):
            result = run_benchmark(samples, bootstrap_ci=True, n_bootstrap=100)

        assert "ci_auroc_mean" in result
        assert "ci_auroc_lower" in result
        assert "ci_auroc_upper" in result
        assert result["ci_auroc_lower"] <= result["ci_auroc_mean"] <= result["ci_auroc_upper"]
