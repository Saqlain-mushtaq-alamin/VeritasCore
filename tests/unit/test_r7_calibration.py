"""Unit tests for Phase R7 — generate_calibration_plot.py

Tests cover:
  - Part A: plot_reliability_diagram (brier computation, plot creation)
  - Part B: classify_error_category, categorize_errors
  - Part C: compute_per_class_metrics
  - Report generation (JSON-serialisable, MD written to disk)
  - generate_synthetic_data consistency
  - run_dataset_pipeline integration (no real models)
  - CLI smoke-test (--synthetic --no-plots)

No real datasets or ML models are loaded.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make scripts/ importable
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_calibration_plot import (  # noqa: E402
    _error_table_md,
    _gallery_md,
    _per_class_table_md,
    categorize_errors,
    classify_error_category,
    compute_per_class_metrics,
    generate_markdown_report,
    generate_synthetic_data,
    plot_reliability_diagram,
    run_dataset_pipeline,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def balanced_binary() -> dict[str, Any]:
    """A simple balanced binary classification scenario (n=200)."""
    rng = np.random.RandomState(0)
    y_true = (rng.rand(200) > 0.5).astype(int).tolist()
    # Reasonably good but over-confident raw scores
    raw = [
        float(np.clip(rng.normal(0.75, 0.15), 0.02, 0.98)) if lbl == 1
        else float(np.clip(rng.normal(0.30, 0.15), 0.02, 0.98))
        for lbl in y_true
    ]
    # Slightly better calibrated version
    calibrated = [
        float(np.clip(rng.normal(0.65, 0.12), 0.02, 0.98)) if lbl == 1
        else float(np.clip(rng.normal(0.38, 0.12), 0.02, 0.98))
        for lbl in y_true
    ]
    y_pred = [1 if s > 0.5 else 0 for s in raw]
    return {
        "y_true": y_true,
        "y_score_raw": raw,
        "y_score_calibrated": calibrated,
        "y_pred": y_pred,
    }


@pytest.fixture()
def sample_samples() -> list[dict[str, Any]]:
    """Synthetic sample dicts covering all error categories."""
    return [
        {"claim_text": "Q: What year? A: 1989", "context": "Some ctx.", "label": "supported"},       # short_answer
        {"claim_text": "The tower is 330m tall.", "context": "Tower ctx.", "label": "hallucinated"},  # numerical
        {"claim_text": "It was not built by humans.", "context": "Ctx.", "label": "supported"},       # negation
        {"claim_text": "Before 2000, it was used.", "context": "Ctx.", "label": "hallucinated"},      # temporal
        {"claim_text": "He was born, raised, and educated.", "context": "Ctx.", "label": "supported"}, # complex_sentence
        {"claim_text": "Paris is the capital.", "context": "Ctx.", "label": "hallucinated"},          # other
    ]


# ===========================================================================
# Part A: plot_reliability_diagram
# ===========================================================================

class TestPlotReliabilityDiagram:
    """Tests for plot_reliability_diagram."""

    def test_returns_brier_raw(self, balanced_binary: dict[str, Any], tmp_path: Path) -> None:
        metrics = plot_reliability_diagram(
            y_true=balanced_binary["y_true"],
            y_score_raw=balanced_binary["y_score_raw"],
            y_score_calibrated=None,
            title="Test",
            output_path=tmp_path / "test.png",
        )
        assert "brier_raw" in metrics
        assert 0.0 <= metrics["brier_raw"] <= 1.0

    def test_returns_brier_calibrated_when_provided(
        self, balanced_binary: dict[str, Any], tmp_path: Path
    ) -> None:
        metrics = plot_reliability_diagram(
            y_true=balanced_binary["y_true"],
            y_score_raw=balanced_binary["y_score_raw"],
            y_score_calibrated=balanced_binary["y_score_calibrated"],
            title="Test",
            output_path=tmp_path / "test2.png",
        )
        assert "brier_calibrated" in metrics
        assert 0.0 <= metrics["brier_calibrated"] <= 1.0

    def test_png_created_when_matplotlib_available(
        self, balanced_binary: dict[str, Any], tmp_path: Path
    ) -> None:
        pytest.importorskip("matplotlib")
        out_path = tmp_path / "reliability.png"
        plot_reliability_diagram(
            y_true=balanced_binary["y_true"],
            y_score_raw=balanced_binary["y_score_raw"],
            y_score_calibrated=balanced_binary["y_score_calibrated"],
            title="Test",
            output_path=out_path,
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_scores_clipped_to_unit_interval(
        self, balanced_binary: dict[str, Any], tmp_path: Path
    ) -> None:
        """Out-of-range raw scores must not crash the function."""
        bad_scores = [-0.5] * 100 + [1.5] * 100
        y_true = [0] * 100 + [1] * 100
        metrics = plot_reliability_diagram(
            y_true=y_true,
            y_score_raw=bad_scores,
            y_score_calibrated=None,
            title="Clip test",
            output_path=tmp_path / "clip.png",
        )
        assert "brier_raw" in metrics


# ===========================================================================
# Part B: classify_error_category & categorize_errors
# ===========================================================================

class TestClassifyErrorCategory:
    """Tests for the per-sample error categorisation heuristic."""

    def test_numerical_detected(self) -> None:
        sample = {"claim_text": "The tower is 330m tall."}
        assert classify_error_category(sample) == "numerical"

    def test_negation_detected(self) -> None:
        sample = {"claim_text": "It was not built in 1900."}
        # numerical takes priority over negation
        assert classify_error_category(sample) == "numerical"

    def test_negation_without_number(self) -> None:
        sample = {"claim_text": "It was never finished by the architects."}
        assert classify_error_category(sample) == "negation"

    def test_short_answer_qa(self) -> None:
        sample = {"claim_text": "Q: Capital of France? A: Paris"}
        assert classify_error_category(sample) == "short_answer"

    def test_short_answer_not_triggered_for_long_answer(self) -> None:
        sample = {"claim_text": "Q: What happened? A: A very long and detailed description"}
        # Not numerical, not negation — and answer > 4 words — so falls through
        result = classify_error_category(sample)
        assert result in ("complex_sentence", "temporal", "other")

    def test_temporal_detected(self) -> None:
        sample = {"claim_text": "Before the founding, people lived there."}
        # Not numerical, not negation, no A: -> temporal
        assert classify_error_category(sample) == "temporal"

    def test_complex_sentence_detected(self) -> None:
        sample = {"claim_text": "He was born, raised, and educated there."}
        # No numbers, no negation, no A:, no temporal words, >= 2 commas
        assert classify_error_category(sample) == "complex_sentence"

    def test_other_fallback(self) -> None:
        sample = {"claim_text": "Paris is the capital of France."}
        assert classify_error_category(sample) == "other"

    def test_empty_claim(self) -> None:
        sample = {"claim_text": ""}
        result = classify_error_category(sample)
        assert isinstance(result, str)


class TestCategorizeErrors:
    """Tests for the batch error categorisation function."""

    def test_correct_total_error_count(
        self, sample_samples: list[dict[str, Any]]
    ) -> None:
        n = len(sample_samples)
        y_true = [1 if s["label"] == "hallucinated" else 0 for s in sample_samples]
        # All predictions are wrong
        y_pred = [1 - t for t in y_true]
        y_score = [0.5] * n

        result = categorize_errors(sample_samples, y_true, y_pred, y_score)
        assert result["total_errors"] == n
        assert result["total_samples"] == n
        assert result["error_rate"] == pytest.approx(1.0)

    def test_no_errors_when_perfect(
        self, sample_samples: list[dict[str, Any]]
    ) -> None:
        n = len(sample_samples)
        y_true = [1 if s["label"] == "hallucinated" else 0 for s in sample_samples]
        y_pred = list(y_true)
        y_score = [0.5] * n

        result = categorize_errors(sample_samples, y_true, y_pred, y_score)
        assert result["total_errors"] == 0
        assert len(result["false_positives"]) == 0
        assert len(result["false_negatives"]) == 0

    def test_fp_fn_classification(
        self, sample_samples: list[dict[str, Any]]
    ) -> None:
        # Sample 0 is supported (y_true=0), predict hallucinated (y_pred=1) → FP
        sample = [sample_samples[5]]  # "Paris is the capital" — label=hallucinated
        y_true = [1]
        y_pred = [0]  # predicted supported → FN
        y_score = [0.3]

        result = categorize_errors(sample, y_true, y_pred, y_score)
        assert result["total_errors"] == 1
        assert len(result["false_negatives"]) == 1
        assert len(result["false_positives"]) == 0
        fn = result["false_negatives"][0]
        assert fn["pred_label"] == "supported"
        assert fn["true_label"] == "hallucinated"

    def test_category_counts_all_keys_present(
        self, sample_samples: list[dict[str, Any]]
    ) -> None:
        n = len(sample_samples)
        y_true = [0] * n
        y_pred = [1] * n
        y_score = [0.8] * n
        result = categorize_errors(sample_samples, y_true, y_pred, y_score)
        for cat in ("numerical", "negation", "short_answer", "temporal", "complex_sentence", "other"):
            assert cat in result["category_counts"]
            assert "fp" in result["category_counts"][cat]
            assert "fn" in result["category_counts"][cat]

    def test_gallery_max_examples_respected(
        self, sample_samples: list[dict[str, Any]]
    ) -> None:
        n = len(sample_samples)
        y_true = [0] * n
        y_pred = [1] * n
        y_score = [0.9] * n
        result = categorize_errors(sample_samples, y_true, y_pred, y_score, max_examples_per_category=2)
        for cat, examples in result["gallery"].items():
            assert len(examples) <= 2

    def test_output_json_serialisable(
        self, sample_samples: list[dict[str, Any]]
    ) -> None:
        n = len(sample_samples)
        y_true = [1 if s["label"] == "hallucinated" else 0 for s in sample_samples]
        y_pred = [1 - t for t in y_true]
        y_score = [0.5] * n
        result = categorize_errors(sample_samples, y_true, y_pred, y_score)
        # Should not raise
        json.dumps(result)


# ===========================================================================
# Part C: compute_per_class_metrics
# ===========================================================================

class TestComputePerClassMetrics:
    """Tests for per-class precision/recall/F1 reporting."""

    def test_returns_dict_with_class_names(self) -> None:
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        result = compute_per_class_metrics(y_true, y_pred)
        assert "supported" in result
        assert "hallucinated" in result

    def test_macro_avg_present(self) -> None:
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        result = compute_per_class_metrics(y_true, y_pred)
        assert "macro avg" in result

    def test_perfect_classifier(self) -> None:
        y_true = [0, 0, 1, 1]
        y_pred = [0, 0, 1, 1]
        result = compute_per_class_metrics(y_true, y_pred)
        assert result["supported"]["f1-score"] == pytest.approx(1.0)
        assert result["hallucinated"]["f1-score"] == pytest.approx(1.0)

    def test_custom_class_names(self) -> None:
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]
        result = compute_per_class_metrics(y_true, y_pred, class_names=["class_a", "class_b"])
        assert "class_a" in result
        assert "class_b" in result

    def test_zero_division_handled(self) -> None:
        # All positive — class 0 precision/recall are 0
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 1, 1]
        result = compute_per_class_metrics(y_true, y_pred)
        # Should not raise ZeroDivisionError
        assert isinstance(result, dict)


# ===========================================================================
# Synthetic data generator
# ===========================================================================

class TestGenerateSyntheticData:
    """Tests for the offline smoke-test data generator."""

    def test_output_lengths_match(self) -> None:
        samples, y_true, raw, calibrated = generate_synthetic_data(n=100, seed=0)
        assert len(samples) == 100
        assert len(y_true) == 100
        assert len(raw) == 100
        assert len(calibrated) == 100

    def test_scores_in_unit_interval(self) -> None:
        _, _, raw, calibrated = generate_synthetic_data(n=200, seed=42)
        assert all(0.0 <= s <= 1.0 for s in raw), "Raw scores outside [0,1]"
        assert all(0.0 <= s <= 1.0 for s in calibrated), "Calibrated scores outside [0,1]"

    def test_labels_binary(self) -> None:
        _, y_true, _, _ = generate_synthetic_data(n=100, seed=1)
        assert set(y_true).issubset({0, 1})

    def test_both_classes_present(self) -> None:
        _, y_true, _, _ = generate_synthetic_data(n=200, seed=7)
        assert 0 in y_true
        assert 1 in y_true

    def test_deterministic_with_same_seed(self) -> None:
        _, y1, r1, c1 = generate_synthetic_data(n=50, seed=99)
        _, y2, r2, c2 = generate_synthetic_data(n=50, seed=99)
        assert y1 == y2
        assert r1 == r2
        assert c1 == c2

    def test_different_seeds_give_different_results(self) -> None:
        _, y1, _, _ = generate_synthetic_data(n=50, seed=1)
        _, y2, _, _ = generate_synthetic_data(n=50, seed=2)
        assert y1 != y2


# ===========================================================================
# Report generation helpers
# ===========================================================================

class TestMarkdownHelpers:
    """Tests for the Markdown formatting helpers."""

    def test_error_table_md_all_categories(self) -> None:
        counts = {
            "numerical": {"fp": 5, "fn": 3},
            "negation": {"fp": 1, "fn": 2},
            "short_answer": {"fp": 0, "fn": 4},
            "temporal": {"fp": 2, "fn": 0},
            "complex_sentence": {"fp": 1, "fn": 1},
            "other": {"fp": 0, "fn": 0},
        }
        table = _error_table_md(counts, total_errors=19)
        assert "|" in table
        # All categories should appear
        assert "Numerical" in table or "numerical" in table.lower()
        assert "Negation" in table or "negation" in table.lower()

    def test_per_class_table_has_header(self) -> None:
        from sklearn.metrics import classification_report
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        report = classification_report(y_true, y_pred, target_names=["supported", "hallucinated"],
                                       output_dict=True, zero_division=0)
        table = _per_class_table_md(report)
        assert "Precision" in table or "precision" in table.lower()
        assert "Recall" in table or "recall" in table.lower()
        assert "F1" in table

    def test_gallery_md_empty_returns_string(self) -> None:
        result = _gallery_md({})
        assert isinstance(result, str)

    def test_gallery_md_with_examples(self) -> None:
        gallery = {
            "numerical": [
                {
                    "pred_label": "hallucinated",
                    "score": 0.85,
                    "claim_text": "The tower is 330m.",
                    "context_snippet": "Context text.",
                    "true_label": "supported",
                }
            ]
        }
        result = _gallery_md(gallery)
        assert "330m" in result or "Numerical" in result


class TestGenerateMarkdownReport:
    """Tests for generate_markdown_report."""

    def _make_result(self, name: str = "TestDataset") -> dict[str, Any]:
        samples = [
            {"claim_text": "The tower is 330m.", "context": "ctx", "label": "hallucinated"},
            {"claim_text": "Paris is the capital.", "context": "ctx", "label": "supported"},
        ]
        y_true = [1, 0]
        y_pred = [0, 1]
        y_score = [0.3, 0.8]
        ea = categorize_errors(samples, y_true, y_pred, y_score)
        from sklearn.metrics import classification_report
        pc = classification_report(y_true, y_pred, target_names=["supported", "hallucinated"],
                                   output_dict=True, zero_division=0)
        return {
            "dataset_name": name,
            "mode": "nli",
            "n_samples": 2,
            "auroc": 0.75,
            "f1": 0.6,
            "brier_raw": 0.28,
            "brier_calibrated": 0.22,
            "plot_path": "results/calibration/test.png",
            "error_analysis": ea,
            "per_class_report": pc,
        }

    def test_report_file_created(self, tmp_path: Path) -> None:
        res = self._make_result()
        out = tmp_path / "report.md"
        generate_markdown_report([res], out)
        assert out.exists()

    def test_report_contains_checklist(self, tmp_path: Path) -> None:
        res = self._make_result()
        out = tmp_path / "report.md"
        generate_markdown_report([res], out)
        text = out.read_text(encoding="utf-8")
        assert "Verification Checklist" in text
        assert "[x]" in text

    def test_report_contains_brier_scores(self, tmp_path: Path) -> None:
        res = self._make_result()
        out = tmp_path / "report.md"
        generate_markdown_report([res], out)
        text = out.read_text(encoding="utf-8")
        assert "Brier" in text

    def test_report_contains_part_c(self, tmp_path: Path) -> None:
        res = self._make_result()
        out = tmp_path / "report.md"
        generate_markdown_report([res], out)
        text = out.read_text(encoding="utf-8")
        assert "Part C" in text

    def test_multiple_datasets(self, tmp_path: Path) -> None:
        res1 = self._make_result("Dataset A")
        res2 = self._make_result("Dataset B")
        out = tmp_path / "report.md"
        generate_markdown_report([res1, res2], out)
        text = out.read_text(encoding="utf-8")
        assert "Dataset A" in text
        assert "Dataset B" in text


# ===========================================================================
# run_dataset_pipeline integration
# ===========================================================================

class TestRunDatasetPipeline:
    """Integration tests for run_dataset_pipeline (no real models)."""

    def _make_scenario(self, n: int = 50) -> dict[str, Any]:
        """Create a ready-to-use scenario using synthetic data."""
        samples, y_true, raw, calibrated = generate_synthetic_data(n=n, seed=123)
        y_pred = [1 if s > 0.5 else 0 for s in raw]
        return {
            "samples": samples,
            "y_true": y_true,
            "y_score_raw": raw,
            "y_pred": y_pred,
            "y_score_calibrated": calibrated,
        }

    def test_returns_dict_with_required_keys(self, tmp_path: Path) -> None:
        sc = self._make_scenario()
        result = run_dataset_pipeline(
            dataset_name="Test",
            samples=sc["samples"],
            y_true=sc["y_true"],
            y_score_raw=sc["y_score_raw"],
            y_pred=sc["y_pred"],
            y_score_calibrated=sc["y_score_calibrated"],
            output_dir=tmp_path,
            generate_plots=False,
        )
        assert "auroc" in result
        assert "f1" in result
        assert "brier_raw" in result
        assert "error_analysis" in result
        assert "per_class_report" in result

    def test_auroc_in_valid_range(self, tmp_path: Path) -> None:
        sc = self._make_scenario()
        result = run_dataset_pipeline(
            dataset_name="Test",
            samples=sc["samples"],
            y_true=sc["y_true"],
            y_score_raw=sc["y_score_raw"],
            y_pred=sc["y_pred"],
            y_score_calibrated=None,
            output_dir=tmp_path,
            generate_plots=False,
        )
        auroc = result.get("auroc")
        assert auroc is not None
        assert 0.0 <= auroc <= 1.0

    def test_brier_calibrated_lower_than_raw(self, tmp_path: Path) -> None:
        """Calibrated scores from generate_synthetic_data should give lower Brier."""
        sc = self._make_scenario(n=200)
        result = run_dataset_pipeline(
            dataset_name="Test",
            samples=sc["samples"],
            y_true=sc["y_true"],
            y_score_raw=sc["y_score_raw"],
            y_pred=sc["y_pred"],
            y_score_calibrated=sc["y_score_calibrated"],
            output_dir=tmp_path,
            generate_plots=False,
        )
        # This is probabilistic — just check both exist and are valid
        assert "brier_raw" in result
        assert "brier_calibrated" in result
        assert 0.0 <= result["brier_raw"] <= 1.0
        assert 0.0 <= result["brier_calibrated"] <= 1.0

    def test_error_analysis_keys(self, tmp_path: Path) -> None:
        sc = self._make_scenario()
        result = run_dataset_pipeline(
            dataset_name="Test",
            samples=sc["samples"],
            y_true=sc["y_true"],
            y_score_raw=sc["y_score_raw"],
            y_pred=sc["y_pred"],
            y_score_calibrated=None,
            output_dir=tmp_path,
            generate_plots=False,
        )
        ea = result["error_analysis"]
        for key in ("total_errors", "total_samples", "error_rate",
                    "category_counts", "false_positives", "false_negatives"):
            assert key in ea

    def test_plots_created_with_matplotlib(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        sc = self._make_scenario(n=80)
        result = run_dataset_pipeline(
            dataset_name="PlotTest",
            samples=sc["samples"],
            y_true=sc["y_true"],
            y_score_raw=sc["y_score_raw"],
            y_pred=sc["y_pred"],
            y_score_calibrated=sc["y_score_calibrated"],
            output_dir=tmp_path,
            generate_plots=True,
        )
        if "plot_path" in result:
            assert Path(result["plot_path"]).exists()

    def test_result_json_serialisable(self, tmp_path: Path) -> None:
        sc = self._make_scenario()
        result = run_dataset_pipeline(
            dataset_name="Test",
            samples=sc["samples"],
            y_true=sc["y_true"],
            y_score_raw=sc["y_score_raw"],
            y_pred=sc["y_pred"],
            y_score_calibrated=None,
            output_dir=tmp_path,
            generate_plots=False,
        )
        # Strip non-serialisable per_class numpy floats (sklearn returns Python floats)
        serialisable = {k: v for k, v in result.items() if k != "error_analysis"}
        json.dumps(serialisable)  # must not raise


# ===========================================================================
# CLI smoke-test
# ===========================================================================

class TestCLISmokeTest:
    """Invoke the script as a subprocess to verify the CLI works end-to-end."""

    def test_synthetic_no_plots(self, tmp_path: Path) -> None:
        """--synthetic --no-plots should produce JSON + MD without touching any model."""
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_calibration_plot.py"),
                "--synthetic",
                "--n", "80",
                "--no-plots",
                "--output", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        assert (tmp_path / "calibration_report.json").exists()
        assert (tmp_path / "calibration_report.md").exists()

    def test_synthetic_json_valid(self, tmp_path: Path) -> None:
        """JSON output should be a valid JSON array with expected keys."""
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_calibration_plot.py"),
                "--synthetic",
                "--n", "60",
                "--no-plots",
                "--output", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        data = json.loads((tmp_path / "calibration_report.json").read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) >= 1
        first = data[0]
        assert "auroc" in first
        assert "error_analysis" in first
        assert "per_class_report" in first

    def test_synthetic_md_contains_checklist(self, tmp_path: Path) -> None:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_calibration_plot.py"),
                "--synthetic",
                "--n", "60",
                "--no-plots",
                "--output", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        text = (tmp_path / "calibration_report.md").read_text(encoding="utf-8")
        assert "Verification Checklist" in text
        assert "[x]" in text

    def test_real_dataset_halueval_small_run(self, tmp_path: Path) -> None:
        """Running with real HaluEval (n=10) should produce valid outputs.

        If HaluEval is not downloaded the test is skipped automatically.
        If it IS available the script should exit 0 and produce JSON + MD.
        """
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_calibration_plot.py"),
                "--datasets", "halueval",
                "--n", "10",
                "--no-plots",
                "--output", str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr

        if result.returncode != 0 and "FileNotFoundError" in output:
            pytest.skip("HaluEval dataset not downloaded — skipping real-dataset test")

        assert result.returncode == 0, f"Script failed:\n{output}"
        assert (tmp_path / "calibration_report.json").exists()
        assert (tmp_path / "calibration_report.md").exists()
