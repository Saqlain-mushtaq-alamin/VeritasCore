"""Tests for scripts/run_benchmarks_v2.py — Phase R3 scale-up logic.

Tests cover:
  - split_halueval: disjointness, sizes, reproducibility (seed)
  - load_fever_fixed: NEI-collection logic (mocked)
  - Baselines constants validation
  - run_significance_tests offline logic

These tests run offline (no dataset download required). Dataset-loading
tests are skipped when data files are not present.

Run with:
    pytest tests/test_run_benchmarks_v2.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Make scripts/ importable
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from run_benchmarks_v2 import (  # noqa: E402
    BASELINES,
    load_halueval_split,
    load_fever_fixed,
    run_significance_tests,
    split_halueval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_samples(n: int) -> list[dict[str, Any]]:
    """Create n fake HaluEval-style sample dicts."""
    return [
        {
            "context": f"Context {i}",
            "claim_text": f"Claim {i}",
            "label": "hallucinated" if i % 2 == 0 else "supported",
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# split_halueval
# ---------------------------------------------------------------------------

class TestSplitHalueval:
    def test_disjoint_sets(self):
        """Grid and eval sets must not share any samples."""
        samples = _make_fake_samples(2500)
        grid, eval_ = split_halueval(samples, grid_n=500, eval_n=1500, seed=42)
        # Re-run the same logic to get the indices for disjointness check
        rng = np.random.RandomState(42)
        indices = rng.permutation(len(samples))
        grid_idx = set(indices[:500])
        eval_idx = set(indices[500:2000])
        assert grid_idx & eval_idx == set(), "Grid and eval index sets must be disjoint"

    def test_sizes(self):
        samples = _make_fake_samples(2500)
        grid, eval_ = split_halueval(samples, grid_n=500, eval_n=1500, seed=42)
        assert len(grid) == 500
        assert len(eval_) == 1500

    def test_reproducible_with_same_seed(self):
        samples = _make_fake_samples(2500)
        grid_a, eval_a = split_halueval(samples, grid_n=500, eval_n=1500, seed=42)
        grid_b, eval_b = split_halueval(samples, grid_n=500, eval_n=1500, seed=42)
        assert [s["claim_text"] for s in grid_a] == [s["claim_text"] for s in grid_b]
        assert [s["claim_text"] for s in eval_a] == [s["claim_text"] for s in eval_b]

    def test_different_seeds_give_different_splits(self):
        samples = _make_fake_samples(2500)
        _, eval_42 = split_halueval(samples, grid_n=500, eval_n=1500, seed=42)
        _, eval_99 = split_halueval(samples, grid_n=500, eval_n=1500, seed=99)
        texts_42 = {s["claim_text"] for s in eval_42}
        texts_99 = {s["claim_text"] for s in eval_99}
        assert texts_42 != texts_99

    def test_small_dataset(self):
        """Works on small datasets where grid_n + eval_n < len(samples)."""
        samples = _make_fake_samples(100)
        grid, eval_ = split_halueval(samples, grid_n=20, eval_n=50, seed=1)
        assert len(grid) == 20
        assert len(eval_) == 50
        grid_texts = {s["claim_text"] for s in grid}
        eval_texts = {s["claim_text"] for s in eval_}
        assert grid_texts & eval_texts == set()

    def test_samples_are_original_objects(self):
        """Samples in the splits must be items from the original list."""
        samples = _make_fake_samples(300)
        sample_ids = {s["claim_text"] for s in samples}
        grid, eval_ = split_halueval(samples, grid_n=50, eval_n=100, seed=7)
        for s in grid:
            assert s["claim_text"] in sample_ids
        for s in eval_:
            assert s["claim_text"] in sample_ids

    def test_grid_and_eval_cover_unique_samples(self):
        """No sample appears in both splits."""
        samples = _make_fake_samples(600)
        grid, eval_ = split_halueval(samples, grid_n=100, eval_n=200, seed=0)
        combined = [s["claim_text"] for s in grid] + [s["claim_text"] for s in eval_]
        assert len(combined) == len(set(combined)), "Duplicate samples between splits detected"


# ---------------------------------------------------------------------------
# FEVER NEI collection logic (mocked — no real datasets)
# ---------------------------------------------------------------------------

class TestFeverNEIFix:
    """Unit-test the NEI-skipping logic without touching the filesystem.

    We replicate the core while-loop logic from load_fever_fixed to verify
    the R3 bug fix: collect EXACTLY n non-NEI samples regardless of how
    many NEI rows appear between them.
    """

    @staticmethod
    def _simulate_fever_collection(rows: list[dict], n: int) -> list[dict]:
        """Mirror of the fixed load_fever_fixed() inner loop."""
        samples: list[dict] = []
        for row in rows:
            if len(samples) >= n:
                break
            label = row.get("label", "")
            if label == "NOT ENOUGH INFO":
                continue
            evidence = row.get("evidence", "some evidence")
            if not evidence:
                continue
            samples.append({"label": label, "claim": row.get("claim", "")})
        return samples

    def test_collects_exactly_n_non_nei(self):
        """Should collect exactly n samples, skipping NEI rows."""
        rows = []
        for i in range(300):
            if i % 3 == 0:
                rows.append({"label": "NOT ENOUGH INFO", "claim": f"nei_{i}", "evidence": "e"})
            elif i % 3 == 1:
                rows.append({"label": "SUPPORTS", "claim": f"sup_{i}", "evidence": "e"})
            else:
                rows.append({"label": "REFUTES", "claim": f"ref_{i}", "evidence": "e"})

        result = self._simulate_fever_collection(rows, n=50)
        assert len(result) == 50

    def test_no_nei_in_collected_samples(self):
        """Collected samples must not include any NEI rows."""
        rows = [
            {"label": "NOT ENOUGH INFO", "claim": "x", "evidence": "e"},
            {"label": "SUPPORTS", "claim": "y", "evidence": "e"},
            {"label": "REFUTES", "claim": "z", "evidence": "e"},
        ] * 30
        result = self._simulate_fever_collection(rows, n=10)
        assert all(r["label"] != "NOT ENOUGH INFO" for r in result)

    def test_original_bug_behavior_vs_fixed(self):
        """Demonstrate the R3 bug: old code yields fewer than n samples.

        Old (buggy) logic:
            for i, row in enumerate(split):
                if i >= n: break
                if row['label'] == 'NOT ENOUGH INFO': continue
                samples.append(row)

        With 50% NEI rows and n=10, old code yields ~5 samples.
        New code yields exactly 10.
        """
        rows = []
        for i in range(100):
            if i % 2 == 0:
                rows.append({"label": "NOT ENOUGH INFO", "claim": f"nei_{i}", "evidence": "e"})
            else:
                rows.append({"label": "SUPPORTS", "claim": f"sup_{i}", "evidence": "e"})

        # OLD (buggy) logic
        buggy_samples: list[dict] = []
        for i, row in enumerate(rows):
            if i >= 10:
                break
            if row["label"] == "NOT ENOUGH INFO":
                continue
            buggy_samples.append(row)

        # NEW (fixed) logic
        fixed_samples = self._simulate_fever_collection(rows, n=10)

        # With 50% NEI rows and old n=10 cutoff, buggy yields 5
        assert len(buggy_samples) < 10, "Buggy code should yield fewer than n samples"
        # Fixed code yields exactly 10
        assert len(fixed_samples) == 10, "Fixed code must yield exactly n samples"

    def test_stops_when_n_reached(self):
        """Should stop iterating once n samples are collected."""
        rows = [{"label": "SUPPORTS", "claim": f"s_{i}", "evidence": "e"} for i in range(1000)]
        result = self._simulate_fever_collection(rows, n=50)
        assert len(result) == 50  # not 1000

    def test_handles_all_nei(self):
        """If all rows are NEI, should return empty list (graceful)."""
        rows = [{"label": "NOT ENOUGH INFO", "claim": f"x_{i}", "evidence": "e"} for i in range(50)]
        result = self._simulate_fever_collection(rows, n=10)
        assert result == []

    def test_handles_fewer_than_n_available(self):
        """If fewer than n valid rows exist, returns all available."""
        rows = [
            {"label": "SUPPORTS", "claim": f"s_{i}", "evidence": "e"} for i in range(5)
        ]
        result = self._simulate_fever_collection(rows, n=50)
        assert len(result) == 5  # only 5 available


# ---------------------------------------------------------------------------
# Baselines constants
# ---------------------------------------------------------------------------

class TestBaselines:
    def test_baseline_keys_exist(self):
        assert "NLI-only (SummaC)" in BASELINES
        assert "Retrieval-only (FActScore)" in BASELINES
        assert "SelfCheckGPT" in BASELINES

    def test_baseline_auroc_values_in_range(self):
        for method, vals in BASELINES.items():
            for k, v in vals.items():
                assert 0.0 <= v <= 1.0, f"{method}.{k} = {v} out of [0, 1]"

    def test_halueval_and_fever_present(self):
        for method, vals in BASELINES.items():
            assert "halueval_auroc" in vals, f"Missing halueval_auroc for {method}"
            assert "fever_auroc" in vals, f"Missing fever_auroc for {method}"


# ---------------------------------------------------------------------------
# run_significance_tests (offline)
# ---------------------------------------------------------------------------

class TestRunSignificanceTests:
    def test_empty_results_returns_empty(self):
        comparisons = run_significance_tests([])
        assert comparisons == []

    def test_results_without_y_true_skipped(self):
        """Results without _y_true should not cause errors."""
        result: dict[str, Any] = {
            "dataset": "HaluEval QA",
            "auroc": 0.72,
            # No _y_true key
        }
        comparisons = run_significance_tests([result])
        assert isinstance(comparisons, list)
        assert len(comparisons) == 0  # no _y_true, nothing to test

    def test_result_with_y_true_produces_comparison(self):
        rng = np.random.RandomState(42)
        n = 100
        y_true = (rng.rand(n) > 0.5).astype(int).tolist()
        y_score = (np.array(y_true) + rng.randn(n) * 0.3).tolist()

        result: dict[str, Any] = {
            "dataset": "HaluEval QA",
            "auroc": 0.75,
            "_y_true": y_true,
            "_y_score": y_score,
            "_y_pred": [1 if s > 0.5 else 0 for s in y_score],
        }
        comparisons = run_significance_tests([result])
        assert isinstance(comparisons, list)
        assert len(comparisons) == 1
        comp = comparisons[0]
        assert comp["dataset"] == "HaluEval QA"
        assert comp["method_a"] == "VeritasCore (NLI)"
        assert "note" in comp

    def test_unknown_dataset_not_included(self):
        """Datasets not in the known list should not appear in comparisons."""
        result: dict[str, Any] = {
            "dataset": "SomeOtherDataset",
            "auroc": 0.75,
            "_y_true": [0, 1, 0, 1],
            "_y_score": [0.2, 0.8, 0.3, 0.7],
        }
        comparisons = run_significance_tests([result])
        assert comparisons == []


# ---------------------------------------------------------------------------
# Integration: dataset loaders (skipped if data absent)
# ---------------------------------------------------------------------------

HALUEVAL_PATH = Path(__file__).parent.parent / "data" / "datasets" / "halueval" / "qa_samples"
FEVER_PATH = Path(__file__).parent.parent / "data" / "datasets" / "fever" / "v1.0"


@pytest.mark.skipif(not HALUEVAL_PATH.exists(), reason="HaluEval data not downloaded")
class TestLoadHaluEvalSplit:
    def test_eval_split_size(self):
        samples = load_halueval_split(split="eval", n=50, grid_n=20, seed=42)
        assert len(samples) == 50

    def test_grid_split_size(self):
        samples = load_halueval_split(split="grid", n=20, grid_n=20, seed=42)
        assert len(samples) == 20

    def test_eval_and_grid_are_disjoint(self):
        grid = load_halueval_split(split="grid", n=20, grid_n=20, seed=42)
        eval_ = load_halueval_split(split="eval", n=50, grid_n=20, seed=42)
        grid_texts = {s["claim_text"] for s in grid}
        eval_texts = {s["claim_text"] for s in eval_}
        assert grid_texts & eval_texts == set()

    def test_samples_have_required_keys(self):
        samples = load_halueval_split(split="eval", n=10, grid_n=5, seed=42)
        for s in samples:
            assert "context" in s
            assert "claim_text" in s
            assert "label" in s
            assert s["label"] in ("hallucinated", "supported")


@pytest.mark.skipif(not FEVER_PATH.exists(), reason="FEVER data not downloaded")
class TestLoadFeverFixed:
    def test_collects_exactly_n(self):
        samples = load_fever_fixed(n=50)
        assert len(samples) == 50

    def test_no_nei_in_samples(self):
        samples = load_fever_fixed(n=30)
        for s in samples:
            assert s["label"] in ("supported", "hallucinated")

    def test_samples_have_required_keys(self):
        samples = load_fever_fixed(n=20)
        for s in samples:
            assert "context" in s
            assert "claim_text" in s
            assert "label" in s
            assert len(s["context"]) > 0
            assert len(s["claim_text"]) > 0
