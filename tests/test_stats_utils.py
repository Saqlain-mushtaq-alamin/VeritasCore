"""Tests for scripts/stats_utils.py — Phase R3 statistical utilities.

Run with:
    pytest tests/test_stats_utils.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from stats_utils import (
    bootstrap_auroc_ci,
    bootstrap_f1_ci,
    delong_test,
    format_ci,
    paired_bootstrap_test,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """Returns a simple 100-sample binary classification scenario."""
    rng = np.random.RandomState(0)
    y_true = (rng.rand(100) > 0.5).astype(int).tolist()
    y_score_a = [t + rng.randn() * 0.3 for t in y_true]  # good classifier
    y_score_b = [t + rng.randn() * 0.5 for t in y_true]  # worse classifier
    y_pred = [1 if s > 0.5 else 0 for s in y_score_a]
    return y_true, y_score_a, y_score_b, y_pred


# ---------------------------------------------------------------------------
# bootstrap_auroc_ci
# ---------------------------------------------------------------------------

class TestBootstrapAurocCI:
    def test_returns_tuple_of_three(self, binary_data):
        y_true, y_score_a, _, _ = binary_data
        result = bootstrap_auroc_ci(y_true, y_score_a, n_bootstrap=100, seed=42)
        assert len(result) == 3

    def test_ci_ordering(self, binary_data):
        """Lower bound ≤ mean ≤ upper bound."""
        y_true, y_score_a, _, _ = binary_data
        mean, lo, hi = bootstrap_auroc_ci(y_true, y_score_a, n_bootstrap=200, seed=42)
        assert lo <= mean <= hi

    def test_mean_close_to_point_estimate(self, binary_data):
        """Bootstrapped mean should be close to the point AUROC."""
        from sklearn.metrics import roc_auc_score
        y_true, y_score_a, _, _ = binary_data
        point = roc_auc_score(y_true, y_score_a)
        mean, _, _ = bootstrap_auroc_ci(y_true, y_score_a, n_bootstrap=500, seed=42)
        assert abs(mean - point) < 0.01

    def test_ci_width_shrinks_with_more_data(self):
        """Wider CI for smaller samples (law of large numbers)."""
        rng = np.random.RandomState(1)
        # Use a noisy classifier so neither set reaches near-perfect AUROC
        y_small = (rng.rand(30) > 0.5).astype(int).tolist()
        score_small = [t + rng.randn() * 0.8 for t in y_small]  # noisy
        y_large = (rng.rand(500) > 0.5).astype(int).tolist()
        score_large = [t + rng.randn() * 0.8 for t in y_large]  # same noise

        _, lo_s, hi_s = bootstrap_auroc_ci(y_small, score_small, n_bootstrap=300, seed=42)
        _, lo_l, hi_l = bootstrap_auroc_ci(y_large, score_large, n_bootstrap=300, seed=42)
        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_deterministic_with_seed(self, binary_data):
        y_true, y_score_a, _, _ = binary_data
        r1 = bootstrap_auroc_ci(y_true, y_score_a, n_bootstrap=100, seed=99)
        r2 = bootstrap_auroc_ci(y_true, y_score_a, n_bootstrap=100, seed=99)
        assert r1 == r2

    def test_accepts_numpy_arrays(self, binary_data):
        y_true, y_score_a, _, _ = binary_data
        result = bootstrap_auroc_ci(
            np.array(y_true), np.array(y_score_a), n_bootstrap=50, seed=42
        )
        assert len(result) == 3


# ---------------------------------------------------------------------------
# bootstrap_f1_ci
# ---------------------------------------------------------------------------

class TestBootstrapF1CI:
    def test_returns_tuple_of_three(self, binary_data):
        y_true, _, _, y_pred = binary_data
        result = bootstrap_f1_ci(y_true, y_pred, n_bootstrap=100, seed=42)
        assert len(result) == 3

    def test_ci_ordering(self, binary_data):
        y_true, _, _, y_pred = binary_data
        mean, lo, hi = bootstrap_f1_ci(y_true, y_pred, n_bootstrap=200, seed=42)
        assert lo <= mean <= hi

    def test_values_in_01_range(self, binary_data):
        y_true, _, _, y_pred = binary_data
        mean, lo, hi = bootstrap_f1_ci(y_true, y_pred, n_bootstrap=100, seed=42)
        assert 0.0 <= lo <= hi <= 1.0

    def test_deterministic_with_seed(self, binary_data):
        y_true, _, _, y_pred = binary_data
        r1 = bootstrap_f1_ci(y_true, y_pred, n_bootstrap=100, seed=7)
        r2 = bootstrap_f1_ci(y_true, y_pred, n_bootstrap=100, seed=7)
        assert r1 == r2


# ---------------------------------------------------------------------------
# delong_test
# ---------------------------------------------------------------------------

class TestDeLongTest:
    def test_returns_z_and_p(self, binary_data):
        y_true, y_score_a, y_score_b, _ = binary_data
        z, p = delong_test(y_true, y_score_a, y_score_b)
        assert isinstance(z, float)
        assert isinstance(p, float)

    def test_p_value_in_01(self, binary_data):
        y_true, y_score_a, y_score_b, _ = binary_data
        _, p = delong_test(y_true, y_score_a, y_score_b)
        assert 0.0 <= p <= 1.0

    def test_identical_scores_high_p(self, binary_data):
        """Identical scores should give p ≈ 1 (cannot reject H0)."""
        y_true, y_score_a, _, _ = binary_data
        _, p = delong_test(y_true, y_score_a, y_score_a)
        assert p > 0.5

    def test_very_different_classifiers_low_p(self):
        """Perfect vs. random classifier should give very low p-value."""
        rng = np.random.RandomState(42)
        n = 200
        y_true = (rng.rand(n) > 0.5).astype(int).tolist()
        y_perfect = [float(t) for t in y_true]  # perfect classifier
        y_random = rng.rand(n).tolist()  # random
        _, p = delong_test(y_true, y_perfect, y_random)
        assert p < 0.05

    def test_raises_on_single_class(self):
        """Should raise ValueError if y_true has only one class."""
        y_true = [1] * 50
        y_score_a = list(np.random.rand(50))
        y_score_b = list(np.random.rand(50))
        with pytest.raises(ValueError):
            delong_test(y_true, y_score_a, y_score_b)

    def test_symmetry_z_sign(self, binary_data):
        """z(A vs B) == -z(B vs A) — sign indicates direction."""
        y_true, y_score_a, y_score_b, _ = binary_data
        z_ab, _ = delong_test(y_true, y_score_a, y_score_b)
        z_ba, _ = delong_test(y_true, y_score_b, y_score_a)
        assert abs(z_ab + z_ba) < 1e-9


# ---------------------------------------------------------------------------
# paired_bootstrap_test
# ---------------------------------------------------------------------------

class TestPairedBootstrapTest:
    def test_returns_delta_and_p(self, binary_data):
        y_true, y_score_a, y_score_b, _ = binary_data
        delta, p = paired_bootstrap_test(y_true, y_score_a, y_score_b, n_bootstrap=200, seed=42)
        assert isinstance(delta, float)
        assert isinstance(p, float)

    def test_p_value_in_01(self, binary_data):
        y_true, y_score_a, y_score_b, _ = binary_data
        _, p = paired_bootstrap_test(y_true, y_score_a, y_score_b, n_bootstrap=200, seed=42)
        assert 0.0 <= p <= 1.0

    def test_identical_scores_high_p(self, binary_data):
        """When A == B, delta == 0 always, so p = mean(delta <= 0) = 1.0.

        Note: The one-sided paired bootstrap test counts the fraction of resamples
        where A is NOT better than B (delta <= 0). When A == B, every resample
        gives delta = 0, so p = 1.0 (we cannot reject H0 that A <= B).
        This is the correct statistical behaviour, not a bug.
        """
        y_true, y_score_a, _, _ = binary_data
        delta, p = paired_bootstrap_test(
            y_true, y_score_a, y_score_a, n_bootstrap=500, seed=42
        )
        assert abs(delta) < 1e-10  # delta should be exactly 0
        assert p == 1.0  # one-sided: cannot reject H0 when A == B

    def test_deterministic_with_seed(self, binary_data):
        y_true, y_score_a, y_score_b, _ = binary_data
        r1 = paired_bootstrap_test(y_true, y_score_a, y_score_b, n_bootstrap=100, seed=55)
        r2 = paired_bootstrap_test(y_true, y_score_a, y_score_b, n_bootstrap=100, seed=55)
        assert r1 == r2

    def test_better_classifier_lower_p(self):
        """The clearly better classifier should have low one-sided p-value."""
        rng = np.random.RandomState(0)
        n = 200
        y_true = (rng.rand(n) > 0.5).astype(int).tolist()
        y_good = [t + rng.randn() * 0.1 for t in y_true]  # very good
        y_poor = [t + rng.randn() * 1.0 for t in y_true]  # noisy
        _, p = paired_bootstrap_test(y_true, y_good, y_poor, n_bootstrap=1000, seed=42)
        assert p < 0.05


# ---------------------------------------------------------------------------
# format_ci
# ---------------------------------------------------------------------------

class TestFormatCI:
    def test_basic_format(self):
        result = format_ci(0.721, 0.695, 0.747)
        assert "0.7210" in result
        assert "0.6950" in result
        assert "0.7470" in result

    def test_custom_decimals(self):
        result = format_ci(0.721, 0.695, 0.747, decimals=2)
        assert "0.72" in result

    def test_returns_string(self):
        assert isinstance(format_ci(0.5, 0.4, 0.6), str)
