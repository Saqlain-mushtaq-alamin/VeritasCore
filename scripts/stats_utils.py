"""Statistical utilities for VeritasCore benchmark evaluation.

Implements:
  - bootstrap_auroc_ci   — Bootstrapped 95% CI for AUROC
  - bootstrap_f1_ci      — Bootstrapped 95% CI for F1
  - delong_test          — DeLong et al. (1988) two-sample AUROC comparison
  - paired_bootstrap_test — Paired bootstrap test for AUROC difference

All functions are deterministic when `seed` is set.

References:
    DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
    Comparing the areas under two or more correlated receiver operating
    characteristic curves: a nonparametric approach. Biometrics, 44(3), 837-845.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


# ---------------------------------------------------------------------------
# Bootstrapped confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_auroc_ci(
    y_true: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrapped confidence interval for AUROC.

    Args:
        y_true:      Ground-truth binary labels (1 = positive / hallucinated).
        y_score:     Continuous score (higher = more likely positive).
        n_bootstrap: Number of bootstrap resamples (≥ 2000 recommended).
        ci:          Confidence level (default 0.95 → 95% CI).
        seed:        Random seed for reproducibility.

    Returns:
        Tuple (mean_auroc, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    aurocs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        # Skip resamples that only contain one class
        if len(np.unique(y_true[idx])) < 2:
            continue
        aurocs.append(float(roc_auc_score(y_true[idx], y_score[idx])))

    aurocs_arr = np.array(aurocs)
    alpha = (1 - ci) / 2
    return (
        float(np.mean(aurocs_arr)),
        float(np.percentile(aurocs_arr, alpha * 100)),
        float(np.percentile(aurocs_arr, (1 - alpha) * 100)),
    )


def bootstrap_f1_ci(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrapped confidence interval for F1 score.

    Args:
        y_true:      Ground-truth binary labels.
        y_pred:      Predicted binary labels.
        n_bootstrap: Number of bootstrap resamples.
        ci:          Confidence level (default 0.95).
        seed:        Random seed for reproducibility.

    Returns:
        Tuple (mean_f1, ci_lower, ci_upper).
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    f1s: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        f1s.append(float(f1_score(y_true[idx], y_pred[idx], zero_division=0)))

    f1s_arr = np.array(f1s)
    alpha = (1 - ci) / 2
    return (
        float(np.mean(f1s_arr)),
        float(np.percentile(f1s_arr, alpha * 100)),
        float(np.percentile(f1s_arr, (1 - alpha) * 100)),
    )


# ---------------------------------------------------------------------------
# DeLong's test (fast structural-component implementation)
# ---------------------------------------------------------------------------

def _delong_structural_components(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute structural components V10 and V01 for DeLong variance estimation.

    Returns:
        (V10, V01, auc) where V10 and V01 are the placement values arrays.
    """
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n1 = int(pos_mask.sum())
    n0 = int(neg_mask.sum())

    pos_scores = y_score[pos_mask]
    neg_scores = y_score[neg_mask]

    # V10[i] = P(score of negative < score of positive i)
    # V01[j] = P(score of positive > score of negative j)
    # Both computed via broadcasting
    V10 = np.mean(
        (neg_scores[:, None] < pos_scores[None, :]).astype(float)
        + 0.5 * (neg_scores[:, None] == pos_scores[None, :]).astype(float),
        axis=0,
    )  # shape (n1,)

    V01 = np.mean(
        (pos_scores[:, None] > neg_scores[None, :]).astype(float)
        + 0.5 * (pos_scores[:, None] == neg_scores[None, :]).astype(float),
        axis=0,
    )  # shape (n0,)

    auc = float(np.mean(V10))  # == np.mean(V01)
    return V10, V01, auc


def delong_test(
    y_true: list[int] | np.ndarray,
    y_score_a: list[float] | np.ndarray,
    y_score_b: list[float] | np.ndarray,
) -> tuple[float, float]:
    """DeLong's test comparing two correlated AUROCs on the same dataset.

    Tests H0: AUROC_A == AUROC_B (two-sided).

    This is the fast structural-component implementation from:
        DeLong et al. (1988), Biometrics 44(3):837-845.

    Args:
        y_true:    Shared binary ground-truth labels.
        y_score_a: Continuous scores for method A.
        y_score_b: Continuous scores for method B.

    Returns:
        (z_statistic, p_value) — two-sided p-value.
    """
    from scipy import stats

    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a)
    y_score_b = np.asarray(y_score_b)

    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true must contain both positive and negative samples.")

    n1 = int((y_true == 1).sum())
    n0 = int((y_true == 0).sum())

    # Structural components for each method
    V10_a, V01_a, auc_a = _delong_structural_components(y_true, y_score_a)
    V10_b, V01_b, auc_b = _delong_structural_components(y_true, y_score_b)

    # Variance-covariance matrix of [AUC_A, AUC_B]
    # Using the DeLong formula: Var(AUC) = (S10 + S01) / (n1 * n0)
    # Cov(AUC_A, AUC_B) = (S10_ab + S01_ab) / (n1 * n0)
    S10 = np.array([
        [np.var(V10_a, ddof=1), np.cov(V10_a, V10_b)[0, 1]],
        [np.cov(V10_a, V10_b)[0, 1], np.var(V10_b, ddof=1)],
    ])
    S01 = np.array([
        [np.var(V01_a, ddof=1), np.cov(V01_a, V01_b)[0, 1]],
        [np.cov(V01_a, V01_b)[0, 1], np.var(V01_b, ddof=1)],
    ])

    cov_matrix = S10 / n0 + S01 / n1

    # Test statistic for H0: AUC_A - AUC_B = 0
    delta = auc_a - auc_b
    var_delta = cov_matrix[0, 0] + cov_matrix[1, 1] - 2 * cov_matrix[0, 1]

    if var_delta <= 0:
        return 0.0, 1.0  # numerically degenerate

    z = delta / np.sqrt(var_delta)
    p_value = float(2 * stats.norm.sf(abs(z)))  # two-sided
    return float(z), p_value


# ---------------------------------------------------------------------------
# Paired bootstrap test
# ---------------------------------------------------------------------------

def paired_bootstrap_test(
    y_true: list[int] | np.ndarray,
    y_score_a: list[float] | np.ndarray,
    y_score_b: list[float] | np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Paired bootstrap test for AUROC difference (one-sided: is A > B?).

    Null hypothesis: AUROC_A ≤ AUROC_B
    Rejects H0 when p_value < α (e.g. 0.05).

    Args:
        y_true:      Shared binary labels.
        y_score_a:   Scores for method A (hypothesised to be better).
        y_score_b:   Scores for method B (baseline).
        n_bootstrap: Number of bootstrap resamples.
        seed:        Random seed for reproducibility.

    Returns:
        (mean_delta_auroc, p_value)  — p_value is one-sided (A > B).
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a)
    y_score_b = np.asarray(y_score_b)

    delta_aurocs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        auroc_a = float(roc_auc_score(y_true[idx], y_score_a[idx]))
        auroc_b = float(roc_auc_score(y_true[idx], y_score_b[idx]))
        delta_aurocs.append(auroc_a - auroc_b)

    arr = np.array(delta_aurocs)
    # One-sided p-value: fraction of resamples where A is NOT better than B
    p_value = float(np.mean(arr <= 0))
    return float(np.mean(arr)), p_value


# ---------------------------------------------------------------------------
# Convenience formatter
# ---------------------------------------------------------------------------

def format_ci(mean: float, lower: float, upper: float, decimals: int = 4) -> str:
    """Format a CI as 'mean (lower–upper)' for display."""
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} ({fmt.format(lower)}–{fmt.format(upper)})"
