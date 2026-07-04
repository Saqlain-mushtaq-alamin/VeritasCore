"""Unit tests for Phase 5 fusion scoring (scorer module).

Uses synthetic ClaimVerdict objects with controlled score values — no real
models are loaded. The train() path is covered with a small synthetic dataset
that can run without GPU/network, validating G5 criteria 1-8 in logic
(the actual >=500-sample, held-out-set AUROC comparison is deferred to the
integration/training script, same pattern as all prior phases).
"""
# ruff: noqa: E501 N803 N806

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from veritascore.core.types import Claim, ClaimVerdict, Verdict, VerificationMode
from veritascore.scorer import (
    FEATURE_NAMES,
    CalibrationModule,
    FusionScorer,
    extract_features,
    features_to_vector,
    verdict_to_vector,
)
from veritascore.scorer.base import BaseScorer

# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_claim(text: str = "Paris is the capital of France.") -> Claim:
    return Claim(id="c1", text=text, source_span=(0, len(text)), source_text=text)


def make_verdict(
    text: str = "Paris is the capital of France.",
    verdict: Verdict = Verdict.SUPPORTED,
    nli_score: float | None = 0.85,
    retrieval_score: float | None = 0.75,
    consistency_score: float | None = 0.80,
    evidence: str | None = "Supporting evidence.",
    mode: VerificationMode = VerificationMode.GROUNDED,
) -> ClaimVerdict:
    return ClaimVerdict(
        claim=make_claim(text),
        verdict=verdict,
        confidence=0.8,
        nli_score=nli_score,
        retrieval_score=retrieval_score,
        consistency_score=consistency_score,
        evidence=evidence,
        reason="Test verdict.",
        verification_mode=mode,
    )


def make_synthetic_data(n: int = 60, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic (X, y) with a clear learnable signal."""
    rng = np.random.default_rng(seed)
    n_pos = n // 2
    n_neg = n - n_pos
    n_feats = len(FEATURE_NAMES)

    positive = np.column_stack([
        rng.uniform(0.6, 1.0, n_pos),   # nli_entailment high
        rng.uniform(0.0, 0.4, n_pos),   # nli_contradiction low
        np.zeros(n_pos),                 # nli_score_is_missing
        rng.uniform(0.5, 1.0, n_pos),   # retrieval_score high
        rng.uniform(0.5, 1.0, n_pos),   # retrieval_agreement high
        np.zeros(n_pos),                 # retrieval_score_is_missing
        rng.uniform(0.5, 1.0, n_pos),   # consistency_score high
        np.zeros(n_pos),                 # consistency_score_is_missing
        rng.uniform(20, 60, n_pos),     # claim_length
        rng.uniform(4, 12, n_pos),      # claim_word_count
        rng.integers(0, 2, n_pos).astype(float),  # has_numbers
        rng.integers(0, 2, n_pos).astype(float),  # has_proper_nouns
        rng.uniform(0, 3, n_pos),       # num_entities
        rng.uniform(0.4, 1.0, n_pos),   # nli_confidence_gap
        np.ones(n_pos),                 # has_evidence
    ])
    negative = np.column_stack([
        rng.uniform(0.0, 0.4, n_neg),
        rng.uniform(0.6, 1.0, n_neg),
        np.zeros(n_neg),
        rng.uniform(0.0, 0.4, n_neg),
        rng.uniform(0.5, 1.0, n_neg),
        np.zeros(n_neg),
        rng.uniform(0.0, 0.4, n_neg),
        np.zeros(n_neg),
        rng.uniform(20, 60, n_neg),
        rng.uniform(4, 12, n_neg),
        rng.integers(0, 2, n_neg).astype(float),
        rng.integers(0, 2, n_neg).astype(float),
        rng.uniform(0, 3, n_neg),
        rng.uniform(0.4, 1.0, n_neg),
        np.zeros(n_neg),
    ])

    assert positive.shape[1] == n_feats, f"positive has {positive.shape[1]} cols, expected {n_feats}"
    X = np.vstack([positive, negative])
    y = np.array([1] * n_pos + [0] * n_neg)
    shuffle = rng.permutation(len(y))
    return X[shuffle], y[shuffle]


# ── BaseScorer Contract ────────────────────────────────────────────────────────

class TestBaseScorer:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseScorer()  # type: ignore[abstract]

    def test_subclass_must_implement_all_methods(self) -> None:
        class Partial(BaseScorer):
            def score_claim(self, verdict: ClaimVerdict) -> float:
                return 0.5

            def is_available(self) -> bool:
                return True

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]


# ── Feature Extraction ─────────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_all_signals_present(self) -> None:
        v = make_verdict(nli_score=0.9, retrieval_score=0.8, consistency_score=0.7)
        f = extract_features(v)
        assert f["nli_entailment"] == pytest.approx(0.9)
        assert f["nli_contradiction"] == pytest.approx(0.1)
        assert f["nli_score_is_missing"] == 0.0
        assert f["retrieval_score"] == pytest.approx(0.8)
        assert f["retrieval_score_is_missing"] == 0.0
        assert f["consistency_score"] == pytest.approx(0.7)
        assert f["consistency_score_is_missing"] == 0.0

    def test_all_signals_missing(self) -> None:
        """G5 criterion 6: handles None signals gracefully."""
        v = make_verdict(nli_score=None, retrieval_score=None, consistency_score=None)
        f = extract_features(v)
        assert f["nli_entailment"] == 0.0
        assert f["nli_contradiction"] == 0.0
        assert f["nli_score_is_missing"] == 1.0
        assert f["retrieval_score"] == 0.0
        assert f["retrieval_score_is_missing"] == 1.0
        assert f["consistency_score"] == 0.0
        assert f["consistency_score_is_missing"] == 1.0

    def test_nli_zero_is_not_treated_as_missing(self) -> None:
        """A computed nli_score of exactly 0.0 must NOT set nli_score_is_missing=1."""
        v = make_verdict(nli_score=0.0)
        f = extract_features(v)
        assert f["nli_score_is_missing"] == 0.0
        assert f["nli_entailment"] == 0.0
        assert f["nli_contradiction"] == pytest.approx(1.0)

    def test_partial_signals_retrieval_only(self) -> None:
        v = make_verdict(nli_score=None, retrieval_score=0.6, consistency_score=None)
        f = extract_features(v)
        assert f["nli_score_is_missing"] == 1.0
        assert f["retrieval_score_is_missing"] == 0.0
        assert f["consistency_score_is_missing"] == 1.0

    def test_claim_level_features_numbers(self) -> None:
        v = make_verdict(text="The tower is 330 meters tall.")
        f = extract_features(v)
        assert f["has_numbers"] == 1.0
        assert f["claim_length"] > 0
        assert f["claim_word_count"] > 0

    def test_claim_level_features_proper_nouns(self) -> None:
        v = make_verdict(text="Paris is in France.")
        f = extract_features(v)
        assert f["has_proper_nouns"] == 1.0
        assert f["num_entities"] >= 2.0

    def test_claim_without_numbers(self) -> None:
        v = make_verdict(text="The sky is blue.")
        f = extract_features(v)
        assert f["has_numbers"] == 0.0

    def test_has_evidence_flag(self) -> None:
        assert extract_features(make_verdict(evidence=None))["has_evidence"] == 0.0
        assert extract_features(make_verdict(evidence="Some text."))["has_evidence"] == 1.0

    def test_nli_confidence_gap(self) -> None:
        v = make_verdict(nli_score=0.9)
        f = extract_features(v)
        assert f["nli_confidence_gap"] == pytest.approx(abs(0.9 - 0.1))

    def test_retrieval_agreement_high(self) -> None:
        assert extract_features(make_verdict(retrieval_score=1.0))["retrieval_agreement"] == pytest.approx(1.0)

    def test_retrieval_agreement_neutral(self) -> None:
        assert extract_features(make_verdict(retrieval_score=0.5))["retrieval_agreement"] == pytest.approx(0.0)

    def test_retrieval_agreement_missing(self) -> None:
        assert extract_features(make_verdict(retrieval_score=None))["retrieval_agreement"] == 0.0

    def test_all_features_are_floats(self) -> None:
        f = extract_features(make_verdict())
        for k, val in f.items():
            assert isinstance(val, float), f"Feature {k!r} is not a float: {val!r}"

    def test_all_feature_names_present(self) -> None:
        f = extract_features(make_verdict())
        for name in FEATURE_NAMES:
            assert name in f, f"Feature {name!r} missing from extract_features() output"

    def test_features_to_vector_length(self) -> None:
        assert len(features_to_vector(extract_features(make_verdict()))) == len(FEATURE_NAMES)

    def test_features_to_vector_handles_missing_keys(self) -> None:
        vec = features_to_vector({"nli_entailment": 0.8})
        assert len(vec) == len(FEATURE_NAMES)
        assert vec[0] == pytest.approx(0.8)

    def test_verdict_to_vector_consistency(self) -> None:
        v = make_verdict()
        assert verdict_to_vector(v) == features_to_vector(extract_features(v))


# ── CalibrationModule ──────────────────────────────────────────────────────────

class TestCalibrationModule:
    @staticmethod
    def _fit_cal(n: int = 100, seed: int = 0) -> CalibrationModule:
        rng = np.random.default_rng(seed)
        raw = rng.uniform(0, 1, n)
        labels = (raw > 0.5).astype(int)
        cal = CalibrationModule()
        cal.fit(raw, labels)
        return cal

    def test_fit_and_calibrate(self) -> None:
        cal = self._fit_cal()
        out = cal.calibrate(np.array([0.2, 0.5, 0.8]))
        assert out.shape == (3,)
        assert all(0.0 <= c <= 1.0 for c in out)

    def test_calibrate_one(self) -> None:
        cal = self._fit_cal()
        assert 0.0 <= cal.calibrate_one(0.7) <= 1.0

    def test_calibrate_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            CalibrationModule().calibrate(np.array([0.5]))

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            CalibrationModule().fit(np.array([0.1, 0.9]), np.array([0, 1, 0]))

    def test_too_few_samples_raises(self) -> None:
        with pytest.raises(ValueError):
            CalibrationModule().fit(np.array([0.5]), np.array([1]))

    def test_single_class_raises(self) -> None:
        with pytest.raises(ValueError):
            CalibrationModule().fit(np.array([0.1, 0.9]), np.array([1, 1]))

    def test_is_fitted(self) -> None:
        cal = CalibrationModule()
        assert not cal.is_fitted()
        cal = self._fit_cal()
        assert cal.is_fitted()

    def test_save_and_load(self, tmp_path: Path) -> None:
        """G5 criterion 3: CalibrationModule serializes/deserializes correctly."""
        cal = self._fit_cal()
        path = tmp_path / "cal.joblib"
        cal.save(path)
        cal2 = CalibrationModule()
        assert cal2.load(path) is True
        assert cal2.is_fitted()
        assert 0.0 <= cal2.calibrate_one(0.7) <= 1.0

    def test_load_nonexistent_returns_false(self, tmp_path: Path) -> None:
        assert CalibrationModule().load(tmp_path / "no.joblib") is False

    def test_save_before_fit_raises(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            CalibrationModule().save(tmp_path / "cal.joblib")


# ── FusionScorer — Heuristic Fallback ─────────────────────────────────────────

class TestFusionScorerHeuristic:
    def test_is_available(self) -> None:
        assert FusionScorer().is_available() is True

    def test_heuristic_with_all_signals(self) -> None:
        """G5 criterion 4: heuristic fallback produces reasonable scores."""
        v = make_verdict(nli_score=0.9, retrieval_score=0.8, consistency_score=0.7)
        score = FusionScorer().score_claim(v)
        assert 0.5 < score < 1.0

    def test_heuristic_with_only_nli(self) -> None:
        v = make_verdict(nli_score=0.9, retrieval_score=None, consistency_score=None)
        assert FusionScorer().score_claim(v) == pytest.approx(0.9)

    def test_heuristic_with_only_retrieval(self) -> None:
        v = make_verdict(nli_score=None, retrieval_score=0.6, consistency_score=None)
        assert FusionScorer().score_claim(v) == pytest.approx(0.6)

    def test_heuristic_no_signals_returns_neutral(self) -> None:
        v = make_verdict(nli_score=None, retrieval_score=None, consistency_score=None)
        assert FusionScorer().score_claim(v) == pytest.approx(0.5)

    def test_heuristic_nli_zero_is_not_excluded(self) -> None:
        """Regression: nli_score=0.0 must be INCLUDED, not silently excluded
        as if it were 'missing' (the spec's `if nli > 0` bug)."""
        v_zero = make_verdict(nli_score=0.0, retrieval_score=None, consistency_score=None)
        v_missing = make_verdict(nli_score=None, retrieval_score=None, consistency_score=None)
        scorer = FusionScorer()
        assert scorer.score_claim(v_zero) != scorer.score_claim(v_missing)
        assert scorer.score_claim(v_zero) == pytest.approx(0.0)
        assert scorer.score_claim(v_missing) == pytest.approx(0.5)


# ── FusionScorer — score_response ─────────────────────────────────────────────

class TestScoreResponse:
    """G5 criterion 5: response-level scoring aggregates claim scores correctly."""

    def test_empty_returns_neutral(self) -> None:
        assert FusionScorer().score_response([]) == pytest.approx(0.5)

    def test_all_supported_high_trust(self) -> None:
        verdicts = [
            make_verdict(nli_score=0.9, verdict=Verdict.SUPPORTED),
            make_verdict(nli_score=0.85, verdict=Verdict.SUPPORTED),
        ]
        assert FusionScorer().score_response(verdicts) > 0.7

    def test_contradiction_pulls_down_score(self) -> None:
        scorer = FusionScorer()
        clean = [make_verdict(nli_score=0.9, verdict=Verdict.SUPPORTED) for _ in range(3)]
        contaminated = clean + [make_verdict(nli_score=0.1, verdict=Verdict.CONTRADICTED)]
        assert scorer.score_response(contaminated) < scorer.score_response(clean)

    def test_contradicted_weight_is_double(self) -> None:
        """Arithmetic: (1*s + 2*c) / 3."""
        scorer = FusionScorer()
        verdicts = [
            make_verdict(nli_score=1.0, verdict=Verdict.SUPPORTED, retrieval_score=None, consistency_score=None),
            make_verdict(nli_score=0.0, verdict=Verdict.CONTRADICTED, retrieval_score=None, consistency_score=None),
        ]
        expected = (1.0 * 1.0 + 2.0 * 0.0) / 3.0
        assert scorer.score_response(verdicts) == pytest.approx(expected)

    def test_single_verdict(self) -> None:
        scorer = FusionScorer()
        v = make_verdict(nli_score=0.8, retrieval_score=None, consistency_score=None)
        assert scorer.score_response([v]) == pytest.approx(scorer.score_claim(v))


# ── FusionScorer — Trained Model ──────────────────────────────────────────────

class TestFusionScorerTrained:
    def test_train_saves_artifacts(self, tmp_path: Path) -> None:
        """G5 criterion 3: model serializes/deserializes correctly (joblib)."""
        X, y = make_synthetic_data()
        metrics = FusionScorer.train(X, y, save_dir=tmp_path)
        assert (tmp_path / "model.joblib").exists()
        assert (tmp_path / "scaler.joblib").exists()
        assert "cv_auroc_mean" in metrics

    def test_train_produces_valid_metrics(self, tmp_path: Path) -> None:
        X, y = make_synthetic_data()
        metrics = FusionScorer.train(X, y, save_dir=tmp_path)
        assert 0.0 <= metrics["cv_auroc_mean"] <= 1.0
        assert metrics["n_samples"] == len(y)
        assert len(metrics["feature_importance"]) == len(FEATURE_NAMES)

    def test_train_class_balance_recorded(self, tmp_path: Path) -> None:
        X, y = make_synthetic_data()
        metrics = FusionScorer.train(X, y, save_dir=tmp_path)
        assert metrics["class_balance"]["positive"] + metrics["class_balance"]["negative"] == len(y)

    def test_load_and_score_with_trained_model(self, tmp_path: Path) -> None:
        X, y = make_synthetic_data()
        FusionScorer.train(X, y, save_dir=tmp_path)
        scorer = FusionScorer(
            model_path=tmp_path / "model.joblib",
            scaler_path=tmp_path / "scaler.joblib",
            calibrator_path=tmp_path / "calibrator.joblib",
        )
        scorer.load()
        assert scorer._model is not None
        v = make_verdict(nli_score=0.9, retrieval_score=0.8, consistency_score=0.7)
        assert 0.0 <= scorer.score_claim(v) <= 1.0

    def test_trained_scores_differ_from_heuristic(self, tmp_path: Path) -> None:
        X, y = make_synthetic_data()
        FusionScorer.train(X, y, save_dir=tmp_path)
        trained = FusionScorer(model_path=tmp_path / "model.joblib", scaler_path=tmp_path / "scaler.joblib")
        trained.load()
        heuristic = FusionScorer()
        verdicts = [
            make_verdict(nli_score=0.9, retrieval_score=0.8, consistency_score=0.7),
            make_verdict(nli_score=0.1, retrieval_score=0.2, consistency_score=0.3),
        ]
        trained_scores = [trained.score_claim(v) for v in verdicts]
        heuristic_scores = [heuristic.score_claim(v) for v in verdicts]
        for s in trained_scores:
            assert 0.0 <= s <= 1.0
        assert any(abs(t - h) > 0.001 for t, h in zip(trained_scores, heuristic_scores, strict=True))

    def test_train_invalid_model_type_raises(self, tmp_path: Path) -> None:
        X, y = make_synthetic_data()
        with pytest.raises(ValueError):
            FusionScorer.train(X, y, save_dir=tmp_path, model_type="random_forest")

    def test_train_too_few_samples_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            FusionScorer.train(np.zeros((5, len(FEATURE_NAMES))), np.array([1, 0, 1, 0, 1]), save_dir=tmp_path)

    def test_train_single_class_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            FusionScorer.train(np.zeros((20, len(FEATURE_NAMES))), np.ones(20, dtype=int), save_dir=tmp_path)

    def test_load_missing_model_uses_heuristic(self, tmp_path: Path) -> None:
        scorer = FusionScorer(
            model_path=tmp_path / "nonexistent_model.joblib",
            scaler_path=tmp_path / "nonexistent_scaler.joblib",
        )
        scorer.load()
        assert scorer._model is None
        v = make_verdict(nli_score=0.8, retrieval_score=None, consistency_score=None)
        assert scorer.score_claim(v) == pytest.approx(0.8)
