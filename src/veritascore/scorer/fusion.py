"""Trained fusion model for combining verification signals.

Combines Phase 2 (NLI), Phase 3 (retrieval), and Phase 4 (consistency)
signals into a single calibrated trust score, per the Phase 5 spec.

Falls back to a hand-tuned heuristic weighted average when no trained
model is available on disk yet (first-run experience, per spec developer
notes: "The heuristic fallback is important — it allows the system to
work before the fusion model is trained").
"""
# ruff: noqa: N803 N806

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from veritascore.core.types import ClaimVerdict, Verdict
from veritascore.scorer.base import BaseScorer
from veritascore.scorer.calibration import CalibrationModule
from veritascore.scorer.features import FEATURE_NAMES, extract_features, features_to_vector

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/fusion/model.joblib")
DEFAULT_SCALER_PATH = Path("models/fusion/scaler.joblib")
DEFAULT_CALIBRATOR_PATH = Path("models/fusion/calibrator.joblib")

# Weight applied to CONTRADICTED claims in score_response()'s weighted
# average — contradictions are a more severe failure mode than mere
# unsupported claims (no evidence either way), so they pull the aggregate
# trust score down harder. Matches the spec's own stated rationale.
_CONTRADICTED_WEIGHT = 2.0
_DEFAULT_WEIGHT = 1.0

# Heuristic fallback weights (used only when no trained model is loaded).
_HEURISTIC_WEIGHTS = {"nli": 0.5, "retrieval": 0.3, "consistency": 0.2}


class FusionScorer(BaseScorer):
    """Fuse NLI, retrieval, and consistency signals into calibrated trust scores.

    Uses a trained logistic regression (upgradable to XGBoost — see
    train(model_type="xgboost")) to combine multiple verification signals
    into a single trust probability. Falls back to a heuristic weighted
    average when no trained model has been loaded.

    Args:
        model_path: Path to the serialized sklearn/XGBoost model.
        scaler_path: Path to the serialized StandardScaler.
        calibrator_path: Path to the serialized CalibrationModule. Optional
            — if absent, the model's raw predict_proba output is used
            directly (which is what the underlying LogisticRegression
            already calibrates reasonably well by default).

    Example:
        >>> scorer = FusionScorer()
        >>> scorer.load()  # Loads trained model if available, else heuristic
        >>> confidence = scorer.score_claim(verdict)
        >>> trust_score = scorer.score_response(verdicts)
    """

    def __init__(
        self,
        model_path: Path | None = None,
        scaler_path: Path | None = None,
        calibrator_path: Path | None = None,
    ) -> None:
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.scaler_path = scaler_path or DEFAULT_SCALER_PATH
        self.calibrator_path = calibrator_path or DEFAULT_CALIBRATOR_PATH
        self._model: Any = None
        self._scaler: Any = None
        self._calibrator: CalibrationModule | None = None

    def load(self) -> None:
        """Load trained model, scaler, and calibrator from disk, if present.

        Missing artifacts are not errors — each falls back independently:
        no model -> heuristic scoring; no scaler -> raw (unscaled)
        features passed to the model; no calibrator -> the model's own
        predict_proba output is used uncalibrated.
        """
        if self.model_path.exists():
            self._model = joblib.load(self.model_path)
            logger.info("Loaded fusion model from %s", self.model_path)
        else:
            logger.warning(
                "No trained model found at %s. Using heuristic fallback scoring.",
                self.model_path,
            )
            self._model = None

        if self.scaler_path.exists():
            self._scaler = joblib.load(self.scaler_path)
            logger.info("Loaded feature scaler from %s", self.scaler_path)

        if self.calibrator_path.exists():
            calibrator = CalibrationModule()
            if calibrator.load(self.calibrator_path):
                self._calibrator = calibrator
                logger.info("Loaded score calibrator from %s", self.calibrator_path)

    def score_claim(self, verdict: ClaimVerdict) -> float:
        """Score a single claim verdict. Returns calibrated confidence [0, 1].

        Args:
            verdict: A ClaimVerdict with zero or more signal fields populated.

        Returns:
            Confidence in [0.0, 1.0]. Uses the trained model (+ optional
            calibration) if loaded, else the heuristic weighted average.
        """
        features = extract_features(verdict)

        if self._model is not None:
            feature_vec = np.array([features_to_vector(features)])
            if self._scaler is not None:
                feature_vec = self._scaler.transform(feature_vec)

            proba = self._model.predict_proba(feature_vec)[0]
            # Class order: [not-supported (0), supported (1)]
            raw_score = float(proba[1])

            if self._calibrator is not None:
                return self._calibrator.calibrate_one(raw_score)
            return raw_score

        return self._heuristic_score(features)

    def score_response(self, verdicts: list[ClaimVerdict]) -> float:
        """Compute overall response trust score from individual claim scores.

        Aggregation is a weighted average where CONTRADICTED claims count
        double (a confident false statement is worse than an unverifiable
        one), matching the spec's own stated rationale.

        Args:
            verdicts: All claim verdicts for a response. May be empty.

        Returns:
            Aggregate trust score in [0.0, 1.0]. Returns 0.5 (neutral) for
            an empty list.
        """
        if not verdicts:
            return 0.5

        scores = [self.score_claim(v) for v in verdicts]

        weights = [
            _CONTRADICTED_WEIGHT if v.verdict == Verdict.CONTRADICTED else _DEFAULT_WEIGHT
            for v in verdicts
        ]

        weighted_sum = sum(s * w for s, w in zip(scores, weights, strict=True))
        total_weight = sum(weights)

        if total_weight <= 0:
            return 0.5

        return weighted_sum / total_weight

    def _heuristic_score(self, features: dict[str, float]) -> float:
        """Fallback heuristic when no trained model is available.

        Only averages over signals that were actually computed (using the
        `*_is_missing` indicator features from extract_features(), rather
        than the reference spec's `if nli > 0` check, which incorrectly
        treats a genuinely-computed score of exactly 0.0 — e.g. a claim
        NLI-scored as having zero entailment probability — as "missing"
        and silently excludes it from the average).

        Args:
            features: Feature dict from extract_features().

        Returns:
            Weighted average confidence in [0.0, 1.0], or 0.5 if no
            signals were computed at all.
        """
        available: list[tuple[float, float]] = []

        if features.get("nli_score_is_missing", 1.0) == 0.0:
            available.append((features["nli_entailment"], _HEURISTIC_WEIGHTS["nli"]))
        if features.get("retrieval_score_is_missing", 1.0) == 0.0:
            available.append((features["retrieval_score"], _HEURISTIC_WEIGHTS["retrieval"]))
        if features.get("consistency_score_is_missing", 1.0) == 0.0:
            available.append((features["consistency_score"], _HEURISTIC_WEIGHTS["consistency"]))

        if not available:
            return 0.5

        weighted = sum(score * weight for score, weight in available)
        total_weight = sum(weight for _, weight in available)
        return weighted / total_weight

    def is_available(self) -> bool:
        """Always True — the heuristic fallback works without a trained model."""
        return True

    @staticmethod
    def train(
        X: np.ndarray,
        y: np.ndarray,
        save_dir: Path = Path("models/fusion"),
        model_type: str = "logistic_regression",
        calibration_holdout_fraction: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Train the fusion model and save it (plus scaler and calibrator) to disk.

        Per the spec's own developer notes: start with LogisticRegression
        (interpretable, fast, less prone to overfitting on small data);
        only use model_type="xgboost" if LR doesn't meet Quality Gate G5.

        A calibration_holdout_fraction of the data is held out (after
        cross-validation, before final-model fitting) specifically to fit
        a CalibrationModule on genuinely unseen predictions — fitting
        Platt scaling on the same data the model was trained on would
        underestimate the model's true overconfidence.

        Args:
            X: Feature matrix (n_samples, n_features), columns in
                FEATURE_NAMES order.
            y: Binary labels (n_samples,). 1 = supported/factual, 0 =
                hallucinated.
            save_dir: Directory to save model/scaler/calibrator artifacts.
            model_type: "logistic_regression" (default) or "xgboost".
            calibration_holdout_fraction: Fraction of data reserved for
                fitting the Platt-scaling calibrator.
            random_state: Random seed for reproducibility.

        Returns:
            Training metrics dict with cv_auroc_mean/std, cv_f1_mean/std,
            feature_importance, n_samples, class_balance, and model_type.

        Raises:
            ValueError: If model_type is not recognized, or if X and y
                have mismatched lengths, or if fewer than 10 samples are
                provided (cross-validation with 5 folds needs a sane
                minimum; the spec's own G5 criterion 2 requires >=500 in
                practice, but we don't hard-fail below that here so unit
                tests can use small synthetic datasets).
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
        from sklearn.preprocessing import StandardScaler

        if model_type not in ("logistic_regression", "xgboost"):
            raise ValueError(
                f"model_type must be 'logistic_regression' or 'xgboost', got '{model_type}'"
            )
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length (got {len(X)} and {len(y)})")
        if len(X) < 10:
            raise ValueError(f"Need at least 10 samples to train, got {len(X)}")
        if len(set(y.tolist())) < 2:
            raise ValueError("Training requires both classes (0 and 1) present in y")

        save_dir.mkdir(parents=True, exist_ok=True)

        # Hold out a calibration split BEFORE any fitting, so the
        # calibrator never sees predictions the model was trained on.
        X_train, X_calib, y_train, y_calib = train_test_split(
            X, y,
            test_size=calibration_holdout_fraction,
            stratify=y,
            random_state=random_state,
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_calib_scaled = scaler.transform(X_calib)

        model = FusionScorer._build_model(model_type, random_state)

        n_splits = min(5, int(np.bincount(y_train.astype(int)).min()))
        if n_splits < 2:
            logger.warning(
                "Smallest class has only %d sample(s) in the training split; "
                "skipping cross-validation and reporting single-split metrics instead.",
                n_splits,
            )
            cv_auroc = np.array([float("nan")])
            cv_f1 = np.array([float("nan")])
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            cv_auroc = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="roc_auc")
            cv_f1 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="f1")

        # Fit final model on the full training split
        model.fit(X_train_scaled, y_train)

        # Fit calibrator on the held-out calibration split's predictions
        calib_raw_scores = model.predict_proba(X_calib_scaled)[:, 1]
        calibrator = CalibrationModule()
        calibrator_fitted = False
        if len(set(y_calib.tolist())) >= 2:
            calibrator.fit(calib_raw_scores, y_calib)
            calibrator_fitted = True
        else:
            logger.warning(
                "Calibration holdout split has only one class present; "
                "skipping calibrator fitting. Raw model scores will be used uncalibrated."
            )

        # Save artifacts
        joblib.dump(model, save_dir / "model.joblib")
        joblib.dump(scaler, save_dir / "scaler.joblib")
        if calibrator_fitted:
            calibrator.save(save_dir / "calibrator.joblib")

        importance = FusionScorer._extract_feature_importance(model, model_type)

        metrics: dict[str, Any] = {
            "model_type": model_type,
            "cv_auroc_mean": float(np.nanmean(cv_auroc)),
            "cv_auroc_std": float(np.nanstd(cv_auroc)),
            "cv_f1_mean": float(np.nanmean(cv_f1)),
            "cv_f1_std": float(np.nanstd(cv_f1)),
            "feature_importance": importance,
            "n_samples": int(len(y)),
            "n_train_samples": int(len(y_train)),
            "n_calibration_samples": int(len(y_calib)),
            "calibrator_fitted": calibrator_fitted,
            "class_balance": {
                "positive": int(np.sum(y == 1)),
                "negative": int(np.sum(y == 0)),
            },
        }

        logger.info(
            "Fusion model (%s) trained: AUROC=%.3f±%.3f",
            model_type, metrics["cv_auroc_mean"], metrics["cv_auroc_std"],
        )
        return metrics

    @staticmethod
    def _build_model(model_type: str, random_state: int) -> Any:
        """Construct an unfitted model instance for the given model_type."""
        if model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as e:
                raise ImportError(
                    "model_type='xgboost' requires the xgboost package: "
                    "pip install veritascore[xgboost]"
                ) from e
            return XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=random_state,
                eval_metric="logloss",
            )

        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )

    @staticmethod
    def _extract_feature_importance(model: Any, model_type: str) -> dict[str, float]:
        """Extract a feature_name -> importance mapping, handling both
        LogisticRegression's .coef_ and XGBoost's .feature_importances_."""
        if model_type == "xgboost":
            importances = model.feature_importances_
            return dict(zip(FEATURE_NAMES, [float(v) for v in importances], strict=True))

        coefs = model.coef_[0]
        return dict(zip(FEATURE_NAMES, [float(v) for v in coefs], strict=True))
