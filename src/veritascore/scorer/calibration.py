"""Probability calibration via Platt scaling.

Listed as a Phase 5 deliverable ("CalibrationModule — Platt scaling for
probability calibration") but never actually implemented in the reference
spec. This fills that gap.

Why calibration matters here specifically: LogisticRegression's
predict_proba() output is already a reasonably well-calibrated probability
in the textbook case, but once class_weight="balanced" is used (as
FusionScorer.train() does, per the spec's own developer notes on class
imbalance) or once the model is upgraded to XGBoost (also explicitly
anticipated by the spec), the raw output drifts from a true calibrated
probability — XGBoost's outputs in particular are well-known to be poorly
calibrated out of the box. Platt scaling (fitting a 1-D logistic regression
from raw score -> true outcome on a held-out calibration set) corrects this
without touching the underlying model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CALIBRATOR_PATH = Path("models/fusion/calibrator.joblib")


class CalibrationModule:
    """Platt scaling: fits a 1-D logistic regression mapping raw scores to
    calibrated probabilities.

    Platt scaling specifically (as opposed to isotonic regression) is
    chosen because it degrades gracefully on small calibration sets —
    isotonic regression can overfit badly below a few hundred samples,
    which is a realistic concern for VeritasCore's target dataset sizes
    (G5 criterion 2 only requires >=500 *training* samples, and the
    calibration split will be smaller still).

    Example:
        >>> calibrator = CalibrationModule()
        >>> calibrator.fit(raw_scores, true_labels)
        >>> calibrated = calibrator.calibrate(np.array([0.83]))
        >>> calibrator.save(Path("models/fusion/calibrator.joblib"))
    """

    def __init__(self) -> None:
        self._calibrator: Any = None

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray) -> None:
        """Fit the Platt scaling calibrator.

        Args:
            raw_scores: Uncalibrated scores in [0, 1] (e.g. raw
                predict_proba output from FusionScorer's underlying model),
                shape (n_samples,).
            labels: True binary labels (1 = supported/factual, 0 =
                hallucinated), shape (n_samples,).

        Raises:
            ValueError: If raw_scores and labels have mismatched lengths,
                or if fewer than 2 samples are provided, or if labels
                contain only one class (Platt scaling is undefined without
                both classes present).
        """
        raw_scores = np.asarray(raw_scores, dtype=float)
        labels = np.asarray(labels)

        if len(raw_scores) != len(labels):
            raise ValueError(
                f"raw_scores and labels must have the same length "
                f"(got {len(raw_scores)} and {len(labels)})"
            )
        if len(raw_scores) < 2:
            raise ValueError("Need at least 2 samples to fit a calibrator")
        if len(set(labels.tolist())) < 2:
            raise ValueError(
                "Platt scaling requires both classes (0 and 1) present in labels"
            )

        from sklearn.linear_model import LogisticRegression

        self._calibrator = LogisticRegression()
        self._calibrator.fit(raw_scores.reshape(-1, 1), labels)
        logger.info("Calibrator fit on %d samples", len(raw_scores))

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        """Apply the fitted calibration mapping to raw scores.

        Args:
            raw_scores: Uncalibrated scores in [0, 1], shape (n_samples,)
                or a scalar-like single-element array.

        Returns:
            Calibrated probabilities, same shape as input.

        Raises:
            RuntimeError: If fit() or load() has not been called yet.
        """
        if self._calibrator is None:
            raise RuntimeError(
                "Calibrator has not been fit or loaded. Call fit() or load() first."
            )

        raw_scores = np.asarray(raw_scores, dtype=float)
        original_shape = raw_scores.shape
        flat = raw_scores.reshape(-1, 1)
        calibrated: np.ndarray = self._calibrator.predict_proba(flat)[:, 1]
        return calibrated.reshape(original_shape)

    def calibrate_one(self, raw_score: float) -> float:
        """Convenience wrapper for calibrating a single score.

        Args:
            raw_score: A single uncalibrated score in [0, 1].

        Returns:
            A single calibrated probability.
        """
        return float(self.calibrate(np.array([raw_score]))[0])

    def is_fitted(self) -> bool:
        """Return True if fit() or load() has populated the calibrator."""
        return self._calibrator is not None

    def save(self, path: Path | str = DEFAULT_CALIBRATOR_PATH) -> None:
        """Serialize the fitted calibrator to disk via joblib.

        Args:
            path: Destination path. Parent directories are created if needed.

        Raises:
            RuntimeError: If the calibrator has not been fit yet.
        """
        if self._calibrator is None:
            raise RuntimeError("Cannot save an unfit calibrator. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._calibrator, path)
        logger.info("Calibrator saved to %s", path)

    def load(self, path: Path | str = DEFAULT_CALIBRATOR_PATH) -> bool:
        """Load a previously-fit calibrator from disk.

        Args:
            path: Source path.

        Returns:
            True if a calibrator was loaded, False if the file doesn't
            exist (this is NOT an error — callers should fall back to
            uncalibrated scores in that case, same pattern as
            FusionScorer.load() falling back to the heuristic).
        """
        path = Path(path)
        if not path.exists():
            logger.info("No calibrator found at %s; calibration unavailable.", path)
            return False

        self._calibrator = joblib.load(path)
        logger.info("Calibrator loaded from %s", path)
        return True
