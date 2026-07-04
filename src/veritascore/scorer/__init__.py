"""Fusion / confidence scoring module (Phase 5).

Combines Phase 2 (NLI), Phase 3 (retrieval), and Phase 4 (consistency)
signals into a single calibrated trust score per claim and per response.

Public API:
    BaseScorer          — abstract interface all scorers implement
    FusionScorer        — trained LR/XGBoost fusion model with heuristic fallback
    CalibrationModule   — Platt scaling for post-hoc probability calibration
    extract_features    — ClaimVerdict -> feature dict
    features_to_vector  — feature dict -> ordered float list
    verdict_to_vector   — ClaimVerdict -> ordered float list (convenience)
    FEATURE_NAMES       — canonical feature ordering list

Example:
    >>> from veritascore.scorer import FusionScorer
    >>> scorer = FusionScorer()
    >>> scorer.load()
    >>> confidence = scorer.score_claim(verdict)
    >>> trust_score = scorer.score_response(verdicts)
"""

from veritascore.scorer.base import BaseScorer
from veritascore.scorer.calibration import CalibrationModule
from veritascore.scorer.features import (
    FEATURE_NAMES,
    extract_features,
    features_to_vector,
    verdict_to_vector,
)
from veritascore.scorer.fusion import FusionScorer

__all__ = [
    "BaseScorer",
    "FusionScorer",
    "CalibrationModule",
    "extract_features",
    "features_to_vector",
    "verdict_to_vector",
    "FEATURE_NAMES",
]
