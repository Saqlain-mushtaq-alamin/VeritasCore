"""Feature extraction from verification signals for the fusion model.

Converts a ClaimVerdict (populated by some subset of Phases 2-4) into a
fixed-order numeric feature vector suitable for sklearn/XGBoost input.

Design note on missing signals (Quality Gate G5 criterion 6 — "Feature
extraction handles missing signals gracefully"): a claim verified in
GROUNDED mode has .retrieval_score = None (Phase 3 never ran); a claim
verified in UNGROUNDED mode has no .nli_score from context (Phase 2 never
ran against this claim, though RetrievalVerifier DOES populate .nli_score
from its own internal NLI calls against retrieved evidence). Rather than
defaulting every missing signal to the same neutral value, this module
distinguishes "signal not computed" (None -> 0.0, with an explicit
`*_is_missing` indicator feature so the model can learn to discount it)
from "signal computed and is genuinely low" (an explicit small float).
"""

from __future__ import annotations

import re

from veritascore.core.types import ClaimVerdict

# Proper-noun token pattern: a capitalized word, optionally followed by
# more capitalized words (crude multi-word named-entity approximation,
# e.g. "Eiffel Tower", "United States"). Word-boundary anchored so it
# doesn't match capitalized letters mid-word.
_PROPER_NOUN_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b")
_NUMBER_PATTERN = re.compile(r"\d+")

# Fixed feature order — MUST stay in sync between extract_features(),
# features_to_vector(), and any trained model's expected input shape.
# Append-only: changing the order or removing a feature invalidates any
# previously-trained/serialized model.
FEATURE_NAMES: list[str] = [
    "nli_entailment",
    "nli_contradiction",
    "nli_score_is_missing",
    "retrieval_score",
    "retrieval_agreement",
    "retrieval_score_is_missing",
    "consistency_score",
    "consistency_score_is_missing",
    "claim_length",
    "claim_word_count",
    "has_numbers",
    "has_proper_nouns",
    "num_entities",
    "nli_confidence_gap",
    "has_evidence",
]


def extract_features(verdict: ClaimVerdict) -> dict[str, float]:
    """Extract a feature dict from a ClaimVerdict.

    Every value in the returned dict is a plain float (booleans are 0.0/1.0)
    so the dict can be fed directly into features_to_vector() or inspected
    for debugging without type surprises.

    Missing-signal handling (G5 criterion 6): if `verdict.nli_score` is
    None, `nli_entailment` defaults to 0.0 and `nli_score_is_missing` is
    set to 1.0 — the same pattern for `retrieval_score` and
    `consistency_score`. This lets a trained model learn "we have no
    opinion here" as distinct from "we checked and it's a 0.0", which a
    naive `or 0.0` default conflates.

    Args:
        verdict: A ClaimVerdict, typically with some subset of
            .nli_score/.retrieval_score/.consistency_score populated
            depending on which verifiers ran (grounded vs. ungrounded
            mode, whether Phase 4 was invoked, etc).

    Returns:
        Dict mapping feature name -> float value. Keys match FEATURE_NAMES.

    Example:
        >>> features = extract_features(verdict)
        >>> features["nli_entailment"]
        0.85
        >>> features["retrieval_score_is_missing"]
        1.0
    """
    claim_text = verdict.claim.text

    nli_missing = verdict.nli_score is None
    retrieval_missing = verdict.retrieval_score is None
    consistency_missing = verdict.consistency_score is None

    nli_entailment: float = verdict.nli_score if verdict.nli_score is not None else 0.0
    nli_contradiction: float = (1.0 - nli_entailment) if not nli_missing else 0.0

    retrieval_score: float = verdict.retrieval_score if verdict.retrieval_score is not None else 0.0
    retrieval_agreement: float = abs(retrieval_score - 0.5) * 2.0 if not retrieval_missing else 0.0

    consistency_score: float = (
        verdict.consistency_score if verdict.consistency_score is not None else 0.0
    )

    proper_nouns = _PROPER_NOUN_PATTERN.findall(claim_text)

    return {
        "nli_entailment": float(nli_entailment),
        "nli_contradiction": float(nli_contradiction),
        "nli_score_is_missing": 1.0 if nli_missing else 0.0,
        "retrieval_score": float(retrieval_score),
        "retrieval_agreement": float(retrieval_agreement),
        "retrieval_score_is_missing": 1.0 if retrieval_missing else 0.0,
        "consistency_score": float(consistency_score),
        "consistency_score_is_missing": 1.0 if consistency_missing else 0.0,
        "claim_length": float(len(claim_text)),
        "claim_word_count": float(len(claim_text.split())),
        "has_numbers": 1.0 if _NUMBER_PATTERN.search(claim_text) else 0.0,
        "has_proper_nouns": 1.0 if proper_nouns else 0.0,
        "num_entities": float(len(proper_nouns)),
        "nli_confidence_gap": float(abs(nli_entailment - nli_contradiction)),
        "has_evidence": 1.0 if verdict.evidence else 0.0,
    }


def features_to_vector(features: dict[str, float]) -> list[float]:
    """Convert a feature dict to an ordered numeric vector.

    Args:
        features: Feature dict, typically from extract_features(). Any
            keys not in FEATURE_NAMES are ignored; any missing keys
            default to 0.0 (graceful degradation, matching G5 criterion 6).

    Returns:
        List of floats in FEATURE_NAMES order.
    """
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def verdict_to_vector(verdict: ClaimVerdict) -> list[float]:
    """Convenience: extract_features() + features_to_vector() in one call.

    Args:
        verdict: A ClaimVerdict.

    Returns:
        List of floats in FEATURE_NAMES order, ready for model.predict().
    """
    return features_to_vector(extract_features(verdict))
