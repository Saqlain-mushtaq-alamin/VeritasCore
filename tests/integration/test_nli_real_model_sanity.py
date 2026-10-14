"""Integration test: Real-model sanity checks for NLI verifier and SummaC baseline.

Validates that real weights of cross-encoder/nli-deberta-v3-base correctly
discriminate entailment, contradiction, and neutral with correct argmax label
names without any mocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from baseline_summac import SummaCZSReimplemented  # noqa: E402
from veritascore.core.types import Claim, Verdict  # noqa: E402
from veritascore.verifier.nli_verifier import NLIVerifier  # noqa: E402

TEST_PAIRS = [
    {
        "name": "entailment",
        "premise": "Paris is the capital and most populous city of France.",
        "hypothesis": "Paris is the capital of France.",
        "expected_label": "entailment",
        "expected_verdict": Verdict.SUPPORTED,
    },
    {
        "name": "contradiction",
        "premise": "The earth orbits around the sun.",
        "hypothesis": "The sun orbits around the earth.",
        "expected_label": "contradiction",
        "expected_verdict": Verdict.CONTRADICTED,
    },
    {
        "name": "neutral",
        "premise": "Alice bought three apples at the grocery store on Tuesday.",
        "hypothesis": "Alice loves making apple pies on weekends.",
        "expected_label": "neutral",
        "expected_verdict": Verdict.UNSUPPORTED,
    },
]


@pytest.fixture(scope="module")
def nli_verifier() -> NLIVerifier:
    verifier = NLIVerifier()
    yield verifier
    verifier.unload()


@pytest.fixture(scope="module")
def summac_scorer() -> SummaCZSReimplemented:
    scorer = SummaCZSReimplemented(device="cuda" if torch.cuda.is_available() else "cpu")
    yield scorer
    scorer.unload()


@pytest.mark.integration
@pytest.mark.parametrize("pair", TEST_PAIRS, ids=lambda p: p["name"])
def test_real_nli_verifier_argmax_labels(nli_verifier: NLIVerifier, pair: dict[str, str]) -> None:
    """Assert NLIVerifier with real weights assigns the correct argmax label."""
    premise = pair["premise"]
    hypothesis = pair["hypothesis"]
    expected_label = pair["expected_label"]

    probs = nli_verifier._run_nli(premise, hypothesis)
    label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
    argmax_idx = int(probs.argmax())
    predicted_label = label_map[argmax_idx]

    assert predicted_label == expected_label, (
        f"NLIVerifier failed on {pair['name']}: predicted {predicted_label} "
        f"with probs [contra={probs[0]:.4f}, neutral={probs[1]:.4f}, entail={probs[2]:.4f}], "
        f"expected {expected_label}"
    )

    claim = Claim(id="c1", text=hypothesis, source_span=(0, len(hypothesis)), source_text=hypothesis)
    verdict = nli_verifier.verify([claim], context=premise)[0]
    assert verdict.verdict == pair["expected_verdict"], (
        f"NLIVerifier verdict mismatch: got {verdict.verdict}, expected {pair['expected_verdict']}"
    )


@pytest.mark.integration
@pytest.mark.parametrize("pair", TEST_PAIRS, ids=lambda p: p["name"])
def test_real_summac_argmax_labels(summac_scorer: SummaCZSReimplemented, pair: dict[str, str]) -> None:
    """Assert SummaCZSReimplemented with real weights assigns the correct argmax label."""
    premise = pair["premise"]
    hypothesis = pair["hypothesis"]
    expected_label = pair["expected_label"]

    enc = summac_scorer.tokenizer(
        [premise],
        [hypothesis],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(summac_scorer.device)

    with torch.no_grad():
        logits = summac_scorer.model(**enc).logits
        pred_idx = int(torch.argmax(logits, dim=-1)[0])
        pred_label = summac_scorer.model.config.id2label[pred_idx].lower()

    assert pred_label == expected_label, (
        f"SummaC model failed on {pair['name']}: predicted {pred_label} (id={pred_idx}), "
        f"expected {expected_label}. id2label={summac_scorer.model.config.id2label}"
    )

    score = summac_scorer.score(premise, hypothesis)
    if expected_label == "entailment":
        assert score > 0.80, f"Expected high SummaC consistency score for entailment, got {score:.4f}"
    else:
        assert score < 0.30, f"Expected low SummaC consistency score for non-entailment, got {score:.4f}"
