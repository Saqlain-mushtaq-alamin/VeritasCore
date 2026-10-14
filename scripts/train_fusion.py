"""Train the Phase 5 fusion model on HaluEval predictions.

Builds a training dataset by running the NLI verifier against HaluEval
samples (one supported, one hallucinated per source row), extracts
features, and trains a LogisticRegression (default) or XGBoost model.

Quality Gate G5 requires:
    - >= 500 training samples (criterion 2)
    - AUROC > baseline heuristic (criterion 1)
    - model saved/loaded correctly via joblib (criterion 3)

Usage:
    python scripts/train_fusion.py --n 300 --model logistic_regression
    python scripts/train_fusion.py --n 300 --model xgboost
"""
# ruff: noqa: E501 N803 N806

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from veritascore.core.types import Claim, Verdict, VerificationMode  # noqa: E402
from veritascore.scorer import FEATURE_NAMES, FusionScorer, extract_features  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data" / "datasets"
MODEL_DIR = Path(__file__).parent.parent / "models" / "fusion"


def build_synthetic_data(n: int = 1000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data covering all 17 features."""
    from validate_phase5 import build_synthetic_data as _build_synth
    return _build_synth(n=n, seed=seed)


def build_training_data_from_halueval(n_source_rows: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Build training data from HaluEval QA grid-search split via real NLI verification.

    Uses the disjoint grid-search split so training never contaminates
    the held-out evaluation split.
    """
    from run_benchmarks_v2 import load_halueval_split
    from veritascore.verifier.nli_verifier import NLIVerifier

    print(f"Loading HaluEval grid-search split (n={n_source_rows}, seed={seed})...")
    samples = load_halueval_split(split="grid", n=n_source_rows, grid_n=n_source_rows, seed=seed)

    print("Loading NLI verifier for feature extraction...")
    nli = NLIVerifier()

    print(f"Extracting features from {len(samples)} samples ...")
    features_list = []
    labels: list[int] = []

    for i, sample in enumerate(samples):
        context = sample.get("context", "")
        claim_text = sample.get("claim_text", "")
        if not context or not claim_text:
            continue

        label = 1 if sample["label"] == "supported" else 0  # 1 = supported, 0 = hallucinated

        claim = Claim(
            id=f"c{i}",
            text=claim_text,
            source_span=(0, len(claim_text)),
            source_text=claim_text,
        )
        try:
            verdict = nli.verify([claim], context=context)[0]
        except Exception:
            from veritascore.core.types import ClaimVerdict
            verdict = ClaimVerdict(
                claim=claim, verdict=Verdict.UNSUPPORTED, confidence=0.5,
                reason="extraction failed", verification_mode=VerificationMode.GROUNDED,
            )
        features_list.append(extract_features(verdict))
        labels.append(label)

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(samples)} samples processed")

    nli.unload()

    X = np.array([[f.get(name, 0.0) for name in FEATURE_NAMES] for f in features_list])
    y = np.array(labels, dtype=int)
    print(f"Built dataset: {len(y)} samples, {int(np.sum(y))} positive, {int(np.sum(1-y))} negative")
    return X, y



def main() -> None:
    parser = argparse.ArgumentParser(description="Train VeritasCore Phase 5 fusion model")
    parser.add_argument("--n", type=int, default=600, help="Number of rows (use >=500 for G5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--synthetic", action="store_true", help="Train on synthetic 17-feature data")
    parser.add_argument(
        "--model", choices=["logistic_regression", "xgboost"], default="logistic_regression"
    )
    parser.add_argument("--save-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"Phase 5 Fusion Model Training — {args.model}")
    print(f"{'=' * 70}")

    if args.synthetic:
        print("Using synthetic 17-feature training data...")
        X, y = build_synthetic_data(args.n, seed=args.seed)
    else:
        X, y = build_training_data_from_halueval(args.n, seed=args.seed)

    if len(y) < 500:
        print(f"WARNING: Only {len(y)} samples. G5 requires >=500 for reliable results. Use --n 600.")

    print(f"\nTraining {args.model} on {len(y)} samples...")
    t0 = time.time()
    metrics = FusionScorer.train(X, y, save_dir=args.save_dir, model_type=args.model)
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}")
    print("Training Results")
    print(f"{'=' * 70}")
    print(f"  Model type:    {metrics['model_type']}")
    print(f"  Samples:       {metrics['n_samples']} (train={metrics['n_train_samples']}, calib={metrics['n_calibration_samples']})")
    print(f"  Class balance: {metrics['class_balance']['positive']} pos / {metrics['class_balance']['negative']} neg")
    print(f"  CV AUROC:      {metrics['cv_auroc_mean']:.4f} ± {metrics['cv_auroc_std']:.4f}")
    print(f"  CV F1:         {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
    print(f"  Calibrator:    {'fitted' if metrics['calibrator_fitted'] else 'skipped (single class in holdout)'}")
    print(f"  Train time:    {elapsed:.1f}s")

    print("\n  Top 5 feature importances:")
    sorted_imp = sorted(metrics["feature_importance"].items(), key=lambda x: abs(x[1]), reverse=True)
    for name, imp in sorted_imp[:5]:
        print(f"    {name:30s}: {imp:+.4f}")

    print(f"\n  Model saved to: {args.save_dir}")
    print(f"\n  [{'PASS' if metrics['cv_auroc_mean'] > 0.7 else 'WARN'}] AUROC > 0.7: {metrics['cv_auroc_mean']:.4f}")
    print(f"  [{'PASS' if len(y) >= 500 else 'WARN'}] Samples >= 500: {len(y)}")


if __name__ == "__main__":
    main()
