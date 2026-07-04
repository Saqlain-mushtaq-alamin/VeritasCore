"""Benchmark NLIVerifier against HaluEval QA / FEVER for Quality Gate G2.

Computes AUROC, F1, Precision, Recall by comparing NLIVerifier verdicts
against ground-truth hallucination labels.

Target (Phase 2 §2.7): AUROC >= 0.72 (SummaC NLI-only baseline, Laban et al. 2022)

Usage:
    python scripts/benchmark_nli_verifier.py --dataset halueval --n 200
    python scripts/benchmark_nli_verifier.py --dataset fever --n 200

Requires datasets downloaded via scripts/download_datasets.py first.
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from veritascore.core.types import Claim, Verdict  # noqa: E402
from veritascore.verifier.nli_verifier import NLIVerifier  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data" / "datasets"


def load_halueval_qa(n: int) -> list[dict[str, Any]]:
    """Load HaluEval QA samples (new dataset format).

    Current schema:
        knowledge
        question
        answer
        hallucination ("yes"/"no")
    """
    from datasets import load_from_disk

    path = DATA_DIR / "halueval" / "qa_samples"
    if not path.exists():
        raise FileNotFoundError(
            f"HaluEval QA samples not found at {path}. "
            "Run: python scripts/download_datasets.py --only halueval"
        )

    ds = load_from_disk(str(path))
    split = ds["data"] if "data" in ds else next(iter(ds.values()))

    samples: list[dict[str, Any]] = []

    for i, row in enumerate(split):
        if i >= n:
            break

        context = row.get("knowledge", "")
        claim = row.get("answer", "")
        hallucination = str(row.get("hallucination", "")).strip().lower()

        if not context or not claim:
            continue

        samples.append(
            {
                "context": context,
                "claim_text": claim,
                "label": (
                    "hallucinated"
                    if hallucination == "yes"
                    else "supported"
                ),
            }
        )

    return samples

def load_fever(n: int) -> list[dict[str, Any]]:
    """Load FEVER labelled_dev samples: (claim, evidence, label)."""
    from datasets import load_from_disk

    path = DATA_DIR / "fever" / "labelled_dev"
    if not path.exists():
        raise FileNotFoundError(
            f"FEVER samples not found at {path}. "
            "Run: python scripts/download_datasets.py --only fever"
        )
    ds = load_from_disk(str(path))
    split = ds["labelled_dev"] if "labelled_dev" in ds else next(iter(ds.values()))

    samples: list[dict[str, Any]] = []
    for i, row in enumerate(split):
        if i >= n:
            break
        label = row.get("label", "")
        if label == "NOT ENOUGH INFO":
            continue  # NLIVerifier doesn't have a direct analogue; skip for binary AUROC
        samples.append({
            "context": row.get("evidence", ""),
            "claim_text": row.get("claim", ""),
            "label": "supported" if label == "SUPPORTS" else "hallucinated",
        })
    return samples


def compute_auroc(y_true: list[int], y_score: list[float]) -> float:
    """Compute AUROC without sklearn dependency assumption issues — use sklearn if present."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def compute_f1_precision_recall(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score
    return (
        float(f1_score(y_true, y_pred, zero_division=0)),
        float(precision_score(y_true, y_pred, zero_division=0)),
        float(recall_score(y_true, y_pred, zero_division=0)),
    )


def run_benchmark(samples: list[dict[str, Any]], dataset_name: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"NLIVerifier Benchmark — {dataset_name} (n={len(samples)})")
    print(f"{'=' * 70}")

    verifier = NLIVerifier()

    y_true: list[int] = []      # 1 = hallucinated/refuted, 0 = supported
    y_score: list[float] = []   # "hallucination score" = 1 - entailment_prob (higher = more likely hallucinated)
    y_pred: list[int] = []
    latencies: list[float] = []
    skipped = 0

    for i, sample in enumerate(samples):
        context = sample["context"]
        claim_text = sample["claim_text"]
        label = sample["label"]

        if not context or not claim_text:
            skipped += 1
            continue

        claim = Claim(
            id=f"bench_{i}",
            text=claim_text,
            source_span=(0, len(claim_text)),
            source_text=claim_text,
        )

        t0 = time.perf_counter()
        try:
            verdict = verifier.verify([claim], context=context)[0]
        except Exception as e:
            print(f"  ⚠ Sample {i} failed: {e}")
            skipped += 1
            continue
        latencies.append(time.perf_counter() - t0)

        true_label = 1 if label == "hallucinated" else 0
        hallucination_score = 1.0 - verdict.nli_score
        predicted_hallucinated = 1 if verdict.verdict in (Verdict.CONTRADICTED, Verdict.UNSUPPORTED) else 0

        y_true.append(true_label)
        y_score.append(hallucination_score)
        y_pred.append(predicted_hallucinated)

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(samples)} processed")

    verifier.unload()

    if skipped:
        print(f"  Skipped {skipped} malformed/failed samples")

    if len(set(y_true)) < 2:
        print("  ✗ Cannot compute AUROC — only one class present in labels")
        return

    auroc = compute_auroc(y_true, y_score)
    f1, precision, recall = compute_f1_precision_recall(y_true, y_pred)
    avg_latency_ms = 1000 * sum(latencies) / len(latencies) if latencies else 0.0

    print(f"\n  Results ({len(y_true)} evaluated samples):")
    print(f"    AUROC:      {auroc:.4f}  (target: >= 0.72)")
    print(f"    F1:         {f1:.4f}")
    print(f"    Precision:  {precision:.4f}")
    print(f"    Recall:     {recall:.4f}")
    print(f"    Avg latency/claim: {avg_latency_ms:.1f} ms")
    print()
    print(f"  Quality Gate G2: [{'✓ PASS' if auroc >= 0.72 else '✗ FAIL'}] AUROC >= 0.72")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NLIVerifier (Quality Gate G2)")
    parser.add_argument("--dataset", choices=["halueval", "fever", "both"], default="halueval")
    parser.add_argument("--n", type=int, default=200, help="Number of source samples to load")
    args = parser.parse_args()

    if args.dataset in ("halueval", "both"):
        try:
            samples = load_halueval_qa(args.n)
            run_benchmark(samples, "HaluEval QA")
        except FileNotFoundError as e:
            print(f"⚠ {e}")

    if args.dataset in ("fever", "both"):
        try:
            samples = load_fever(args.n)
            run_benchmark(samples, "FEVER labelled_dev")
        except FileNotFoundError as e:
            print(f"⚠ {e}")


if __name__ == "__main__":
    main()
