"""Run full benchmarks on HaluEval and FEVER datasets.

Produces:
- Per-dataset AUROC, F1, Precision, Recall
- Comparison with published baselines
- Breakdown by verification mode
- Latency statistics

Saves results as JSON to the output directory and generates a markdown
comparison table via generate_report.py.

Usage:
    python scripts/run_benchmarks.py --datasets halueval fever --output tests/benchmarks/results/
    python scripts/run_benchmarks.py --datasets halueval --n 200 --mode nli
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from veritascore.core.types import Claim, Verdict  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data" / "datasets"

# Published baseline AUROC values for comparison table
BASELINES: dict[str, dict[str, float]] = {
    "NLI-only (SummaC)": {
        "halueval_auroc": 0.720,
        "fever_auroc": 0.700,
    },
    "Retrieval-only (FActScore)": {
        "halueval_auroc": 0.680,
        "fever_auroc": 0.740,
    },
    "SelfCheckGPT": {
        "halueval_auroc": 0.740,
        "fever_auroc": 0.690,
    },
}


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_halueval_qa(n: int) -> list[dict[str, Any]]:
    """Load HaluEval QA samples (requires download_datasets.py first)."""
    from datasets import load_from_disk

    path = DATA_DIR / "halueval" / "qa_samples"
    if not path.exists():
        raise FileNotFoundError(
            f"HaluEval QA not found at {path}. "
            "Run: python scripts/download_datasets.py --only halueval"
        )

    ds = load_from_disk(str(path))
    split = ds["data"] if "data" in ds else next(iter(ds.values()))

    samples: list[dict[str, Any]] = []
    for i, row in enumerate(split):
        if i >= n:
            break
        context = row.get("knowledge", "")
        question = row.get("question", "")
        answer = row.get("answer", "")
        hallucination = str(row.get("hallucination", "")).strip().lower()
        if not context or not answer:
            continue
        claim_text = f"Q: {question}  A: {answer}" if question else answer
        samples.append({
            "context": context,
            "claim_text": claim_text,
            "label": "hallucinated" if hallucination == "yes" else "supported",
        })
    return samples


def load_fever(n: int) -> list[dict[str, Any]]:
    """Load FEVER validation samples (requires download_datasets.py first)."""
    from datasets import load_from_disk

    path = DATA_DIR / "fever" / "v1.0"
    if not path.exists():
        raise FileNotFoundError(
            f"FEVER not found at {path}. "
            "Run: python scripts/download_datasets.py --only fever"
        )
    ds = load_from_disk(str(path))
    split = ds["validation"] if "validation" in ds else next(iter(ds.values()))

    samples: list[dict[str, Any]] = []
    for i, row in enumerate(split):
        if i >= n:
            break
        label = row.get("label", "")
        if label == "NOT ENOUGH INFO":
            continue
        raw_evidence = row.get("evidence", [])
        if isinstance(raw_evidence, list) and raw_evidence:
            context = " ".join(
                triple[2] for triple in raw_evidence
                if isinstance(triple, (list, tuple)) and len(triple) >= 3 and triple[2]
            )
        else:
            context = str(raw_evidence) if raw_evidence else ""
        if not context:
            continue
        samples.append({
            "context": context,
            "claim_text": row.get("claim", ""),
            "label": "supported" if label == "SUPPORTS" else "hallucinated",
        })
    return samples


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: list[int],
    y_score: list[float],
    y_pred: list[int],
    latencies: list[float],
) -> dict[str, float]:
    """Compute full metric suite from lists of labels, scores, predictions."""
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    metrics: dict[str, float] = {}

    if len(set(y_true)) >= 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["auroc"] = float("nan")

    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

    if latencies:
        metrics["avg_latency_ms"] = float(np.mean(latencies) * 1000)
        metrics["p95_latency_ms"] = float(np.percentile(latencies, 95) * 1000)
    else:
        metrics["avg_latency_ms"] = 0.0
        metrics["p95_latency_ms"] = 0.0

    return metrics


# ---------------------------------------------------------------------------
# NLI benchmark runner
# ---------------------------------------------------------------------------

def run_nli_benchmark(
    samples: list[dict[str, Any]],
    dataset_name: str,
) -> dict[str, Any]:
    """Run NLI-only verification on a sample list, return metric dict."""
    from veritascore.verifier.nli_verifier import NLIVerifier

    print(f"\n{'=' * 60}")
    print(f"NLI Benchmark — {dataset_name} (n={len(samples)})")
    print(f"{'=' * 60}")

    verifier = NLIVerifier()
    y_true: list[int] = []
    y_score: list[float] = []
    y_pred: list[int] = []
    latencies: list[float] = []
    skipped = 0

    for i, sample in enumerate(samples):
        if not sample.get("context") or not sample.get("claim_text"):
            skipped += 1
            continue

        claim = Claim(
            id=f"bench_{i}",
            text=sample["claim_text"],
            source_span=(0, len(sample["claim_text"])),
            source_text=sample["claim_text"],
        )

        t0 = time.perf_counter()
        try:
            verdict = verifier.verify([claim], context=sample["context"])[0]
        except Exception as e:
            print(f"  Warning: sample {i} failed: {e}")
            skipped += 1
            continue
        latencies.append(time.perf_counter() - t0)

        fwd_e = float(verdict.nli_score or 0.0)
        fwd_c = float(verdict.contradiction_score or 0.0)
        rev_e = float(verdict.reverse_entailment_score or 0.0)
        hallucination_score = fwd_c - 0.2 * fwd_e - 0.6 * rev_e

        y_true.append(1 if sample["label"] == "hallucinated" else 0)
        y_score.append(hallucination_score)
        y_pred.append(
            1 if verdict.verdict in (Verdict.CONTRADICTED, Verdict.UNSUPPORTED) else 0
        )

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(samples)} processed")

    verifier.unload()

    if skipped:
        print(f"  Skipped {skipped} samples")

    metrics = compute_metrics(y_true, y_score, y_pred, latencies)
    metrics["n_samples"] = len(y_true)
    metrics["n_skipped"] = skipped
    metrics["mode"] = "nli"
    metrics["dataset"] = dataset_name

    print(f"\n  AUROC: {metrics['auroc']:.4f}")
    print(f"  F1:    {metrics['f1']:.4f}")
    print(f"  Avg latency: {metrics['avg_latency_ms']:.1f} ms/claim")

    return metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VeritasCore benchmarks on HaluEval and FEVER"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["halueval", "fever"],
        default=["halueval", "fever"],
        help="Which datasets to benchmark",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of samples per dataset",
    )
    parser.add_argument(
        "--mode",
        choices=["nli", "all"],
        default="nli",
        help="Verification mode to benchmark (nli=NLI-only, all=full pipeline)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/benchmarks/results"),
        help="Output directory for JSON results",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    all_results: list[dict[str, Any]] = []

    if "halueval" in args.datasets:
        try:
            samples = load_halueval_qa(args.n)
            if args.mode in ("nli", "all"):
                result = run_nli_benchmark(samples, "HaluEval QA")
                all_results.append(result)
                out_path = args.output / "halueval_results.json"
                out_path.write_text(json.dumps(result, indent=2))
                print(f"\n  Results saved → {out_path}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if "fever" in args.datasets:
        try:
            samples = load_fever(args.n)
            if args.mode in ("nli", "all"):
                result = run_nli_benchmark(samples, "FEVER (validation)")
                all_results.append(result)
                out_path = args.output / "fever_results.json"
                out_path.write_text(json.dumps(result, indent=2))
                print(f"\n  Results saved → {out_path}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if all_results:
        summary_path = args.output / "summary.json"
        summary_path.write_text(json.dumps({"results": all_results, "baselines": BASELINES}, indent=2))
        print(f"\nSummary saved → {summary_path}")
        print("\nRun `python scripts/generate_report.py` to generate the comparison table.")


if __name__ == "__main__":
    main()
