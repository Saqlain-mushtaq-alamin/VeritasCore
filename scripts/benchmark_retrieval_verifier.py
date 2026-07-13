"""Benchmark RetrievalVerifier against HaluEval for Quality Gate G3.

Computes F1, Precision, Recall by comparing RetrievalVerifier verdicts
against ground-truth hallucination labels, using LIVE web search (Tavily
or Brave, whichever is configured) — NOT the provided reference context,
since this benchmarks the ungrounded (retrieval) path specifically.

Target (Phase 3 Quality Gate G3 criterion 4): F1 >= baseline retrieval-only method.

Usage:
    python scripts/benchmark_retrieval_verifier.py --dataset halueval --n 50

Requires:
    - A TAVILY_API_KEY or BRAVE_API_KEY configured (.env or environment)
    - Datasets downloaded via scripts/download_datasets.py
    - The NLI model downloaded via scripts/download_models.py

Note: live web search is rate-limited and slow. Start with a SMALL --n
(e.g. 20-50) to sanity-check before running a larger benchmark — search
results are cached, so re-runs of the same sample set are fast/free.
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
from veritascore.verifier.retrieval_verifier import RetrievalVerifier  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data" / "datasets"
print("### THIS IS THE MODIFIED BENCHMARK ###")

def load_halueval_qa(n: int) -> list[dict[str, Any]]:
    """Load HaluEval QA samples for the retrieval (ungrounded) benchmark.

    Unlike the NLI benchmark (Phase 2), this does NOT use the provided
    `knowledge` field as context — RetrievalVerifier retrieves its own
    evidence from the web, simulating the real ungrounded-mode scenario.
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

        samples.append(
           {
                "claim_text": row["answer"],
                 "label": "hallucinated"
                if row["hallucination"].lower() == "yes"
                else "supported",
           }
    )

    return samples


def compute_f1_precision_recall(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score
    return (
        float(f1_score(y_true, y_pred, zero_division=0)),
        float(precision_score(y_true, y_pred, zero_division=0)),
        float(recall_score(y_true, y_pred, zero_division=0)),
    )


def run_benchmark(samples: list[dict[str, Any]], dataset_name: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"RetrievalVerifier Benchmark — {dataset_name} (n={len(samples)})")
    print(f"{'=' * 70}")

    nli_verifier = NLIVerifier()
    verifier = RetrievalVerifier(nli_verifier=nli_verifier)

    if not verifier.is_available():
        print(
            "  Warning: no search API key configured — running against "
            "OfflineRetriever, which will likely return all-UNSUPPORTED "
            "verdicts on a cold cache."
        )

    y_true: list[int] = []
    y_pred: list[int] = []
    skipped = 0
    latencies: list[float] = []

    for i, sample in enumerate(samples):
        claim_text = sample["claim_text"]
        label = sample["label"]
        if not claim_text:
            skipped += 1
            continue

        claim = Claim(
            id=f"bench_{i}", text=claim_text,
            source_span=(0, len(claim_text)), source_text=claim_text,
        )

        t0 = time.perf_counter()
        try:
            verdict = verifier.verify([claim])[0]
        except Exception:
            import traceback
            print(f"\n===== SAMPLE {i} FAILED =====")
            traceback.print_exc()
            skipped += 1
            continue
        latencies.append(time.perf_counter() - t0)

        true_label = 1 if label == "hallucinated" else 0
        predicted_hallucinated = 1 if verdict.verdict in (Verdict.CONTRADICTED, Verdict.UNSUPPORTED) else 0

        y_true.append(true_label)
        y_pred.append(predicted_hallucinated)

        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(samples)} processed")

    nli_verifier.unload()

    if skipped:
        print(f"  Skipped {skipped} malformed/failed samples")

    if len(set(y_true)) < 2:
        print("  Cannot compute F1 — only one class present in labels")
        return

    f1, precision, recall = compute_f1_precision_recall(y_true, y_pred)
    avg_latency_s = sum(latencies) / len(latencies) if latencies else 0.0

    print(f"\n  Results ({len(y_true)} evaluated samples):")
    print(f"    F1:         {f1:.4f}")
    print(f"    Precision:  {precision:.4f}")
    print(f"    Recall:     {recall:.4f}")
    print(f"    Avg latency/claim: {avg_latency_s:.2f}s")
    print()
    print("  Quality Gate G3: report this F1 against your chosen retrieval-only baseline.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RetrievalVerifier (Quality Gate G3)")
    parser.add_argument("--dataset", choices=["halueval"], default="halueval")
    parser.add_argument("--n", type=int, default=50, help="Number of source samples to load")
    args = parser.parse_args()

    if args.dataset == "halueval":
        try:
            samples = load_halueval_qa(args.n)
            run_benchmark(samples, "HaluEval QA (ungrounded)")
        except FileNotFoundError as e:
            print(f"Warning: {e}")


if __name__ == "__main__":
    main()
