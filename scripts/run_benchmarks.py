"""Run full benchmarks on HaluEval and FEVER datasets.

Produces:
- Per-dataset AUROC, F1, Precision, Recall (with optional 95% bootstrapped CIs)
- Comparison with published baselines
- Breakdown by verification mode
- Latency statistics

Fixes (R3):
- FEVER loader now collects exactly n non-NEI samples (was: skip after counting)
- HaluEval split: --split eval uses a separate held-out set from any grid search
- Added --seed, --split, --bootstrap-ci, --n-bootstrap flags

Saves results as JSON to the output directory and generates a markdown
comparison table via generate_report.py.

Usage:
    python scripts/run_benchmarks.py --datasets halueval fever --output tests/benchmarks/results/
    python scripts/run_benchmarks.py --datasets halueval --n 200 --mode nli
    python scripts/run_benchmarks.py --datasets halueval fever --n 1500 --split eval --bootstrap-ci
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))              # for stats_utils
sys.path.insert(0, str(SCRIPTS_DIR.parent / "src"))  # for veritascore

from stats_utils import bootstrap_auroc_ci, bootstrap_f1_ci, format_ci  # noqa: E402
from veritascore.core.types import Claim, Verdict  # noqa: E402

DATA_DIR = SCRIPTS_DIR.parent / "data" / "datasets"

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

def load_halueval_qa(
    n: int,
    split: str = "all",
    grid_n: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load HaluEval QA samples with optional train/eval splitting (R3 fix).

    Args:
        n:      Number of samples desired.
        split:  'all' (no split), 'eval' (held-out set), 'grid' (tuning set).
        grid_n: Size of grid-search split (default 500).
        seed:   Random seed for reproducible splitting.

    Returns:
        List of sample dicts.
    """
    from datasets import load_from_disk

    path = DATA_DIR / "halueval" / "qa_samples"
    if not path.exists():
        raise FileNotFoundError(
            f"HaluEval QA not found at {path}. "
            "Run: python scripts/download_datasets.py --only halueval"
        )

    ds = load_from_disk(str(path))
    raw_split = ds["data"] if "data" in ds else next(iter(ds.values()))

    # Load enough rows to cover both splits
    max_load = max(grid_n + n + 200, 2500) if split != "all" else n
    all_samples: list[dict[str, Any]] = []
    for i, row in enumerate(raw_split):
        if i >= max_load:
            break
        context = row.get("knowledge", "")
        question = row.get("question", "")
        answer = row.get("answer", "")
        hallucination = str(row.get("hallucination", "")).strip().lower()
        if not context or not answer:
            continue
        claim_text = f"Q: {question}  A: {answer}" if question else answer
        all_samples.append({
            "context": context,
            "claim_text": claim_text,
            "label": "hallucinated" if hallucination == "yes" else "supported",
        })

    if split == "all":
        return all_samples[:n]

    # Deterministic split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_samples))
    grid_idx = set(indices[:grid_n].tolist())
    eval_idx = set(indices[grid_n : grid_n + n].tolist())
    assert len(grid_idx & eval_idx) == 0, "BUG: grid and eval sets overlap!"

    if split == "grid":
        chosen = [all_samples[i] for i in indices[:grid_n]]
        print(f"  [HaluEval] Grid-search split: n={len(chosen)} (seed={seed})")
    else:  # eval
        chosen = [all_samples[i] for i in indices[grid_n : grid_n + n]]
        print(f"  [HaluEval] Held-out eval split: n={len(chosen)} (seed={seed}, disjoint from grid)")

    return chosen


def load_fever(n: int) -> list[dict[str, Any]]:
    """Load exactly n non-NEI FEVER validation samples (R3 bug fix).

    BUG FIX: The original version iterated over exactly n rows and then
    skipped NEI rows — meaning the actual eval set was often ~133 samples
    when requesting n=200. This version iterates until exactly n non-NEI
    samples are collected.
    """
    from datasets import load_from_disk

    path = DATA_DIR / "fever" / "v1.0"
    if not path.exists():
        raise FileNotFoundError(
            f"FEVER not found at {path}. "
            "Run: python scripts/download_datasets.py --only fever"
        )
    ds = load_from_disk(str(path))
    raw_split = ds["validation"] if "validation" in ds else next(iter(ds.values()))

    samples: list[dict[str, Any]] = []
    skipped_nei = 0
    skipped_no_evidence = 0

    for row in raw_split:
        if len(samples) >= n:  # collect EXACTLY n, not skip-then-count
            break
        label = row.get("label", "")
        if label == "NOT ENOUGH INFO":
            skipped_nei += 1
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
            skipped_no_evidence += 1
            continue
        samples.append({
            "context": context,
            "claim_text": row.get("claim", ""),
            "label": "supported" if label == "SUPPORTS" else "hallucinated",
        })

    print(
        f"  [FEVER] Collected {len(samples)} non-NEI samples "
        f"(skipped NEI={skipped_nei}, no-evidence={skipped_no_evidence})"
    )
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
    bootstrap_ci: bool = False,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run NLI-only verification on a sample list, return metric dict.

    Args:
        samples:      Sample list with context, claim_text, label.
        dataset_name: Display name for the dataset.
        bootstrap_ci: Whether to compute bootstrapped 95% CIs.
        n_bootstrap:  Number of bootstrap resamples (≥ 2000 for publication).
        seed:         Random seed for bootstrapping.
    """
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
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()

    print(f"\n  AUROC: {metrics['auroc']:.4f}")
    print(f"  F1:    {metrics['f1']:.4f}")
    print(f"  Avg latency: {metrics['avg_latency_ms']:.1f} ms/claim")

    # Bootstrapped CIs
    if bootstrap_ci and len(y_true) >= 10:
        print(f"  Computing bootstrapped CIs (n_bootstrap={n_bootstrap})...")
        auc_mean, auc_lo, auc_hi = bootstrap_auroc_ci(y_true, y_score, n_bootstrap, seed=seed)
        f1_mean, f1_lo, f1_hi = bootstrap_f1_ci(y_true, y_pred, n_bootstrap, seed=seed)
        metrics["ci_auroc_mean"] = auc_mean
        metrics["ci_auroc_lower"] = auc_lo
        metrics["ci_auroc_upper"] = auc_hi
        metrics["ci_f1_mean"] = f1_mean
        metrics["ci_f1_lower"] = f1_lo
        metrics["ci_f1_upper"] = f1_hi
        metrics["ci_n_bootstrap"] = n_bootstrap
        metrics["ci_level"] = 0.95
        print(f"  AUROC 95% CI: {format_ci(auc_mean, auc_lo, auc_hi)}")
        print(f"  F1    95% CI: {format_ci(f1_mean, f1_lo, f1_hi)}")

    return metrics


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VeritasCore benchmarks on HaluEval and FEVER",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for HaluEval splitting and bootstrapping",
    )
    parser.add_argument(
        "--split",
        choices=["grid", "eval", "all"],
        default="all",
        help="HaluEval split: 'eval' = held-out (R3), 'grid' = tuning only, 'all' = no split",
    )
    parser.add_argument(
        "--bootstrap-ci",
        action="store_true",
        help="Compute bootstrapped 95%% CIs for all metrics",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap resamples",
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
            samples = load_halueval_qa(args.n, split=args.split, seed=args.seed)
            if args.mode in ("nli", "all"):
                result = run_nli_benchmark(
                    samples, "HaluEval QA",
                    bootstrap_ci=args.bootstrap_ci,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                )
                result["split"] = args.split
                result["seed"] = args.seed
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
                result = run_nli_benchmark(
                    samples, "FEVER (validation)",
                    bootstrap_ci=args.bootstrap_ci,
                    n_bootstrap=args.n_bootstrap,
                    seed=args.seed,
                )
                result["seed"] = args.seed
                all_results.append(result)
                out_path = args.output / "fever_results.json"
                out_path.write_text(json.dumps(result, indent=2))
                print(f"\n  Results saved → {out_path}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if all_results:
        summary = {
            "results": all_results,
            "baselines": BASELINES,
            "config": {
                "n": args.n,
                "seed": args.seed,
                "split": args.split,
                "bootstrap_ci": args.bootstrap_ci,
                "n_bootstrap": args.n_bootstrap,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        summary_path = args.output / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary saved → {summary_path}")
        print("\nRun `python scripts/generate_report.py` to generate the comparison table.")
        print("For full R3 evaluation use: python scripts/run_benchmarks_v2.py")


if __name__ == "__main__":
    main()
