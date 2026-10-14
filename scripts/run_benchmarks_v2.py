"""VeritasCore Scale Evaluation — Phase R3 unified benchmark runner.

Fixes vs. run_benchmarks.py:
  - Train/test separation: HaluEval split into grid-search (n=500) and
    held-out eval (n=1500) sets; never overlap.
  - FEVER fix: collects exactly n non-NEI samples (not n rows then skips).
  - Bootstrapped 95% CIs on every metric (AUROC and F1).
  - DeLong's test and paired bootstrap for AUROC significance.
  - TruthfulQA benchmark support.
  - All results saved as structured JSON with full metadata.
  - Fixed random seed throughout for reproducibility.

Usage:
    # Full-scale benchmark with CIs and significance tests
    python scripts/run_benchmarks_v2.py \\
        --datasets halueval fever \\
        --n 1500 \\
        --seed 42 \\
        --bootstrap-ci \\
        --n-bootstrap 2000 \\
        --split eval \\
        --output tests/benchmarks/results/

    # Quick smoke-test (small n, fewer bootstrap resamples)
    python scripts/run_benchmarks_v2.py \\
        --datasets halueval fever \\
        --n 50 --seed 42 --bootstrap-ci --n-bootstrap 100 --split eval

    # TruthfulQA
    python scripts/run_benchmarks_v2.py --datasets truthfulqa --n 817
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

# Make sure the scripts/ dir is importable and veritascore src is on path
SCRIPTS_DIR = Path(__file__).parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))          # for stats_utils
sys.path.insert(0, str(REPO_ROOT / "src"))    # for veritascore

from stats_utils import (  # noqa: E402
    bootstrap_auroc_ci,
    bootstrap_f1_ci,
    delong_test,
    format_ci,
    paired_bootstrap_test,
)
from veritascore.core.types import Claim, Verdict  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "datasets"

# Published baselines: Only empirical re-runs or explicitly cited published baselines
# using identical datasets and protocols are included. Cross-dataset estimates
# (e.g., SelfCheckGPT on WikiBio, FActScore on biographical LM output) are purged.
BASELINES: dict[str, dict[str, float]] = {
    # Populated dynamically from empirical re-runs in results directory
}


# ---------------------------------------------------------------------------
# Dataset loaders — fixed versions
# ---------------------------------------------------------------------------

def split_halueval(
    samples: list[dict[str, Any]],
    grid_n: int = 500,
    eval_n: int = 1500,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split HaluEval into non-overlapping grid-search and held-out eval sets.

    Args:
        samples:  Full list of HaluEval samples.
        grid_n:   Number of samples for grid search / hyperparameter tuning.
        eval_n:   Number of samples for held-out evaluation.
        seed:     Random seed for permutation.

    Returns:
        (grid_set, eval_set) — disjoint subsets.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(samples))
    grid_idx = indices[:grid_n]
    eval_idx = indices[grid_n : grid_n + eval_n]

    # Verify disjointness
    assert len(set(grid_idx) & set(eval_idx)) == 0, "BUG: grid and eval sets overlap!"

    grid_set = [samples[i] for i in grid_idx]
    eval_set = [samples[i] for i in eval_idx]
    return grid_set, eval_set


def load_halueval_qa_all(max_n: int = 10000) -> list[dict[str, Any]]:
    """Load up to max_n HaluEval QA samples (raw, unsplit).

    Uses sequential indexing so the split is reproducible via seed.
    """
    from datasets import load_from_disk  # type: ignore[import-untyped]

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
        if i >= max_n:
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


def load_halueval_split(
    split: str = "eval",
    n: int = 1500,
    grid_n: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load HaluEval with proper train/test separation.

    Args:
        split:  'grid' (hyperparameter tuning), 'eval' (held-out), or 'all'.
        n:      Desired size of the chosen split.
        grid_n: Size of the grid-search split (default 500).
        seed:   Reproducibility seed.

    Returns:
        List of sample dicts.
    """
    # Load enough to cover both splits
    raw = load_halueval_qa_all(max_n=max(grid_n + n + 200, 2500))

    if split == "all":
        return raw[:n]

    grid_set, eval_set = split_halueval(raw, grid_n=grid_n, eval_n=n, seed=seed)

    if split == "grid":
        samples = grid_set
        print(f"  [HaluEval] Grid-search split: n={len(samples)} (seed={seed})")
    else:  # eval
        samples = eval_set
        print(f"  [HaluEval] Held-out eval split: n={len(samples)} (seed={seed}, disjoint from grid)")

    return samples


def load_fever_fixed(n: int = 1000) -> list[dict[str, Any]]:
    """Load exactly n non-NEI FEVER validation samples.

    BUG FIX vs. original: the original iterated over rows 0..n and THEN
    skipped NEI rows, producing fewer than n samples. This version
    iterates until it has collected exactly n non-NEI samples.
    """
    from datasets import load_from_disk  # type: ignore[import-untyped]

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
        if len(samples) >= n:
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


def load_truthfulqa(n: int = 817) -> list[dict[str, Any]]:
    """Load TruthfulQA multiple-choice samples (Exploratory Negative Probe).

    Note on validity: In TruthfulQA, questions cannot semantically entail answers.
    Using question as premise and answer as hypothesis tests prior bias / consistency
    rather than factual entailment against reference evidence. Results must be
    reported strictly as an exploratory negative probe highlighting NLI limits
    without external retrieval.
    Also collects complete question choice sets so no question is truncated mid-choice.
    """
    from datasets import load_from_disk  # type: ignore[import-untyped]

    path = DATA_DIR / "truthfulqa" / "multiple_choice"
    if not path.exists():
        raise FileNotFoundError(
            f"TruthfulQA not found at {path}. "
            "Run: python scripts/download_datasets.py --only truthfulqa"
        )

    ds = load_from_disk(str(path))
    raw_split = ds["validation"] if "validation" in ds else next(iter(ds.values()))

    samples: list[dict[str, Any]] = []
    questions_loaded = 0
    for row in raw_split:
        question = row.get("question", "")
        mc1 = row.get("mc1_targets", {})
        if not mc1 or not question:
            continue
        choices = mc1.get("choices", [])
        labels = mc1.get("labels", [])
        if not choices or not labels:
            continue

        q_samples: list[dict[str, Any]] = []
        for choice, lbl in zip(choices, labels):
            if not choice:
                continue
            q_samples.append({
                "context": question,
                "claim_text": choice,
                "label": "supported" if lbl == 1 else "hallucinated",
            })

        if not q_samples:
            continue

        # Keep whole question sets together to avoid mid-question truncation
        if len(samples) + len(q_samples) > n and len(samples) > 0:
            break

        samples.extend(q_samples)
        questions_loaded += 1

    print(
        f"  [TruthfulQA] [Exploratory Negative Probe] Loaded {len(samples)} "
        f"(question, answer) pairs from {questions_loaded} complete questions"
    )
    return samples


# ---------------------------------------------------------------------------
# NLI benchmark runner — returns structured dict
# ---------------------------------------------------------------------------

def run_nli_benchmark(
    samples: list[dict[str, Any]],
    dataset_name: str,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run NLI-only verification and return a structured result dict.

    Args:
        samples:      List of {'context', 'claim_text', 'label'} dicts.
        dataset_name: Display name for the dataset.
        bootstrap_ci: Whether to compute bootstrapped CIs.
        n_bootstrap:  Number of bootstrap resamples.
        seed:         Random seed for bootstrapping.

    Returns:
        dict with keys: auroc, f1, precision, recall, n_samples, n_skipped,
        avg_latency_ms, p95_latency_ms, mode, dataset, [ci_auroc_*], [ci_f1_*]
    """
    from sklearn.metrics import (  # type: ignore[import-untyped]
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from veritascore.verifier.nli_verifier import NLIVerifier  # noqa: E402

    print(f"\n{'=' * 70}")
    print(f"NLI Benchmark — {dataset_name} (n={len(samples)})")
    print(f"{'=' * 70}")

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
        # Scoring formula (grid-searched on separate split per R3)
        hallucination_score = fwd_c - 0.2 * fwd_e - 0.6 * rev_e

        y_true.append(1 if sample["label"] == "hallucinated" else 0)
        y_score.append(hallucination_score)
        y_pred.append(
            1 if verdict.verdict in (Verdict.CONTRADICTED, Verdict.UNSUPPORTED) else 0
        )

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(samples)} processed")

    verifier.unload()

    if skipped:
        print(f"  Skipped {skipped} samples")

    result: dict[str, Any] = {
        "mode": "nli",
        "dataset": dataset_name,
        "n_samples": len(y_true),
        "n_skipped": skipped,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scoring_formula": "fwd_c - 0.2*fwd_e - 0.6*rev_e",
    }

    if len(set(y_true)) < 2:
        print("  FAIL: Cannot compute AUROC — only one class present")
        result["auroc"] = None
        result["f1"] = None
        return result

    result["auroc"] = float(roc_auc_score(y_true, y_score))
    result["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    result["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    result["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

    if latencies:
        result["avg_latency_ms"] = float(np.mean(latencies) * 1000)
        result["p95_latency_ms"] = float(np.percentile(latencies, 95) * 1000)
    else:
        result["avg_latency_ms"] = 0.0
        result["p95_latency_ms"] = 0.0

    # Bootstrapped CIs
    if bootstrap_ci and len(y_true) >= 10:
        print(f"  Computing bootstrapped CIs (n_bootstrap={n_bootstrap})...")
        auc_mean, auc_lo, auc_hi = bootstrap_auroc_ci(y_true, y_score, n_bootstrap, seed=seed)
        f1_mean, f1_lo, f1_hi = bootstrap_f1_ci(y_true, y_pred, n_bootstrap, seed=seed)
        result["ci_auroc_mean"] = auc_mean
        result["ci_auroc_lower"] = auc_lo
        result["ci_auroc_upper"] = auc_hi
        result["ci_f1_mean"] = f1_mean
        result["ci_f1_lower"] = f1_lo
        result["ci_f1_upper"] = f1_hi
        result["ci_n_bootstrap"] = n_bootstrap
        result["ci_level"] = 0.95

    # Store raw arrays for significance testing
    result["_y_true"] = [int(v) for v in y_true]
    result["_y_score"] = [float(v) for v in y_score]
    result["_y_pred"] = [int(v) for v in y_pred]

    # Print summary
    print(f"\n  Results ({result['n_samples']} samples, {skipped} skipped):")
    print(f"    AUROC:     {result['auroc']:.4f}", end="")
    if "ci_auroc_mean" in result:
        print(f"  95% CI: {format_ci(result['ci_auroc_mean'], result['ci_auroc_lower'], result['ci_auroc_upper'])}")
    else:
        print()
    print(f"    F1:        {result['f1']:.4f}", end="")
    if "ci_f1_mean" in result:
        print(f"  95% CI: {format_ci(result['ci_f1_mean'], result['ci_f1_lower'], result['ci_f1_upper'])}")
    else:
        print()
    print(f"    Precision: {result.get('precision', float('nan')):.4f}")
    print(f"    Recall:    {result.get('recall', float('nan')):.4f}")
    print(f"    Avg latency: {result['avg_latency_ms']:.1f} ms/claim")
    print(f"    P95 latency: {result['p95_latency_ms']:.1f} ms/claim")

    return result


# ---------------------------------------------------------------------------
# Significance testing across result pairs
# ---------------------------------------------------------------------------

def run_significance_tests(
    results: list[dict[str, Any]],
    results_dir: Path | None = None,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run DeLong's test and paired bootstrap between VeritasCore and SummaC baseline.

    Args:
        results:     List of VeritasCore result dicts (with _y_true, _y_score).
        results_dir: Directory containing baseline result JSON files.
        n_bootstrap: Number of bootstrap iterations for paired bootstrap test.
        seed:        Random seed.

    Returns:
        List of paired statistical comparison dicts with z-stat, p-values, CIs.
    """
    comparisons: list[dict[str, Any]] = []

    dataset_baseline_files = {
        "HaluEval QA": [
            "halueval_qa_summac_oracle_results.json",
            "halueval_summac_oracle_results.json",
        ],
        "FEVER (validation)": [
            "fever_validation_summac_oracle_results.json",
            "fever_summac_oracle_results.json",
        ],
    }

    for dataset, baseline_filenames in dataset_baseline_files.items():
        vc_results = [r for r in results if r.get("dataset") == dataset and ("_y_true" in r or "y_true" in r)]
        if not vc_results:
            continue
        vc = vc_results[0]
        y_true_vc = np.array(vc.get("_y_true", vc.get("y_true", [])))
        y_score_vc = np.array(vc.get("_y_score", vc.get("y_score", [])))

        print(f"\n{'=' * 70}")
        print(f"Paired Significance Test — {dataset}")
        print(f"{'=' * 70}")
        print(f"  VeritasCore AUROC: {vc.get('auroc', 'N/A'):.4f} (n={len(y_true_vc)})")

        baseline_file: Path | None = None
        if results_dir is not None:
            for bf_name in baseline_filenames:
                p = results_dir / bf_name
                if p.exists():
                    baseline_file = p
                    break

        if baseline_file is None:
            msg = (
                f"SummaC baseline file not found in {results_dir} "
                f"({', '.join(baseline_filenames)}). "
                f"Run: python scripts/baseline_summac.py --dataset {dataset.split()[0].lower()} "
                f"--n {len(y_true_vc)} --output {results_dir}"
            )
            print(f"  [NOTE] {msg}")
            comparisons.append({
                "dataset": dataset,
                "method_a": "VeritasCore (NLI)",
                "auroc_a": vc.get("auroc"),
                "method_b": "SummaC ZS (not found)",
                "status": "baseline_not_found",
                "note": msg,
            })
            continue

        try:
            baseline_data = json.loads(baseline_file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [ERROR] Failed to read {baseline_file}: {e}")
            continue

        y_true_b = np.array(baseline_data.get("y_true", baseline_data.get("_y_true", [])))
        y_score_b = np.array(baseline_data.get("y_score", baseline_data.get("_y_score", [])))

        if len(y_true_b) == 0 or len(y_score_b) == 0:
            msg = (
                f"Baseline file {baseline_file.name} does not contain raw 'y_true'/'y_score'. "
                "Re-run scripts/baseline_summac.py to generate paired raw scores."
            )
            print(f"  [WARN] {msg}")
            comparisons.append({
                "dataset": dataset,
                "method_a": "VeritasCore (NLI)",
                "auroc_a": vc.get("auroc"),
                "method_b": "SummaC ZS (reimplemented)",
                "auroc_b": baseline_data.get("auroc"),
                "status": "raw_scores_missing",
                "note": msg,
            })
            continue

        if len(y_true_vc) != len(y_true_b):
            msg = (
                f"Sample size mismatch: VeritasCore n={len(y_true_vc)} vs "
                f"SummaC n={len(y_true_b)}. Cannot run paired test. "
                "Re-run with matching --n."
            )
            print(f"  [WARN] {msg}")
            comparisons.append({
                "dataset": dataset,
                "method_a": "VeritasCore (NLI)",
                "auroc_a": vc.get("auroc"),
                "method_b": "SummaC ZS (reimplemented)",
                "auroc_b": baseline_data.get("auroc"),
                "status": "sample_mismatch",
                "note": msg,
            })
            continue

        if not np.array_equal(y_true_vc, y_true_b):
            msg = (
                "Label mismatch between VeritasCore and baseline samples! "
                "Runs must use identical seeds, splits, and order for paired testing."
            )
            print(f"  [WARN] {msg}")
            comparisons.append({
                "dataset": dataset,
                "method_a": "VeritasCore (NLI)",
                "auroc_a": vc.get("auroc"),
                "method_b": "SummaC ZS (reimplemented)",
                "auroc_b": baseline_data.get("auroc"),
                "status": "label_mismatch",
                "note": msg,
            })
            continue

        # Execute DeLong's test and paired bootstrap
        delong_res = delong_test(y_true_vc, y_score_vc, y_score_b)
        paired_boot = paired_bootstrap_test(
            y_true_vc, y_score_vc, y_score_b, n_resamples=n_bootstrap, seed=seed
        )

        auroc_vc = float(vc.get("auroc", 0.0))
        auroc_b = float(baseline_data.get("auroc", 0.0))
        delta_auroc = auroc_vc - auroc_b

        comp = {
            "dataset": dataset,
            "method_a": "VeritasCore (NLI)",
            "auroc_a": auroc_vc,
            "method_b": "SummaC ZS (reimplemented, DeBERTa-v3)",
            "auroc_b": auroc_b,
            "delta_auroc": float(delta_auroc),
            "delong_z": float(delong_res["z"]),
            "delong_p_value": float(delong_res["p_value"]),
            "delong_ci_diff": [float(x) for x in delong_res.get("ci_diff", (0, 0, 0))],
            "paired_bootstrap_p_value": float(paired_boot["p_value"]),
            "n_samples": len(y_true_vc),
            "statistically_significant_p05": bool(delong_res["p_value"] < 0.05),
        }
        comparisons.append(comp)

        print(f"  SummaC ZS AUROC:       {auroc_b:.4f}")
        print(f"  AUROC Delta (VC-Base): {delta_auroc:+.4f}")
        print(f"  DeLong z-statistic:    {delong_res['z']:.4f}")
        print(f"  DeLong p-value:        {delong_res['p_value']:.4e}")
        if delong_res.get("ci_diff"):
            mean_d, lo_d, hi_d = delong_res["ci_diff"]
            print(f"  DeLong 95% CI on Diff: {format_ci(mean_d, lo_d, hi_d)}")
        print(f"  Paired Bootstrap p:    {paired_boot['p_value']:.4e}")
        print(f"  Significant (p < 0.05): {'YES' if delong_res['p_value'] < 0.05 else 'NO'}")

    return comparisons


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VeritasCore Phase R3: Scale Up Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["halueval", "fever", "truthfulqa"],
        default=["halueval", "fever"],
        help="Datasets to benchmark",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1500,
        help="Number of held-out eval samples (HaluEval) or non-NEI samples (FEVER)",
    )
    parser.add_argument(
        "--grid-n",
        type=int,
        default=500,
        help="Size of the HaluEval grid-search split (must be disjoint from eval)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting and bootstrapping",
    )
    parser.add_argument(
        "--split",
        choices=["grid", "eval", "all"],
        default="eval",
        help="HaluEval split to use: 'eval' (held-out), 'grid' (tune only), 'all' (no split)",
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
        help="Number of bootstrap resamples (≥ 2000 recommended for publication)",
    )
    parser.add_argument(
        "--significance-test",
        action="store_true",
        help="Run DeLong's test and paired bootstrap between result pairs",
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

    print("\n" + "=" * 70)
    print("VeritasCore Phase R3: Scale-Up Evaluation")
    print(f"  Datasets:    {', '.join(args.datasets)}")
    print(f"  n:           {args.n}")
    print(f"  seed:        {args.seed}")
    print(f"  split:       {args.split}")
    print(f"  bootstrap:   {args.bootstrap_ci} (n_bootstrap={args.n_bootstrap})")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # HaluEval
    # -----------------------------------------------------------------------
    if "halueval" in args.datasets:
        try:
            print("\n[HaluEval] Loading samples...")
            samples = load_halueval_split(
                split=args.split,
                n=args.n,
                grid_n=args.grid_n,
                seed=args.seed,
            )
            result = run_nli_benchmark(
                samples,
                "HaluEval QA",
                bootstrap_ci=args.bootstrap_ci,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            # Add split metadata
            result["split"] = args.split
            result["grid_n"] = args.grid_n
            result["seed"] = args.seed
            all_results.append(result)

            # Save full results with raw scores for paired significance testing
            save_result = {k: v for k, v in result.items() if not k.startswith("_")}
            save_result["y_true"] = result.get("_y_true", [])
            save_result["y_score"] = result.get("_y_score", [])
            save_result["y_pred"] = result.get("_y_pred", [])
            out_path = args.output / "halueval_results_v2.json"
            out_path.write_text(json.dumps(save_result, indent=2))
            print(f"\n  Results saved → {out_path}")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    # -----------------------------------------------------------------------
    # FEVER (fixed NEI collection)
    # -----------------------------------------------------------------------
    if "fever" in args.datasets:
        try:
            print("\n[FEVER] Loading samples (collecting exactly n non-NEI)...")
            samples = load_fever_fixed(n=args.n)
            result = run_nli_benchmark(
                samples,
                "FEVER (validation)",
                bootstrap_ci=args.bootstrap_ci,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            result["seed"] = args.seed
            all_results.append(result)

            save_result = {k: v for k, v in result.items() if not k.startswith("_")}
            save_result["y_true"] = result.get("_y_true", [])
            save_result["y_score"] = result.get("_y_score", [])
            save_result["y_pred"] = result.get("_y_pred", [])
            out_path = args.output / "fever_results_v2.json"
            out_path.write_text(json.dumps(save_result, indent=2))
            print(f"\n  Results saved → {out_path}")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    # -----------------------------------------------------------------------
    # TruthfulQA
    # -----------------------------------------------------------------------
    if "truthfulqa" in args.datasets:
        try:
            print("\n[TruthfulQA] Loading samples...")
            samples = load_truthfulqa(n=args.n)
            result = run_nli_benchmark(
                samples,
                "TruthfulQA (MC) [Exploratory Negative Probe]",
                bootstrap_ci=args.bootstrap_ci,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
            result["seed"] = args.seed
            all_results.append(result)

            save_result = {k: v for k, v in result.items() if not k.startswith("_")}
            save_result["y_true"] = result.get("_y_true", [])
            save_result["y_score"] = result.get("_y_score", [])
            save_result["y_pred"] = result.get("_y_pred", [])
            out_path = args.output / "truthfulqa_results_v2.json"
            out_path.write_text(json.dumps(save_result, indent=2))
            print(f"\n  Results saved → {out_path}")
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    # -----------------------------------------------------------------------
    # Significance tests
    # -----------------------------------------------------------------------
    sig_comparisons: list[dict[str, Any]] = []
    if args.significance_test:
        print("\n" + "-" * 70)
        print("Significance Tests (vs. SummaC Baseline)")
        print("-" * 70)
        sig_comparisons = run_significance_tests(
            all_results,
            results_dir=args.output,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )

    # -----------------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------------
    if all_results:
        save_results = [{k: v for k, v in r.items() if not k.startswith("_")} for r in all_results]
        # Summary omits raw arrays to keep file size reasonable
        summary_results = [
            {k: v for k, v in r.items() if k not in ("y_true", "y_score", "y_pred")}
            for r in save_results
        ]
        summary = {
            "phase": "R3",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "datasets": args.datasets,
                "n": args.n,
                "grid_n": args.grid_n,
                "seed": args.seed,
                "split": args.split,
                "bootstrap_ci": args.bootstrap_ci,
                "n_bootstrap": args.n_bootstrap,
            },
            "results": summary_results,
            "significance_tests": sig_comparisons,
        }
        summary_path = args.output / "summary_v2.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary saved → {summary_path}")

        # Print comparison table
        print("\n" + "=" * 70)
        print("COMPARISON TABLE")
        print("=" * 70)
        header = f"{'Method':<35} {'HaluEval AUROC':>18} {'FEVER AUROC':>14}"
        print(header)
        print("-" * 70)

        # Check for empirical SummaC baseline in output dir
        summac_hu_file = args.output / "halueval_qa_summac_oracle_results.json"
        if not summac_hu_file.exists():
            summac_hu_file = args.output / "halueval_summac_oracle_results.json"
        summac_fv_file = args.output / "fever_validation_summac_oracle_results.json"
        if not summac_fv_file.exists():
            summac_fv_file = args.output / "fever_summac_oracle_results.json"

        summac_hu_str = "N/A"
        if summac_hu_file.exists():
            try:
                s_data = json.loads(summac_hu_file.read_text(encoding="utf-8"))
                summac_hu_str = f"{s_data['auroc']:.4f}"
                if "ci_auroc_lower" in s_data:
                    summac_hu_str += f" ({s_data['ci_auroc_lower']:.3f}-{s_data['ci_auroc_upper']:.3f})"
            except Exception:
                pass

        summac_fv_str = "N/A"
        if summac_fv_file.exists():
            try:
                s_data = json.loads(summac_fv_file.read_text(encoding="utf-8"))
                summac_fv_str = f"{s_data['auroc']:.4f}"
                if "ci_auroc_lower" in s_data:
                    summac_fv_str += f" ({s_data['ci_auroc_lower']:.3f}-{s_data['ci_auroc_upper']:.3f})"
            except Exception:
                pass

        print(f"  {'SummaC ZS (Empirical Re-run)':<33} {summac_hu_str:>18} {summac_fv_str:>14}")
        print("-" * 70)

        # VeritasCore results
        for r in summary_results:
            auroc_str = f"{r['auroc']:.4f}" if r.get("auroc") is not None else "N/A"
            if "ci_auroc_lower" in r:
                auroc_str = (
                    f"{r['ci_auroc_mean']:.4f} "
                    f"({r['ci_auroc_lower']:.4f}–{r['ci_auroc_upper']:.4f})"
                )
            ds = r.get("dataset", "?")
            if "HaluEval" in ds:
                print(f"  {'VeritasCore (NLI) [eval split]':<33} {auroc_str:>18}")
            elif "FEVER" in ds:
                print(f"  {'VeritasCore (NLI)':<33} {'':>18} {auroc_str:>14}")
            elif "TruthfulQA" in ds:
                print(f"  {'VeritasCore (NLI) [TruthfulQA]':<33} {auroc_str}")

        print("=" * 70)
        print("\n✓ Phase R3 evaluation complete.")
        print("  Next: run significance tests with --significance-test flag")
        print("        run ablation study (Phase R5)\n")
        print("        run ablation study (Phase R5)\n")


if __name__ == "__main__":
    main()
