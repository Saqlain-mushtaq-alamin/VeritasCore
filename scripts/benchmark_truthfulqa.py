"""Benchmark VeritasCore against TruthfulQA (multiple-choice).

TruthfulQA has 817 questions, each with multiple answer choices.
The mc1_targets field provides: a list of answer choices and binary labels
(1 = correct/truthful, 0 = incorrect/false).

Evaluation strategy:
  We treat each (question, answer_choice) pair as an NLI instance:
    - context  = the question
    - claim    = the answer choice
    - label    = 1 (hallucinated/false) if mc1_targets.labels == 0 else 0

The NLIVerifier assigns each claim a hallucination score. We measure
AUROC for distinguishing false answers (label=1) from truthful answers (label=0).

This measures whether VeritasCore's NLI signal can detect factually incorrect
answers in a question-answering context.

Usage:
    python scripts/benchmark_truthfulqa.py
    python scripts/benchmark_truthfulqa.py --n 200 --bootstrap-ci --n-bootstrap 1000
    python scripts/benchmark_truthfulqa.py --output tests/benchmarks/results/
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
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from stats_utils import bootstrap_auroc_ci, bootstrap_f1_ci, format_ci  # noqa: E402
from veritascore.core.types import Claim, Verdict  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "datasets"


def load_truthfulqa(n: int = 817) -> list[dict[str, Any]]:
    """Load TruthfulQA multiple-choice samples as (question, answer, label) pairs.

    Returns:
        List of dicts: {context, claim_text, label, question_idx, choice_idx}
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
    total_questions = 0

    for q_idx, row in enumerate(raw_split):
        question = row.get("question", "")
        mc1 = row.get("mc1_targets", {})
        if not mc1 or not question:
            continue

        choices = mc1.get("choices", [])
        labels = mc1.get("labels", [])

        if not choices or not labels or len(choices) != len(labels):
            continue

        total_questions += 1

        for c_idx, (choice, lbl) in enumerate(zip(choices, labels)):
            if not choice:
                continue
            # lbl=1 means truthful/correct; we want to detect false answers
            # so label=1 (hallucinated) when lbl=0 (wrong answer)
            samples.append({
                "context": question,
                "claim_text": choice,
                "label": "hallucinated" if lbl == 0 else "supported",
                "question_idx": q_idx,
                "choice_idx": c_idx,
                "raw_label": lbl,
            })

        if len(samples) >= n:
            break

    print(
        f"  [TruthfulQA] Loaded {len(samples)} (question, answer) pairs "
        f"from {total_questions} questions"
    )
    return samples[:n]


def run_benchmark(
    samples: list[dict[str, Any]],
    bootstrap_ci: bool = False,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run NLI benchmark on TruthfulQA samples."""
    from sklearn.metrics import (  # type: ignore[import-untyped]
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from veritascore.verifier.nli_verifier import NLIVerifier  # noqa: E402

    print(f"\n{'=' * 70}")
    print(f"TruthfulQA Benchmark (n={len(samples)} question-answer pairs)")
    print(f"{'=' * 70}")
    print("  Context  = the question")
    print("  Claim    = each answer choice")
    print("  Label    = hallucinated (incorrect answer) / supported (correct answer)")

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
            id=f"tqa_{i}",
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

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(samples)} processed")

    verifier.unload()

    if skipped:
        print(f"  Skipped {skipped} samples")

    result: dict[str, Any] = {
        "mode": "nli",
        "dataset": "TruthfulQA (MC)",
        "n_samples": len(y_true),
        "n_skipped": skipped,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scoring_formula": "fwd_c - 0.2*fwd_e - 0.6*rev_e",
        "note": (
            "NLI applied to (question, answer_choice) pairs. "
            "AUROC measures ability to rank incorrect choices higher than correct ones."
        ),
    }

    if len(set(y_true)) < 2:
        print("  FAIL: Cannot compute AUROC — only one class in sample")
        result["auroc"] = None
        return result

    result["auroc"] = float(roc_auc_score(y_true, y_score))
    result["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    result["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    result["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    result["avg_latency_ms"] = float(np.mean(latencies) * 1000) if latencies else 0.0
    result["p95_latency_ms"] = float(np.percentile(latencies, 95) * 1000) if latencies else 0.0

    if bootstrap_ci and len(y_true) >= 10:
        print(f"  Computing bootstrapped CIs (n_bootstrap={n_bootstrap})...")
        auc_mean, auc_lo, auc_hi = bootstrap_auroc_ci(y_true, y_score, n_bootstrap, seed=seed)
        f1_mean, f1_lo, f1_hi = bootstrap_f1_ci(y_true, y_pred, n_bootstrap, seed=seed)
        result.update({
            "ci_auroc_mean": auc_mean,
            "ci_auroc_lower": auc_lo,
            "ci_auroc_upper": auc_hi,
            "ci_f1_mean": f1_mean,
            "ci_f1_lower": f1_lo,
            "ci_f1_upper": f1_hi,
            "ci_n_bootstrap": n_bootstrap,
            "ci_level": 0.95,
        })

    # Print results
    print(f"\n  Results ({result['n_samples']} samples):")
    print(f"    AUROC:     {result['auroc']:.4f}", end="")
    if "ci_auroc_mean" in result:
        print(
            f"  95% CI: "
            f"{format_ci(result['ci_auroc_mean'], result['ci_auroc_lower'], result['ci_auroc_upper'])}"
        )
    else:
        print()
    print(f"    F1:        {result['f1']:.4f}")
    print(f"    Precision: {result['precision']:.4f}")
    print(f"    Recall:    {result['recall']:.4f}")
    print(f"    Avg latency: {result['avg_latency_ms']:.1f} ms/pair")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark VeritasCore on TruthfulQA (multiple-choice)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=817, help="Max number of QA pairs to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrapping")
    parser.add_argument("--bootstrap-ci", action="store_true", help="Compute bootstrapped 95% CIs")
    parser.add_argument("--n-bootstrap", type=int, default=2000, help="Bootstrap resamples")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/benchmarks/results"),
        help="Output directory for results JSON",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    try:
        samples = load_truthfulqa(n=args.n)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    result = run_benchmark(
        samples,
        bootstrap_ci=args.bootstrap_ci,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    out_path = args.output / "truthfulqa_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
