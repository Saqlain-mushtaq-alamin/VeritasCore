"""Evaluate Forward-Only VeritasCore NLI baseline on HaluEval (n=1,500) and FEVER (n=1,000).

Computes AUROC, F1, and bootstrapped 95% CIs on the exact identical held-out splits,
providing a direct ablation comparison for Table 1 to show the exact contribution
of the bidirectional NLI pass.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent / "src"))

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from run_benchmarks_v2 import load_fever_fixed, load_halueval_split
from stats_utils import bootstrap_auroc_ci, bootstrap_f1_ci, format_ci
from veritascore.core.types import Claim
from veritascore.verifier.nli_verifier import NLIVerifier

RESULTS_DIR = SCRIPTS_DIR.parent / "tests" / "benchmarks" / "results"


def run_forward_only_eval(
    samples: list[dict],
    dataset_name: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    print(f"\n--- Running Forward-Only VeritasCore on {dataset_name} (n={len(samples)}) ---")
    # Initialize verifier with use_bidirectional=False
    verifier = NLIVerifier(use_bidirectional=False)

    y_true = []
    y_score = []
    t0 = time.time()

    for i, s in enumerate(samples):
        claim = Claim(
            id=f"fwd_{i}",
            text=s["claim_text"],
            source_span=(0, len(s["claim_text"])),
            source_text=s["claim_text"],
        )
        ctx = s["context"]
        label = 1 if s["label"] == "hallucinated" else 0
        verdict = verifier.verify([claim], context=ctx)[0]

        # In forward-only mode, contradiction is the primary hallucination score,
        # or (contradiction - entailment)
        score = verdict.contradiction_score - verdict.nli_score
        y_true.append(label)
        y_score.append(score)

        if (i + 1) % 250 == 0 or (i + 1) == len(samples):
            print(f"  Processed {i + 1}/{len(samples)} ({time.time() - t0:.1f}s)")

    verifier.unload()

    y_true_arr = np.array(y_true, dtype=int)
    y_score_arr = np.array(y_score, dtype=float)

    auroc = float(roc_auc_score(y_true_arr, y_score_arr))
    _, auroc_lo, auroc_hi = bootstrap_auroc_ci(y_true_arr, y_score_arr, n_bootstrap=n_bootstrap, seed=seed)

    # Threshold for F1
    best_thresh = float(np.median(y_score_arr))
    y_pred = (y_score_arr >= best_thresh).astype(int)
    f1 = float(f1_score(y_true_arr, y_pred))

    print(f"Result for {dataset_name}:")
    print(f"  AUROC: {format_ci(auroc, auroc_lo, auroc_hi)}")
    print(f"  F1:    {f1:.4f}")

    return {
        "dataset": dataset_name,
        "n_samples": len(samples),
        "auroc": auroc,
        "ci_auroc_lower": auroc_lo,
        "ci_auroc_upper": auroc_hi,
        "f1": f1,
        "y_true": y_true,
        "y_score": y_score,
    }


def main() -> None:
    halu_samples = load_halueval_split(split="eval", n=1500, grid_n=500, seed=42)
    halu_res = run_forward_only_eval(halu_samples, "HaluEval QA")

    fever_samples = load_fever_fixed(n=1000)
    fever_res = run_forward_only_eval(fever_samples, "FEVER (validation)")

    out_file = RESULTS_DIR / "forward_only_nli_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"halueval": halu_res, "fever": fever_res}, f, indent=2)
    print(f"\nSaved forward-only results to: {out_file}")


if __name__ == "__main__":
    main()
