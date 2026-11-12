"""Run paired DeLong test and CIs on difference between Forward-Only (a) and Bidirectional (c).

Computes:
1. AUROC and 95% bootstrap CIs for:
   (a) fwd_c - fwd_e (forward-only)
   (b) fwd_c - 0.2*fwd_e (asymmetric forward-only)
   (c) fwd_c - 0.2*fwd_e - 0.6*rev_e (bidirectional)
2. Paired DeLong test for (a) vs (c), including 95% CI on the AUROC difference.
3. Paired DeLong test for (a) vs (b), including 95% CI on the difference.
4. Paired bootstrap test (n=10,000 resamples).
5. Updates tests/benchmarks/results/table1_results.json and significance_results.json.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from run_benchmarks_v2 import load_fever_fixed, load_halueval_split
from stats_utils import (
    bootstrap_auroc_ci,
    bootstrap_f1_ci,
    delong_test_with_ci,
    format_ci,
    format_p_val,
    paired_bootstrap_test,
)
from veritascore.core.types import Claim, Verdict
from veritascore.verifier.nli_verifier import NLIVerifier

RESULTS_DIR = REPO_ROOT / "tests" / "benchmarks" / "results"


def score_bidirectional_batch(
    samples: list[dict[str, Any]],
    dataset_name: str,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Score samples with NLIVerifier(use_bidirectional=True).

    Returns:
        (y_true, scores_a, scores_b, scores_c)
        where:
          scores_a = fwd_c - fwd_e
          scores_b = fwd_c - 0.2*fwd_e
          scores_c = fwd_c - 0.2*fwd_e - 0.6*rev_e
    """
    print(f"\n[Scoring] Running bidirectional NLI on {dataset_name} ({len(samples)} samples)...")
    verifier = NLIVerifier(use_bidirectional=True)
    t0 = time.time()

    y_true: list[int] = []
    scores_a: list[float] = []
    scores_b: list[float] = []
    scores_c: list[float] = []

    for i, s in enumerate(samples):
        claim = Claim(
            id=f"pair_{i}",
            text=s["claim_text"],
            source_span=(0, len(s["claim_text"])),
            source_text=s["claim_text"],
        )
        verdict = verifier.verify([claim], context=s["context"])[0]

        fwd_e = float(verdict.nli_score or 0.0)
        fwd_c = float(verdict.contradiction_score or 0.0)
        rev_e = float(verdict.reverse_entailment_score or 0.0)

        s_a = fwd_c - fwd_e
        s_b = fwd_c - 0.2 * fwd_e
        s_c = fwd_c - 0.2 * fwd_e - 0.6 * rev_e

        lbl = 1 if s["label"] == "hallucinated" else 0
        y_true.append(lbl)
        scores_a.append(s_a)
        scores_b.append(s_b)
        scores_c.append(s_c)

        if (i + 1) % 250 == 0 or (i + 1) == len(samples):
            print(f"  {i + 1}/{len(samples)} done ({time.time() - t0:.1f}s)")

    verifier.unload()
    return y_true, scores_a, scores_b, scores_c


def run_comparisons_for_dataset(
    dataset_name: str,
    samples: list[dict[str, Any]],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run full formula comparison and paired tests."""
    y_true, scores_a, scores_b, scores_c = score_bidirectional_batch(samples, dataset_name)
    y_true_arr = np.array(y_true, dtype=int)
    scores_a_arr = np.array(scores_a, dtype=float)
    scores_b_arr = np.array(scores_b, dtype=float)
    scores_c_arr = np.array(scores_c, dtype=float)

    # 1. Formula AUROCs & CIs
    print(f"\n[Evaluating] Computing AUROCs and 95% CIs ({n_bootstrap} bootstrap resamples)...")
    mean_a, lo_a, hi_a = bootstrap_auroc_ci(y_true_arr, scores_a_arr, n_bootstrap=n_bootstrap, seed=seed)
    mean_b, lo_b, hi_b = bootstrap_auroc_ci(y_true_arr, scores_b_arr, n_bootstrap=n_bootstrap, seed=seed)
    mean_c, lo_c, hi_c = bootstrap_auroc_ci(y_true_arr, scores_c_arr, n_bootstrap=n_bootstrap, seed=seed)

    pt_a = float(roc_auc_score(y_true_arr, scores_a_arr))
    pt_b = float(roc_auc_score(y_true_arr, scores_b_arr))
    pt_c = float(roc_auc_score(y_true_arr, scores_c_arr))

    # F1 scores
    # Threshold at 0.0 (where contradiction > entailment)
    f1_a = float(f1_score(y_true_arr, (scores_a_arr > 0).astype(int), zero_division=0))
    f1_b = float(f1_score(y_true_arr, (scores_b_arr > 0).astype(int), zero_division=0))
    f1_c = float(f1_score(y_true_arr, (scores_c_arr > 0).astype(int), zero_division=0))

    print(f"  Formula (a) fwd_c - fwd_e:     AUROC = {pt_a:.4f} ({lo_a:.4f}–{hi_a:.4f}) | F1 = {f1_a:.4f}")
    print(f"  Formula (b) fwd_c - 0.2*fwd_e: AUROC = {pt_b:.4f} ({lo_b:.4f}–{hi_b:.4f}) | F1 = {f1_b:.4f}")
    print(f"  Formula (c) bidirectional:     AUROC = {pt_c:.4f} ({lo_c:.4f}–{hi_c:.4f}) | F1 = {f1_c:.4f}")

    # 2. Paired DeLong: (a) vs (c) [Forward-only vs Bidirectional]
    z_ac, p_ac, delta_ac, lo_ac, hi_ac = delong_test_with_ci(y_true_arr, scores_a_arr, scores_c_arr)
    boot_delta_ac, boot_p_ac = paired_bootstrap_test(y_true_arr, scores_a_arr, scores_c_arr, n_bootstrap=n_bootstrap, seed=seed)

    print(f"\n  [Paired Test: (a) vs (c) Bidirectional]")
    print(f"    Delta AUROC: {delta_ac:+.4f} (95% CI: {lo_ac:+.4f} to {hi_ac:+.4f})")
    print(f"    DeLong z = {z_ac:.4f}, p = {p_ac:.4e}")
    print(f"    Paired Bootstrap p = {boot_p_ac:.4f}")

    # 3. Paired DeLong: (a) vs (b) [Forward-only vs Asymmetric]
    z_ab, p_ab, delta_ab, lo_ab, hi_ab = delong_test_with_ci(y_true_arr, scores_a_arr, scores_b_arr)
    boot_delta_ab, boot_p_ab = paired_bootstrap_test(y_true_arr, scores_a_arr, scores_b_arr, n_bootstrap=n_bootstrap, seed=seed)

    print(f"\n  [Paired Test: (a) vs (b) Asymmetric]")
    print(f"    Delta AUROC: {delta_ab:+.4f} (95% CI: {lo_ab:+.4f} to {hi_ab:+.4f})")
    print(f"    DeLong z = {z_ab:.4f}, p = {p_ab:.4e}")

    return {
        "dataset": dataset_name,
        "n_samples": len(y_true),
        "formula_a": {
            "auroc": pt_a,
            "ci_auroc_lower": lo_a,
            "ci_auroc_upper": hi_a,
            "f1": f1_a,
            "y_score": scores_a,
        },
        "formula_b": {
            "auroc": pt_b,
            "ci_auroc_lower": lo_b,
            "ci_auroc_upper": hi_b,
            "f1": f1_b,
            "y_score": scores_b,
        },
        "formula_c": {
            "auroc": pt_c,
            "ci_auroc_lower": lo_c,
            "ci_auroc_upper": hi_c,
            "f1": f1_c,
            "y_score": scores_c,
        },
        "comparison_a_vs_c": {
            "dataset": dataset_name,
            "method_a": "VeritasCore NLI (a) Forward-only (fwd_c - fwd_e)",
            "method_b": "VeritasCore NLI (c) Bidirectional (fwd_c - 0.2*fwd_e - 0.6*rev_e)",
            "auroc_a": pt_a,
            "auroc_b": pt_c,
            "delta_auroc": delta_ac,
            "ci_diff_lower": lo_ac,
            "ci_diff_upper": hi_ac,
            "delong_z": z_ac,
            "delong_p_value": p_ac,
            "paired_bootstrap_delta": boot_delta_ac,
            "paired_bootstrap_p_value": boot_p_ac,
            "statistically_significant_p05": bool(p_ac < 0.05),
        },
        "comparison_a_vs_b": {
            "dataset": dataset_name,
            "method_a": "VeritasCore NLI (a) Forward-only (fwd_c - fwd_e)",
            "method_b": "VeritasCore NLI (b) Asymmetric (fwd_c - 0.2*fwd_e)",
            "auroc_a": pt_a,
            "auroc_b": pt_b,
            "delta_auroc": delta_ab,
            "ci_diff_lower": lo_ab,
            "ci_diff_upper": hi_ab,
            "delong_z": z_ab,
            "delong_p_value": p_ab,
            "paired_bootstrap_delta": boot_delta_ab,
            "paired_bootstrap_p_value": boot_p_ab,
            "statistically_significant_p05": bool(p_ab < 0.05),
        },
        "y_true": y_true,
    }


def main() -> None:
    print("=" * 70)
    print("PAIRED DELONG TEST & CI: FORWARD-ONLY (a) vs BIDIRECTIONAL (c)")
    print("=" * 70)

    # 1. HaluEval QA eval split (n=1500)
    halu_samples = load_halueval_split(split="eval", n=1500, grid_n=500, seed=42)
    halu_res = run_comparisons_for_dataset("HaluEval QA", halu_samples)

    # 2. FEVER validation split (n=1000)
    fever_samples = load_fever_fixed(n=1000)
    fever_res = run_comparisons_for_dataset("FEVER (validation)", fever_samples)

    # Update table1_results.json
    table1_path = RESULTS_DIR / "table1_results.json"
    t1_data = {
        "fwd_c - fwd_e": {
            "halueval": {
                "dataset": "HaluEval QA",
                "n_samples": halu_res["n_samples"],
                "auroc": halu_res["formula_a"]["auroc"],
                "ci_auroc_lower": halu_res["formula_a"]["ci_auroc_lower"],
                "ci_auroc_upper": halu_res["formula_a"]["ci_auroc_upper"],
                "f1": halu_res["formula_a"]["f1"],
                "scoring_formula": "fwd_c - fwd_e",
            },
            "fever": {
                "dataset": "FEVER (validation)",
                "n_samples": fever_res["n_samples"],
                "auroc": fever_res["formula_a"]["auroc"],
                "ci_auroc_lower": fever_res["formula_a"]["ci_auroc_lower"],
                "ci_auroc_upper": fever_res["formula_a"]["ci_auroc_upper"],
                "f1": fever_res["formula_a"]["f1"],
                "scoring_formula": "fwd_c - fwd_e",
            },
        },
        "fwd_c - 0.2*fwd_e": {
            "halueval": {
                "dataset": "HaluEval QA",
                "n_samples": halu_res["n_samples"],
                "auroc": halu_res["formula_b"]["auroc"],
                "ci_auroc_lower": halu_res["formula_b"]["ci_auroc_lower"],
                "ci_auroc_upper": halu_res["formula_b"]["ci_auroc_upper"],
                "f1": halu_res["formula_b"]["f1"],
                "scoring_formula": "fwd_c - 0.2*fwd_e",
            },
            "fever": {
                "dataset": "FEVER (validation)",
                "n_samples": fever_res["n_samples"],
                "auroc": fever_res["formula_b"]["auroc"],
                "ci_auroc_lower": fever_res["formula_b"]["ci_auroc_lower"],
                "ci_auroc_upper": fever_res["formula_b"]["ci_auroc_upper"],
                "f1": fever_res["formula_b"]["f1"],
                "scoring_formula": "fwd_c - 0.2*fwd_e",
            },
        },
        "bidirectional": {
            "halueval": {
                "dataset": "HaluEval QA",
                "n_samples": halu_res["n_samples"],
                "auroc": halu_res["formula_c"]["auroc"],
                "ci_auroc_lower": halu_res["formula_c"]["ci_auroc_lower"],
                "ci_auroc_upper": halu_res["formula_c"]["ci_auroc_upper"],
                "f1": halu_res["formula_c"]["f1"],
                "scoring_formula": "bidirectional",
            },
            "fever": {
                "dataset": "FEVER (validation)",
                "n_samples": fever_res["n_samples"],
                "auroc": fever_res["formula_c"]["auroc"],
                "ci_auroc_lower": fever_res["formula_c"]["ci_auroc_lower"],
                "ci_auroc_upper": fever_res["formula_c"]["ci_auroc_upper"],
                "f1": fever_res["formula_c"]["f1"],
                "scoring_formula": "bidirectional",
            },
        },
    }
    table1_path.write_text(json.dumps(t1_data, indent=2), encoding="utf-8")
    print(f"\n[Saved] Updated {table1_path}")

    # Update significance_results.json
    sig_path = RESULTS_DIR / "significance_results.json"
    existing_sig = []
    if sig_path.exists():
        try:
            existing_sig = json.loads(sig_path.read_text(encoding="utf-8"))
        except Exception:
            existing_sig = []

    # Filter out any old formula comparisons, keeping SummaC comparisons
    summac_comps = [c for c in existing_sig if "SummaC" in c.get("method_b", "")]

    formula_comps = [
        halu_res["comparison_a_vs_c"],
        fever_res["comparison_a_vs_c"],
        halu_res["comparison_a_vs_b"],
        fever_res["comparison_a_vs_b"],
    ]

    all_sig = summac_comps + formula_comps
    sig_path.write_text(json.dumps(all_sig, indent=2), encoding="utf-8")
    print(f"[Saved] Updated {sig_path}")

    # Save detailed formula comparison report
    report_path = RESULTS_DIR / "formula_comparison_report.json"
    report_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "halueval": {
            "n_samples": halu_res["n_samples"],
            "formula_a": {k: v for k, v in halu_res["formula_a"].items() if k != "y_score"},
            "formula_b": {k: v for k, v in halu_res["formula_b"].items() if k != "y_score"},
            "formula_c": {k: v for k, v in halu_res["formula_c"].items() if k != "y_score"},
            "comparison_a_vs_c": halu_res["comparison_a_vs_c"],
            "comparison_a_vs_b": halu_res["comparison_a_vs_b"],
        },
        "fever": {
            "n_samples": fever_res["n_samples"],
            "formula_a": {k: v for k, v in fever_res["formula_a"].items() if k != "y_score"},
            "formula_b": {k: v for k, v in fever_res["formula_b"].items() if k != "y_score"},
            "formula_c": {k: v for k, v in fever_res["formula_c"].items() if k != "y_score"},
            "comparison_a_vs_c": fever_res["comparison_a_vs_c"],
            "comparison_a_vs_b": fever_res["comparison_a_vs_b"],
        },
    }
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"[Saved] Saved detailed report to {report_path}")

    print("\n" + "=" * 70)
    print("SUMMARY OF DELONG FORMULA COMPARISONS")
    print("=" * 70)
    for ds_res in (halu_res, fever_res):
        ds = ds_res["dataset"]
        comp = ds_res["comparison_a_vs_c"]
        comp_b = ds_res["comparison_a_vs_b"]
        print(f"\n{ds}:")
        print(f"  Forward-only (a) vs Bidirectional (c):")
        print(f"    AUROC (a): {comp['auroc_a']:.4f} vs AUROC (c): {comp['auroc_b']:.4f}")
        print(f"    Delta: {comp['delta_auroc']:+.4f} (95% CI: {comp['ci_diff_lower']:+.4f} to {comp['ci_diff_upper']:+.4f})")
        print(f"    DeLong z: {comp['delong_z']:.4f}, p-value: {comp['delong_p_value']:.4e}")
        print(f"    Bootstrap p-value: {comp['paired_bootstrap_p_value']:.4f}")
        print(f"  Forward-only (a) vs Asymmetric (b):")
        print(f"    AUROC (a): {comp_b['auroc_a']:.4f} vs AUROC (b): {comp_b['auroc_b']:.4f}")
        print(f"    Delta: {comp_b['delta_auroc']:+.4f} (95% CI: {comp_b['ci_diff_lower']:+.4f} to {comp_b['ci_diff_upper']:+.4f})")
        print(f"    DeLong z: {comp_b['delong_z']:.4f}, p-value: {comp_b['delong_p_value']:.4e}")


if __name__ == "__main__":
    main()

