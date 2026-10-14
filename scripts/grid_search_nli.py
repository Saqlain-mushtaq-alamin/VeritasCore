"""Phase R3: Grid Search over NLI scoring formulas on HaluEval tuning split.

Searches for optimal coefficients in:
    hallucination_score = fwd_contradiction - beta * fwd_entailment - gamma * rev_entailment

Strictly evaluates on the tuning grid split (n=500, split='grid', seed=42)
to guarantee zero data leakage into the held-out evaluation split (split='eval').

Outputs:
    tests/benchmarks/results/grid_search_results.json
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
from sklearn.metrics import roc_auc_score

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_benchmarks_v2 import load_halueval_split  # noqa: E402
from veritascore.core.types import Claim  # noqa: E402
from veritascore.verifier.nli_verifier import NLIVerifier  # noqa: E402


def run_grid_search(
    n: int = 500,
    seed: int = 42,
    output_path: Path = Path("tests/benchmarks/results/grid_search_results.json"),
) -> dict[str, Any]:
    print("=" * 70)
    print(f"Phase R3: NLI Scoring Formula Grid Search (n={n}, split='grid', seed={seed})")
    print("=" * 70)

    # 1. Load grid split
    samples = load_halueval_split(split="grid", n=n, grid_n=n, seed=seed)
    print(f"Loaded {len(samples)} samples from grid split.")

    # 2. Extract raw NLI probabilities using real model weights
    verifier = NLIVerifier()
    fwd_e_list: list[float] = []
    fwd_c_list: list[float] = []
    rev_e_list: list[float] = []
    y_true: list[int] = []

    print("Extracting forward and reverse NLI signals...")
    t0 = time.perf_counter()

    for i, sample in enumerate(samples):
        context = sample.get("context", "")
        claim_text = sample.get("claim_text", "")
        if not context or not claim_text:
            continue

        claim = Claim(id=f"grid_{i}", text=claim_text, source_span=(0, len(claim_text)), source_text=claim_text)
        verdict = verifier.verify([claim], context=context)[0]

        fwd_e = float(verdict.nli_score or 0.0)
        fwd_c = float(verdict.contradiction_score or 0.0)
        rev_e = float(verdict.reverse_entailment_score or 0.0)

        fwd_e_list.append(fwd_e)
        fwd_c_list.append(fwd_c)
        rev_e_list.append(rev_e)
        y_true.append(1 if sample["label"] == "hallucinated" else 0)

        if (i + 1) % 100 == 0:
            print(f"  ... {i + 1}/{len(samples)} processed")

    verifier.unload()
    print(f"Extraction complete in {time.perf_counter() - t0:.1f}s.")

    fwd_e_arr = np.array(fwd_e_list)
    fwd_c_arr = np.array(fwd_c_list)
    rev_e_arr = np.array(rev_e_list)
    y_true_arr = np.array(y_true)

    # 3. Individual signal baselines
    # Invert entailment since high entailment = NOT hallucinated
    auroc_fwd_e = float(roc_auc_score(y_true_arr, -fwd_e_arr))
    auroc_fwd_c = float(roc_auc_score(y_true_arr, fwd_c_arr))
    auroc_rev_e = float(roc_auc_score(y_true_arr, -rev_e_arr))

    print("\nIndividual Signal AUROCs (Grid Split):")
    print(f"  Raw entailment (-fwd_e):          {auroc_fwd_e:.4f}")
    print(f"  Raw contradiction (fwd_c):        {auroc_fwd_c:.4f}")
    print(f"  Raw reverse entailment (-rev_e):  {auroc_rev_e:.4f}")

    # 4. Grid search over beta and gamma
    # formula: fwd_c - beta * fwd_e - gamma * rev_e
    betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    gammas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

    grid_results = []
    best_auroc = -1.0
    best_config: tuple[float, float] = (0.0, 0.0)

    for beta in betas:
        for gamma in gammas:
            scores = fwd_c_arr - beta * fwd_e_arr - gamma * rev_e_arr
            auc = float(roc_auc_score(y_true_arr, scores))
            entry = {
                "beta": beta,
                "gamma": gamma,
                "formula": f"fwd_c - {beta}*fwd_e - {gamma}*rev_e",
                "auroc": auc,
            }
            grid_results.append(entry)
            if auc > best_auroc:
                best_auroc = auc
                best_config = (beta, gamma)

    # Specific key combinations from benchmarks.md Table 3
    auc_fwd_c_minus_02_e = float(roc_auc_score(y_true_arr, fwd_c_arr - 0.2 * fwd_e_arr))
    auc_selected = float(roc_auc_score(y_true_arr, fwd_c_arr - 0.2 * fwd_e_arr - 0.6 * rev_e_arr))

    print("\nKey Benchmark Formulas (Grid Split):")
    print(f"  fwd_c - 0.2*fwd_e:               {auc_fwd_c_minus_02_e:.4f}")
    print(f"  fwd_c - 0.2*fwd_e - 0.6*rev_e:   {auc_selected:.4f}  <-- Selected in paper")
    print(f"  Optimal grid search point:       {best_auroc:.4f} (beta={best_config[0]}, gamma={best_config[1]})")

    results_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": "HaluEval QA",
        "split": "grid",
        "n_samples": len(y_true),
        "seed": seed,
        "individual_signals": {
            "fwd_entailment": auroc_fwd_e,
            "fwd_contradiction": auroc_fwd_c,
            "rev_entailment": auroc_rev_e,
        },
        "selected_formula": {
            "formula": "fwd_c - 0.2*fwd_e - 0.6*rev_e",
            "beta": 0.2,
            "gamma": 0.6,
            "auroc": auc_selected,
        },
        "intermediate_formula": {
            "formula": "fwd_c - 0.2*fwd_e",
            "auroc": auc_fwd_c_minus_02_e,
        },
        "optimal_grid_point": {
            "beta": best_config[0],
            "gamma": best_config[1],
            "auroc": best_auroc,
        },
        "all_combinations": grid_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
    print(f"\nGrid search results successfully saved → {output_path}")

    return results_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NLI formula grid search on tuning split")
    parser.add_argument("--n", type=int, default=500, help="Size of grid split")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/benchmarks/results/grid_search_results.json"),
    )
    args = parser.parse_args()

    run_grid_search(n=args.n, seed=args.seed, output_path=args.output)
