"""Compute Inter-Annotator Agreement (IAA) for VeritasCore human evaluation.

Computes:
- Cohen's Kappa for all annotator pairs
- Fleiss's Kappa across all raters
- Observed percent agreement

Quality and Integrity Constraint:
Refuses to run if any two submitted annotation files are byte-identical,
preventing placeholder or copied CSVs from producing fake agreement metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def check_files_distinct(file_paths: list[Path]) -> None:
    """Verify that all submitted annotation files are non-empty and byte-distinct.

    Raises:
        FileNotFoundError: If any file does not exist.
        ValueError: If any file is empty or any two files are byte-identical.
    """
    hashes: dict[str, Path] = {}
    for p in file_paths:
        if not p.exists():
            raise FileNotFoundError(f"Annotation file not found: {p}")
        data = p.read_bytes()
        if len(data.strip()) == 0:
            raise ValueError(f"Annotation file is empty: {p}")
        h = hashlib.sha256(data).hexdigest()
        if h in hashes:
            raise ValueError(
                f"Integrity check failed: File '{p.name}' is byte-identical to '{hashes[h].name}'. "
                "Refusing to compute IAA on copied/unfilled template files. "
                "Distinct submissions from independent annotators are required."
            )
        hashes[h] = p


def compute_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Compute Fleiss's Kappa for m raters classifying N subjects into k categories.

    Args:
        ratings_matrix: Array of shape (N, k) where ratings_matrix[i, j]
            is the number of raters who assigned subject i to category j.

    Returns:
        Fleiss's Kappa coefficient in [-1, 1].
    """
    N, k = ratings_matrix.shape
    n = np.sum(ratings_matrix[0, :])  # number of raters per subject
    if n <= 1:
        raise ValueError("Fleiss's Kappa requires at least 2 raters per subject.")

    # p_j: overall proportion of assignments to category j
    p_j = np.sum(ratings_matrix, axis=0) / (N * n)

    # P_i: extent of agreement for the i-th subject
    P_i = (np.sum(ratings_matrix**2, axis=1) - n) / (n * (n - 1))

    # Mean P_i
    P_bar = np.mean(P_i)
    # Expected agreement by chance
    P_bar_e = np.sum(p_j**2)

    if np.isclose(1.0 - P_bar_e, 0.0):
        return 1.0 if np.isclose(P_bar, 1.0) else 0.0

    kappa = (P_bar - P_bar_e) / (1.0 - P_bar_e)
    return float(kappa)


def parse_annotation_csv(path: Path, label_col: str = "label") -> pd.Series:
    """Extract standard binary or categorical annotation labels from a CSV."""
    df = pd.read_csv(path, comment="#")
    # Try common label column names if default not found
    candidates = [
        label_col, "overall", "factual", "atomic", "self_contained",
        "verdict", "hallucination", "annotation", "gold_label", "ground_truth"
    ]
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        raise ValueError(
            f"Could not find label column in '{path.name}'. Available columns: {list(df.columns)}"
        )
    return df[col].astype(str).str.strip().str.lower()


def compute_iaa(
    file_paths: list[Path],
    label_col: str = "label",
) -> dict[str, Any]:
    """Compute pairwise Cohen's Kappa and overall Fleiss's Kappa across annotation files."""
    check_files_distinct(file_paths)

    series_list = [parse_annotation_csv(p, label_col=label_col) for p in file_paths]
    names = [p.stem for p in file_paths]
    n_samples = len(series_list[0])

    for i, s in enumerate(series_list):
        if len(s) != n_samples:
            raise ValueError(
                f"Row count mismatch: '{names[0]}' has {n_samples} rows, but '{names[i]}' has {len(s)} rows."
            )

    categories = sorted(list(set.union(*[set(s.unique()) for s in series_list])))
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Pairwise Cohen's Kappa
    pairwise: list[dict[str, Any]] = []
    for i in range(len(file_paths)):
        for j in range(i + 1, len(file_paths)):
            s_i = series_list[i]
            s_j = series_list[j]
            k_val = cohen_kappa_score(s_i, s_j)
            obs_agr = float(np.mean(s_i == s_j))
            pairwise.append({
                "rater_1": names[i],
                "rater_2": names[j],
                "cohen_kappa": float(k_val),
                "percent_agreement": float(obs_agr),
            })

    # Fleiss's Kappa
    ratings_matrix = np.zeros((n_samples, len(categories)), dtype=int)
    for row_idx in range(n_samples):
        for s in series_list:
            cat = s.iloc[row_idx]
            ratings_matrix[row_idx, cat_to_idx[cat]] += 1

    fleiss_k = compute_fleiss_kappa(ratings_matrix)
    mean_pairwise_k = float(np.mean([p["cohen_kappa"] for p in pairwise]))

    return {
        "n_annotators": len(file_paths),
        "n_samples": n_samples,
        "categories": categories,
        "pairwise_cohen_kappa": pairwise,
        "mean_cohen_kappa": mean_pairwise_k,
        "fleiss_kappa": fleiss_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Inter-Annotator Agreement (Cohen's & Fleiss's Kappa)")
    parser.add_argument("files", nargs="+", type=Path, help="Paths to 2 or more annotation CSV files")
    parser.add_argument("--label-col", default="label", help="Name of column containing labels")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    if len(args.files) < 2:
        print("Error: At least two annotation files are required to compute agreement.")
        sys.exit(1)

    try:
        report = compute_iaa(args.files, label_col=args.label_col)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("=" * 65)
    print("Inter-Annotator Agreement (IAA) Report")
    print("=" * 65)
    print(f"Annotators: {report['n_annotators']}")
    print(f"Samples:    {report['n_samples']}")
    print(f"Categories: {', '.join(report['categories'])}")
    print(f"\nOverall Fleiss's Kappa: {report['fleiss_kappa']:.4f}")
    print(f"Mean Pairwise Cohen:    {report['mean_cohen_kappa']:.4f}")
    print("\nPairwise Agreement:")
    for p in report["pairwise_cohen_kappa"]:
        print(f"  {p['rater_1']} vs {p['rater_2']}: Cohen's κ = {p['cohen_kappa']:.4f} (Observed: {100*p['percent_agreement']:.1f}%)")


if __name__ == "__main__":
    main()
