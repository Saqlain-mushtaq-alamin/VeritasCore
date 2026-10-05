"""compute_iaa.py — Phase R6: Compute inter-annotator agreement (Cohen's/Fleiss' kappa).

Usage:
    # Two-annotator mode (Cohen's kappa):
    python scripts/compute_iaa.py --csv docs/r6_annotation_spreadsheet.csv --a1 A1 --a2 A2

    # Three-annotator mode (Fleiss' kappa):
    python scripts/compute_iaa.py --csv docs/r6_annotation_spreadsheet.csv --a1 A1 --a2 A2 --a3 A3

    # Read from two separate CSV files (one per annotator):
    python scripts/compute_iaa.py --csv-a1 annotator1.csv --csv-a2 annotator2.csv

    # Run on simulated annotations (for CI/testing):
    python scripts/compute_iaa.py --csv docs/r6_annotation_spreadsheet.csv --simulated

Outputs:
    - Console report with per-criterion kappa scores and pass/fail gates
    - Optional JSON report written to docs/r6_iaa_report.json
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DEFAULT_CSV_PATH = Path(__file__).parent.parent / "docs" / "r6_annotation_spreadsheet.csv"
DEFAULT_REPORT_PATH = Path(__file__).parent.parent / "docs" / "r6_iaa_report.json"

CRITERIA = ["atomic", "self_contained", "factual", "overall"]
KAPPA_TARGET = 0.60  # Landis & Koch "substantial" agreement


# ── Cohen's Kappa (two annotators) ────────────────────────────────────────────

def cohen_kappa(y1: list[int], y2: list[int]) -> float:
    """Compute Cohen's kappa for two annotators with binary labels."""
    if len(y1) != len(y2):
        raise ValueError(f"Annotation vectors must have equal length: {len(y1)} vs {len(y2)}")
    if len(y1) == 0:
        return float("nan")

    n = len(y1)
    agree = sum(a == b for a, b in zip(y1, y2))
    po = agree / n  # observed agreement

    # Expected agreement by chance
    p1_pos = sum(y1) / n
    p2_pos = sum(y2) / n
    pe = (p1_pos * p2_pos) + ((1 - p1_pos) * (1 - p2_pos))

    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    kappa = (po - pe) / (1 - pe)
    return round(kappa, 4)


# ── Fleiss' Kappa (three or more annotators) ──────────────────────────────────

def fleiss_kappa(ratings_matrix: list[list[int]], n_categories: int = 2) -> float:
    """
    Compute Fleiss' kappa for k annotators and binary labels.

    ratings_matrix: shape (n_items, n_annotators), values in {0, 1}
    n_categories: number of label categories (2 for binary)
    """
    if not ratings_matrix:
        return float("nan")

    n_items = len(ratings_matrix)
    n_raters = len(ratings_matrix[0])
    N = n_items
    n = n_raters

    # p_j = proportion of all assignments to category j
    p = [0.0] * n_categories
    for row in ratings_matrix:
        for label in row:
            p[label] += 1
    total_assignments = N * n
    p = [x / total_assignments for x in p]

    # P_i = extent of agreement for item i
    P_i = []
    for row in ratings_matrix:
        counts = [row.count(j) for j in range(n_categories)]
        if n <= 1:
            P_i.append(1.0)
        else:
            pi = (sum(c * (c - 1) for c in counts)) / (n * (n - 1))
            P_i.append(pi)

    P_bar = sum(P_i) / N  # mean of P_i
    P_e_bar = sum(pj ** 2 for pj in p)  # expected agreement by chance

    if P_e_bar == 1.0:
        return 1.0 if P_bar == 1.0 else 0.0

    kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
    return round(kappa, 4)


# ── CSV Loading ────────────────────────────────────────────────────────────────

def load_annotations(csv_path: Path) -> list[dict[str, Any]]:
    """Load annotation rows from CSV."""
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def group_by_annotator(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group annotation rows by annotator ID."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ann = row.get("annotator", "").strip()
        if ann:
            groups[ann].append(row)
    return dict(groups)


def align_annotations(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
) -> dict[str, tuple[list[int], list[int]]]:
    """
    Align two annotator's rows by (sample_id, decomposer, claim_num) key.
    Returns dict: criterion -> (labels_a, labels_b)
    """
    key_fn = lambda r: (r["sample_id"], r.get("decomposer", ""), str(r.get("claim_num", "")))

    a_by_key = {key_fn(r): r for r in rows_a}
    b_by_key = {key_fn(r): r for r in rows_b}
    shared_keys = sorted(set(a_by_key) & set(b_by_key))

    aligned: dict[str, tuple[list[int], list[int]]] = {c: ([], []) for c in CRITERIA}
    for k in shared_keys:
        ra = a_by_key[k]
        rb = b_by_key[k]
        for crit in CRITERIA:
            va = ra.get(crit, "")
            vb = rb.get(crit, "")
            if va != "" and vb != "":
                try:
                    aligned[crit][0].append(int(va))
                    aligned[crit][1].append(int(vb))
                except ValueError:
                    pass  # skip non-numeric cells

    return aligned


def interpret_kappa(kappa: float) -> str:
    """Return Landis & Koch label for a kappa value."""
    if kappa < 0:
        return "Poor"
    elif kappa < 0.20:
        return "Slight"
    elif kappa < 0.40:
        return "Fair"
    elif kappa < 0.60:
        return "Moderate"
    elif kappa < 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


def simulate_second_annotator(rows: list[dict[str, Any]], seed: int = 99) -> list[dict[str, Any]]:
    """
    Produce a second simulated annotator with ~85% agreement (for CI testing).
    Adds small random noise to A1's labels.
    """
    import random
    rng = random.Random(seed)
    sim_rows = []
    for row in rows:
        new_row = dict(row)
        new_row["annotator"] = "SIM-A2"
        for crit in CRITERIA:
            v = row.get(crit, "")
            if v != "":
                # ~15% chance to flip the label (realistic IAA noise)
                if rng.random() < 0.15:
                    new_row[crit] = str(1 - int(v))
        # Recompute overall from the criteria
        try:
            new_row["overall"] = str(
                int(new_row.get("atomic", "0")) &
                int(new_row.get("self_contained", "0")) &
                int(new_row.get("factual", "0"))
            )
        except (ValueError, TypeError):
            pass
        sim_rows.append(new_row)
    return sim_rows


def print_iaa_report(
    kappa_results: dict[str, float],
    n_aligned: dict[str, int],
    annotators: list[str],
    kappa_type: str,
) -> bool:
    """Print formatted IAA report. Returns True if all gates pass."""
    print(f"\n{'=' * 70}")
    print(f"Phase R6 — Inter-Annotator Agreement Report ({kappa_type})")
    print(f"{'=' * 70}")
    print(f"  Annotators: {', '.join(annotators)}")
    print(f"\n  {'Criterion':<22} {'κ':>6}  {'N pairs':>8}  {'Interpretation':<22}  {'Gate'}")
    print(f"  {'-'*22} {'-'*6}  {'-'*8}  {'-'*22}  {'-'*6}")

    all_pass = True
    for crit in CRITERIA:
        k = kappa_results.get(crit, float("nan"))
        n = n_aligned.get(crit, 0)
        interp = interpret_kappa(k) if not (k != k) else "N/A"  # nan check
        passes = k >= KAPPA_TARGET if k == k else False
        gate = "PASS" if passes else "FAIL"
        if crit == "overall" and not passes:
            all_pass = False
        k_str = f"{k:.4f}" if k == k else "  nan"
        print(f"  {crit:<22} {k_str:>6}  {n:>8}  {interp:<22}  [{gate}]")

    print()
    if all_pass:
        print("  [ALL GATES PASSED] Primary metric (Overall κ) meets target >= 0.60.")
    else:
        print("  [GATE FAILED] Overall κ did not meet the target of >= 0.60.")
        print("  Action: Review annotation guidelines, run calibration round, re-annotate.")

    return all_pass


def write_report_json(
    kappa_results: dict[str, float],
    n_aligned: dict[str, int],
    annotators: list[str],
    kappa_type: str,
    all_pass: bool,
    out_path: Path,
) -> None:
    """Write JSON report."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "phase": "R6",
        "kappa_type": kappa_type,
        "annotators": annotators,
        "kappa_target": KAPPA_TARGET,
        "results": {
            crit: {
                "kappa": kappa_results.get(crit),
                "n_aligned_pairs": n_aligned.get(crit, 0),
                "interpretation": interpret_kappa(kappa_results[crit]) if crit in kappa_results else None,
                "passes": (kappa_results.get(crit, 0) >= KAPPA_TARGET),
            }
            for crit in CRITERIA
        },
        "primary_gate_passed": all_pass,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\n  [JSON] IAA report written to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase R6: Compute inter-annotator agreement (Cohen's/Fleiss' kappa)."
    )
    parser.add_argument(
        "--csv", type=str, default=str(DEFAULT_CSV_PATH),
        help=f"Combined annotation CSV with all annotators (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument("--a1", type=str, default=None, help="Annotator 1 ID (e.g. A1)")
    parser.add_argument("--a2", type=str, default=None, help="Annotator 2 ID (e.g. A2)")
    parser.add_argument("--a3", type=str, default=None, help="Optional annotator 3 ID for Fleiss kappa")
    parser.add_argument(
        "--simulated", action="store_true",
        help="Run with simulated second annotator (for CI testing).",
    )
    parser.add_argument(
        "--output-json", type=str, default=str(DEFAULT_REPORT_PATH),
        help=f"Path to output IAA report JSON (default: {DEFAULT_REPORT_PATH})",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"  [ERROR] CSV not found: {csv_path}")
        print(f"  Run: python scripts/run_human_eval.py --decomposer rule --simulate-annotations")
        sys.exit(1)

    rows = load_annotations(csv_path)
    by_annotator = group_by_annotator(rows)

    annotator_ids = sorted(by_annotator.keys())
    print(f"\nLoaded {len(rows)} rows from {csv_path.name}")
    print(f"Annotators found: {annotator_ids}")

    # Simulated mode: generate a synthetic A2 from A1's labels
    if args.simulated:
        a1_id = annotator_ids[0] if annotator_ids else None
        if not a1_id:
            print("  [ERROR] No annotated rows found. Run --simulate-annotations first.")
            sys.exit(1)
        print(f"\n  [SIM] Generating simulated second annotator from '{a1_id}' labels...")
        sim_rows = simulate_second_annotator(by_annotator[a1_id])
        by_annotator["SIM-A2"] = sim_rows
        annotator_ids = [a1_id, "SIM-A2"]

    elif args.a1 and args.a2:
        annotator_ids = [args.a1, args.a2]
        if args.a3:
            annotator_ids.append(args.a3)

    # Need at least 2 annotators
    if len(by_annotator) < 2:
        print(f"  [ERROR] Need at least 2 annotators; found: {list(by_annotator.keys())}")
        print("  Use --simulated to generate a synthetic second annotator.")
        sys.exit(1)

    # Pick the first two for Cohen's kappa
    ann_a, ann_b = annotator_ids[0], annotator_ids[1]
    a_rows = by_annotator.get(ann_a, [])
    b_rows = by_annotator.get(ann_b, [])

    if not a_rows or not b_rows:
        print(f"  [ERROR] Missing annotations for {ann_a} or {ann_b}")
        sys.exit(1)

    aligned = align_annotations(a_rows, b_rows)
    kappa_results: dict[str, float] = {}
    n_aligned: dict[str, int] = {}

    # Cohen's kappa (2 annotators) or extend to Fleiss' (3+)
    if len(annotator_ids) >= 3 and annotator_ids[2] in by_annotator:
        ann_c = annotator_ids[2]
        c_rows = by_annotator[ann_c]
        kappa_type = "Fleiss' κ"

        # Build ratings matrix for each criterion
        key_fn = lambda r: (r["sample_id"], r.get("decomposer", ""), str(r.get("claim_num", "")))
        a_by_key = {key_fn(r): r for r in a_rows}
        b_by_key = {key_fn(r): r for r in b_rows}
        c_by_key = {key_fn(r): r for r in c_rows}
        shared = sorted(set(a_by_key) & set(b_by_key) & set(c_by_key))

        for crit in CRITERIA:
            matrix = []
            for k in shared:
                va = a_by_key[k].get(crit, "")
                vb = b_by_key[k].get(crit, "")
                vc = c_by_key[k].get(crit, "")
                try:
                    matrix.append([int(va), int(vb), int(vc)])
                except ValueError:
                    pass
            kappa_results[crit] = fleiss_kappa(matrix) if matrix else float("nan")
            n_aligned[crit] = len(matrix)
        annotators_used = [ann_a, ann_b, ann_c]

    else:
        kappa_type = "Cohen's κ"
        for crit in CRITERIA:
            ya, yb = aligned[crit]
            kappa_results[crit] = cohen_kappa(ya, yb) if ya else float("nan")
            n_aligned[crit] = len(ya)
        annotators_used = [ann_a, ann_b]

    all_pass = print_iaa_report(kappa_results, n_aligned, annotators_used, kappa_type)

    write_report_json(
        kappa_results, n_aligned, annotators_used, kappa_type, all_pass,
        Path(args.output_json),
    )

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
