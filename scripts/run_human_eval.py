"""run_human_eval.py — Phase R6: Human Evaluation for Decomposition Quality.

Runs both RuleDecomposer and LLMDecomposer on the 100-sample stratified
annotation set and produces a CSV annotation spreadsheet.

Steps:
    1. Load 100 evaluation samples from tests/fixtures/human_eval_samples.json
    2. Run RuleDecomposer on all 100 samples
    3. Optionally run LLMDecomposer on all 100 samples (requires GPU/model)
    4. Produce annotation_spreadsheet.csv for human annotators
    5. Optionally run simulated self-annotation for CI/testing purposes

Usage:
    python scripts/run_human_eval.py --decomposer rule
    python scripts/run_human_eval.py --decomposer both
    python scripts/run_human_eval.py --decomposer rule --simulate-annotations
    python scripts/run_human_eval.py --decomposer rule --output-csv docs/r6_annotations.csv
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from veritascore.decomposer.rule_decomposer import RuleDecomposer  # noqa: E402
from veritascore.core.types import Claim  # noqa: E402

FIXTURES_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "human_eval_samples.json"
DEFAULT_CSV_PATH = Path(__file__).parent.parent / "docs" / "r6_annotation_spreadsheet.csv"
DEFAULT_RESULTS_PATH = Path(__file__).parent.parent / "docs" / "r6_decomposer_results.json"

CSV_COLUMNS = [
    "sample_id",
    "source",
    "original_text",
    "decomposer",
    "claim_num",
    "claim_text",
    "atomic",
    "self_contained",
    "factual",
    "overall",
    "annotator",
    "notes",
]


def load_samples() -> dict[str, Any]:
    """Load the human evaluation sample fixture."""
    with open(FIXTURES_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_decomposer_on_samples(
    decomposer: Any,
    samples: list[dict[str, Any]],
    source: str,
    decomposer_name: str,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """
    Decompose all samples and return a flat list of claim rows.

    Each row: sample_id, source, original_text, decomposer, claim_num, claim_text
    (annotation columns left blank for human annotators).
    """
    rows = []
    n_success = 0
    n_fail = 0
    latencies: list[float] = []

    print(f"\n  Running {decomposer_name} on {len(samples)} {source} samples...")

    for sample in samples:
        sid = sample["id"]
        text = sample["claim_text"]

        t0 = time.perf_counter()
        try:
            claims: list[Claim] = decomposer.decompose(text)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            n_success += 1

            if not claims:
                # Decomposer produced no output — treat original as one claim
                claims_text = [text]
            else:
                claims_text = [c.text for c in claims]

            for i, ct in enumerate(claims_text, start=1):
                rows.append({
                    "sample_id": sid,
                    "source": source,
                    "original_text": text,
                    "decomposer": decomposer_name.lower(),
                    "claim_num": i,
                    "claim_text": ct,
                    "atomic": "",
                    "self_contained": "",
                    "factual": "",
                    "overall": "",
                    "annotator": "",
                    "notes": "",
                })

            if verbose:
                print(f"    [{sid}] {len(claims_text)} claim(s): {claims_text[:2]}")

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            n_fail += 1
            # On failure, include a row for the original text
            rows.append({
                "sample_id": sid,
                "source": source,
                "original_text": text,
                "decomposer": decomposer_name.lower(),
                "claim_num": 1,
                "claim_text": text,
                "atomic": "",
                "self_contained": "",
                "factual": "",
                "overall": "",
                "annotator": "",
                "notes": f"ERROR: {exc}",
            })
            if verbose:
                print(f"    [{sid}] ERROR: {exc}")

    avg_ms = (1000 * sum(latencies) / len(latencies)) if latencies else 0.0
    print(f"    Done: {n_success}/{len(samples)} success, {n_fail} fail, avg {avg_ms:.1f}ms")
    return rows


def simulate_annotations(rows: list[dict[str, Any]], annotator_id: str = "A1") -> list[dict[str, Any]]:
    """
    Simulate human annotations using heuristic rules (for CI/testing only).
    
    This is NOT a substitute for real human annotation. It is used to:
    1. Validate the pipeline end-to-end in CI without human annotators.
    2. Produce realistic-looking output for IAA computation testing.
    
    Heuristics used:
    - Atomic: 0 if "and" or comma-separated conjunction pattern present
    - Self-Contained: 0 if starts with pronoun (He/She/It/They/This/That)
    - Factual: 0 if starts with opinion prefix ("I think", "Perhaps", etc.)
    - Overall: 1 iff all three are 1
    """
    import re

    OPINION_PREFIXES = (
        "i think", "i believe", "i feel", "perhaps", "maybe",
        "possibly", "probably", "it seems", "arguably", "reportedly",
        "note that", "please note",
    )
    DANGLING_PRONOUNS = re.compile(
        r"^(he|she|it|they|this|that|these|those|his|her|its|their)\b",
        re.IGNORECASE,
    )
    COMPOUND_PATTERN = re.compile(
        r"\b(and|but|while|whereas)\b.*(,|\band\b)",
        re.IGNORECASE,
    )

    simulated = []
    for row in rows:
        ct = row["claim_text"].strip()
        ct_lower = ct.lower()

        # Atomicity check
        has_compound = bool(
            COMPOUND_PATTERN.search(ct) or
            (", and " in ct) or
            (" and " in ct and ct.count(" and ") > 1)
        )
        atomic = 0 if has_compound else 1

        # Self-containedness check
        self_contained = 0 if DANGLING_PRONOUNS.match(ct) else 1

        # Factuality check
        factual = 0 if any(ct_lower.startswith(p) for p in OPINION_PREFIXES) else 1

        overall = 1 if (atomic and self_contained and factual) else 0

        new_row = dict(row)
        new_row["atomic"] = atomic
        new_row["self_contained"] = self_contained
        new_row["factual"] = factual
        new_row["overall"] = overall
        new_row["annotator"] = annotator_id
        simulated.append(new_row)

    return simulated


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Write annotation rows to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  [CSV] Annotation spreadsheet written to: {out_path}")
    print(f"        {len(rows)} claim rows across {len({r['sample_id'] for r in rows})} samples")


def compute_per_decomposer_accuracy(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """
    Compute per-decomposer per-criterion accuracy from annotated rows.
    
    Only rows with non-empty 'overall' are included.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("overall") != "":
            groups[row["decomposer"]].append(row)

    results: dict[str, dict[str, float]] = {}
    for decomp, group_rows in groups.items():
        total = len(group_rows)
        if total == 0:
            continue

        def pct(key: str) -> float:
            vals = [r[key] for r in group_rows if r.get(key) != ""]
            return round(100 * sum(int(v) for v in vals) / len(vals), 1) if vals else 0.0

        results[decomp] = {
            "total_claims": total,
            "atomic_pct": pct("atomic"),
            "self_contained_pct": pct("self_contained"),
            "factual_pct": pct("factual"),
            "overall_pct": pct("overall"),
        }

    return results


def print_accuracy_table(accuracy: dict[str, dict[str, float]]) -> None:
    """Print formatted accuracy table."""
    print(f"\n{'=' * 70}")
    print("Quality Gate R6 — Decomposition Accuracy (Per-Criterion)")
    print(f"{'=' * 70}")
    print(f"\n  {'Decomposer':<18} {'Atomic':>7} {'Self-Cont':>10} {'Factual':>8} {'Overall':>8} {'N Claims':>9}")
    print(f"  {'-'*18} {'-'*7} {'-'*10} {'-'*8} {'-'*8} {'-'*9}")
    for decomp, m in accuracy.items():
        print(
            f"  {decomp:<18} {m['atomic_pct']:>6.1f}% {m['self_contained_pct']:>9.1f}% "
            f"{m['factual_pct']:>7.1f}% {m['overall_pct']:>7.1f}% {m['total_claims']:>9}"
        )


def write_results_json(
    accuracy: dict[str, dict[str, float]],
    decomposer_names: list[str],
    out_path: Path,
) -> None:
    """Write machine-readable results JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "phase": "R6",
        "description": "Human evaluation accuracy per decomposer (simulated or real annotations)",
        "results": accuracy,
        "decomposers_evaluated": decomposer_names,
        "note": (
            "These results are from SIMULATED annotations (heuristic-based). "
            "Replace with real annotator data for publication."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  [JSON] Results written to: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase R6: Run decomposers on 100-sample human eval set and produce annotation CSV."
    )
    parser.add_argument(
        "--decomposer", choices=["rule", "llm", "both"], default="rule",
        help="Which decomposer(s) to evaluate (llm requires GPU/model download).",
    )
    parser.add_argument(
        "--output-csv", type=str, default=str(DEFAULT_CSV_PATH),
        help=f"Path to output CSV annotation spreadsheet (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--output-json", type=str, default=str(DEFAULT_RESULTS_PATH),
        help=f"Path to output results JSON (default: {DEFAULT_RESULTS_PATH})",
    )
    parser.add_argument(
        "--simulate-annotations", action="store_true",
        help=(
            "Apply heuristic self-annotation (for CI/testing only). "
            "NOT a substitute for real human annotation."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Print each sample's claims.")
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Per-sample timeout in seconds for LLM decomposer (default: 60s)",
    )
    args = parser.parse_args()

    data = load_samples()
    halueval = data["halueval_samples"]
    fever = data["fever_samples"]
    all_samples = halueval + fever
    print(f"\nLoaded {len(all_samples)} samples ({len(halueval)} HaluEval + {len(fever)} FEVER)")

    all_rows: list[dict[str, Any]] = []
    decomposers_run: list[str] = []

    # --- RuleDecomposer ---
    if args.decomposer in ("rule", "both"):
        print(f"\n{'=' * 70}")
        print("RuleDecomposer Evaluation")
        print(f"{'=' * 70}")
        rule_decomp = RuleDecomposer()
        rule_rows = run_decomposer_on_samples(rule_decomp, halueval, "halueval", "rule", args.verbose)
        rule_rows += run_decomposer_on_samples(rule_decomp, fever, "fever", "rule", args.verbose)
        all_rows.extend(rule_rows)
        decomposers_run.append("rule")

    # --- LLMDecomposer ---
    if args.decomposer in ("llm", "both"):
        print(f"\n{'=' * 70}")
        print("LLMDecomposer Evaluation")
        print(f"{'=' * 70}")
        from veritascore.decomposer.llm_decomposer import LLMDecomposer
        llm_decomp = LLMDecomposer(fallback_on_error=False, timeout_per_sample=args.timeout)
        llm_rows = run_decomposer_on_samples(llm_decomp, halueval, "halueval", "llm", args.verbose)
        llm_rows += run_decomposer_on_samples(llm_decomp, fever, "fever", "llm", args.verbose)
        llm_decomp.unload()
        all_rows.extend(llm_rows)
        decomposers_run.append("llm")

    # --- Simulate annotations (for CI testing) ---
    if args.simulate_annotations:
        print("\n  [SIM] Applying simulated annotations (heuristic-based, for CI only)...")
        all_rows = simulate_annotations(all_rows, annotator_id="SIM-A1")
        accuracy = compute_per_decomposer_accuracy(all_rows)
        print_accuracy_table(accuracy)
        write_results_json(accuracy, decomposers_run, Path(args.output_json))

    # --- Write CSV ---
    write_csv(all_rows, Path(args.output_csv))

    print(f"\n  Total claim rows in spreadsheet: {len(all_rows)}")
    print(f"\n  Next steps:")
    print(f"    1. Distribute {args.output_csv} to human annotators.")
    print(f"    2. Annotators fill: atomic, self_contained, factual, overall, annotator, notes.")
    print(f"    3. Run: python scripts/compute_iaa.py --csv {args.output_csv}")
    print(f"    4. Report Cohen's kappa >= 0.60 for substantial agreement.")


if __name__ == "__main__":
    main()
