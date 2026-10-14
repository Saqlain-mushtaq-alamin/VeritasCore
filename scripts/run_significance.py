"""Compute paired DeLong's test and paired bootstrap tests on saved benchmark scores.

Loads raw y_true and y_score arrays from:
  - tests/benchmarks/results/halueval_results_v2.json
  - tests/benchmarks/results/halueval_qa_summac_oracle_results.json
  - tests/benchmarks/results/fever_results_v2.json
  - tests/benchmarks/results/fever_validation_summac_oracle_results.json

Outputs:
  tests/benchmarks/results/significance_results.json
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_benchmarks_v2 import run_significance_tests  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired DeLong tests on saved benchmark results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("tests/benchmarks/results"),
        help="Directory containing saved JSON results",
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = args.results_dir
    hu_vc_file = results_dir / "halueval_results_v2.json"
    fv_vc_file = results_dir / "fever_results_v2.json"

    vc_results: list[dict[str, Any]] = []
    if hu_vc_file.exists():
        vc_results.append(json.loads(hu_vc_file.read_text(encoding="utf-8")))
    if fv_vc_file.exists():
        vc_results.append(json.loads(fv_vc_file.read_text(encoding="utf-8")))

    if not vc_results:
        print(f"No VeritasCore results found in {results_dir}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("PAIRED SIGNIFICANCE TESTING (DeLong's Test + Paired Bootstrap)")
    print(f"Results directory: {results_dir.resolve()}")
    print("=" * 70)

    comparisons = run_significance_tests(
        vc_results,
        results_dir=results_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    out_file = results_dir / "significance_results.json"
    out_file.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    print(f"\nSignificance test comparisons saved → {out_file}\n")


if __name__ == "__main__":
    main()
