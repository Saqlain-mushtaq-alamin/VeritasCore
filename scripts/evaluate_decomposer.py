"""Evaluate decomposer quality against the 50-sample annotated dataset.

Implements the Quality Gate G1 checks from the Phase 1 plan:
    - Atomic accuracy spot-check (manual review log)
    - Claims-per-sentence ratio
    - Span mapping validity
    - Latency benchmarking (LLM vs Rule)

Usage:
    python scripts/evaluate_decomposer.py --decomposer rule
    python scripts/evaluate_decomposer.py --decomposer llm
    python scripts/evaluate_decomposer.py --decomposer both --verbose
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

# Force UTF-8 output on Windows so Unicode symbols don't crash CP1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from veritascore.decomposer.rule_decomposer import RuleDecomposer  # noqa: E402

FIXTURES_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "decomposition_samples.json"


def load_samples() -> list[dict[str, Any]]:
    """Load the 50-sample annotated evaluation dataset."""
    with open(FIXTURES_PATH) as f:
        data = json.load(f)
    return data["samples"]


def check_span_validity(claims: list[Any], response: str) -> tuple[int, int]:
    """Return (valid_count, total_count) for span boundary checks."""
    valid = 0
    for c in claims:
        start, end = c.source_span
        if 0 <= start < end <= len(response):
            valid += 1
    return valid, len(claims)


def evaluate_decomposer(
    decomposer: Any,
    samples: list[dict[str, Any]],
    name: str,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the decomposer over all samples and collect metrics."""
    print(f"\n{'=' * 70}")
    print(f"Evaluating: {name}")
    print(f"{'=' * 70}")

    in_range_count = 0
    total_valid_spans = 0
    total_spans = 0
    total_claims = 0
    total_sentences_estimate = 0
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []

    for sample in samples:
        response = sample["response"]
        min_c = sample["min_claims"]
        max_c = sample["max_claims"]

        t0 = time.perf_counter()
        try:
            claims = decomposer.decompose(response)
        except Exception as e:
            failures.append({"id": sample["id"], "error": str(e)})
            continue
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

        n_claims = len(claims)
        total_claims += n_claims
        # Rough sentence estimate for ratio metric
        total_sentences_estimate += max(1, response.count(".") + response.count("!") + response.count("?"))

        in_range = min_c <= n_claims <= max_c
        if in_range:
            in_range_count += 1

        valid_spans, n_spans = check_span_validity(claims, response)
        total_valid_spans += valid_spans
        total_spans += n_spans

        if verbose or not in_range:
            status = "[PASS]" if in_range else "[FAIL]"
            print(
                f"  {status} [{sample['id']:>2}] expected [{min_c}-{max_c}], got {n_claims} "
                f"| {sample['notes'][:50]}"
            )
            if verbose:
                for c in claims:
                    print(f"        - {c.text}")

    n = len(samples)
    n_run = len(latencies)

    results = {
        "name": name,
        "n_samples": n,
        "n_successful": n_run,
        "n_failures": len(failures),
        "in_range_count": in_range_count,
        "in_range_pct": round(100 * in_range_count / n, 1) if n else 0.0,
        "span_validity_pct": round(100 * total_valid_spans / total_spans, 1) if total_spans else 100.0,
        "avg_claims_per_sample": round(total_claims / n_run, 2) if n_run else 0.0,
        "claims_per_sentence_ratio": round(total_claims / total_sentences_estimate, 2) if total_sentences_estimate else 0.0,
        "avg_latency_ms": round(1000 * sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "max_latency_ms": round(1000 * max(latencies), 2) if latencies else 0.0,
        "failures": failures,
    }

    print(f"\n  Summary for {name}:")
    print(f"    In expected range:      {results['in_range_count']}/{n} ({results['in_range_pct']}%)")
    print(f"    Span validity:          {results['span_validity_pct']}%")
    print(f"    Avg claims/sample:      {results['avg_claims_per_sample']}")
    print(f"    Claims/sentence ratio:  {results['claims_per_sentence_ratio']} (target: 1.5-3.0)")
    print(f"    Avg latency:            {results['avg_latency_ms']} ms")
    print(f"    Max latency:            {results['max_latency_ms']} ms")
    if failures:
        print(f"    [WARN] Failures: {len(failures)}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VeritasCore claim decomposers")
    parser.add_argument(
        "--decomposer", choices=["rule", "llm", "both"], default="rule",
        help="Which decomposer to evaluate (llm requires model download)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print every claim, not just failures")
    args = parser.parse_args()

    samples = load_samples()
    print(f"Loaded {len(samples)} annotated samples from {FIXTURES_PATH.name}")

    all_results = []

    if args.decomposer in ("rule", "both"):
        rule_decomposer = RuleDecomposer()
        all_results.append(evaluate_decomposer(rule_decomposer, samples, "RuleDecomposer", args.verbose))

    if args.decomposer in ("llm", "both"):
        from veritascore.decomposer.llm_decomposer import LLMDecomposer
        llm_decomposer = LLMDecomposer(fallback_on_error=False)
        all_results.append(evaluate_decomposer(llm_decomposer, samples, "LLMDecomposer", args.verbose))
        llm_decomposer.unload()

    print(f"\n{'=' * 70}")
    print("Quality Gate G1 — Target Checklist")
    print(f"{'=' * 70}")
    for r in all_results:
        print(f"\n  {r['name']}:")
        print(f"    [{'PASS' if r['in_range_pct'] >= 90 else 'FAIL'}] Atomic accuracy >90%   : {r['in_range_pct']}%")
        print(f"    [{'PASS' if r['span_validity_pct'] >= 85 else 'FAIL'}] Span validity >85%     : {r['span_validity_pct']}%")
        print(f"    [{'PASS' if 1.0 <= r['claims_per_sentence_ratio'] <= 3.5 else 'WARN'}] Claims/sentence 1.5-3.0: {r['claims_per_sentence_ratio']}")


if __name__ == "__main__":
    main()
