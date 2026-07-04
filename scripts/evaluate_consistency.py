"""Evaluate SemanticConsistencyChecker against the curated consistency
fixture for Quality Gate G4.

Usage:
    python scripts/evaluate_consistency.py
    python scripts/evaluate_consistency.py --threshold 0.25 --verbose
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from veritascore.core.types import Claim  # noqa: E402
from veritascore.verifier.consistency import SemanticConsistencyChecker  # noqa: E402

FIXTURES_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "consistency_samples.json"


def load_samples() -> list[dict[str, Any]]:
    with open(FIXTURES_PATH) as f:
        return json.load(f)["samples"]


def make_claim(text: str, idx: int) -> Claim:
    return Claim(id=f"c{idx}", text=text, source_span=(0, len(text)), source_text=text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VeritasCore semantic consistency checker")
    parser.add_argument("--threshold", type=float, default=0.3, help="Off-topic relevance threshold")
    parser.add_argument("--verbose", action="store_true", help="Print every claim score")
    args = parser.parse_args()

    samples = load_samples()
    print(f"Loaded {len(samples)} curated samples from {FIXTURES_PATH.name}")

    checker = SemanticConsistencyChecker(relevance_threshold=args.threshold)

    t0 = time.time()
    checker._load_model()
    load_time = time.time() - t0
    print(f"\nModel load time: {load_time * 1000:.0f}ms (target: <2000ms)")

    on_topic_total = 0
    on_topic_passed = 0
    off_topic_total = 0
    off_topic_passed = 0
    flagged_total = 0
    flagged_correct = 0

    idx = 0
    for sample in samples:
        query = sample["query"]

        on_topic_claims = []
        for text in sample["on_topic_claims"]:
            idx += 1
            claim = make_claim(text, idx)
            on_topic_claims.append(claim)
            score = checker.score_claim(claim, query)
            on_topic_total += 1
            ok = score > 0.5
            if ok:
                on_topic_passed += 1
            if args.verbose or not ok:
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] on-topic  score={score:.3f}  {text[:60]!r}")

        off_topic_claims = []
        for text in sample["off_topic_claims"]:
            idx += 1
            claim = make_claim(text, idx)
            off_topic_claims.append(claim)
            score = checker.score_claim(claim, query)
            off_topic_total += 1
            ok = score < 0.3
            if ok:
                off_topic_passed += 1
            if args.verbose or not ok:
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] off-topic score={score:.3f}  {text[:60]!r}")

        # off_topic_claims-list detection check (full check_consistency path)
        all_claims = on_topic_claims + off_topic_claims
        response_text = " ".join(c.text for c in all_claims)
        result = checker.check_consistency(all_claims, query, response_text)
        for claim in off_topic_claims:
            flagged_total += 1
            if claim.id in result.off_topic_claims:
                flagged_correct += 1

    checker.unload()

    on_topic_rate = on_topic_passed / on_topic_total if on_topic_total else 0.0
    off_topic_rate = off_topic_passed / off_topic_total if off_topic_total else 0.0
    flagged_rate = flagged_correct / flagged_total if flagged_total else 0.0

    print(f"\n{'=' * 70}")
    print("Quality Gate G4 — Results")
    print(f"{'=' * 70}")
    print(f"  [{'PASS' if on_topic_rate > 0.90 else 'FAIL'}] Criterion 1: on-topic >0.5 on >90% of samples")
    print(f"        {on_topic_passed}/{on_topic_total} = {on_topic_rate:.1%}")
    print(f"  [{'PASS' if off_topic_rate > 0.80 else 'FAIL'}] Criterion 2: off-topic <0.3 on >80% of samples")
    print(f"        {off_topic_passed}/{off_topic_total} = {off_topic_rate:.1%}")
    print(f"  [{'PASS' if flagged_rate > 0.80 else 'FAIL'}] Criterion 6: off_topic_claims list flags >80% of planted items")
    print(f"        {flagged_correct}/{flagged_total} = {flagged_rate:.1%}")
    print(f"  [{'PASS' if load_time < 2.0 else 'FAIL'}] Criterion 4: model loads in <2s")
    print(f"        {load_time * 1000:.0f}ms")


if __name__ == "__main__":
    main()
