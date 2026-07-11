"""Diagnostic: check consistency scores for fixture on-topic claims."""
import json
from pathlib import Path
from veritascore.core.types import Claim
from veritascore.verifier.consistency import SemanticConsistencyChecker

FIXTURES_PATH = Path("tests/fixtures/consistency_samples.json")

checker = SemanticConsistencyChecker()
checker._load_model()

with open(FIXTURES_PATH) as f:
    data = json.load(f)

total = 0
passed = 0
failed_items = []

for sample in data["samples"]:
    query = sample["query"]
    for claim_text in sample["on_topic_claims"]:
        claim = Claim(id=f"c{total}", text=claim_text, source_span=(0, len(claim_text)), source_text=claim_text)
        score = checker.score_claim(claim, query)
        total += 1
        if score > 0.5:
            passed += 1
        else:
            failed_items.append((score, query, claim_text))

pass_rate = passed / total
print(f"Pass rate: {pass_rate:.1%} ({passed}/{total})")
print(f"\nFailed items (score <= 0.5):")
for score, query, claim in sorted(failed_items):
    print(f"  [{score:.3f}] Q: {query}")
    print(f"            C: {claim}")

checker.unload()
