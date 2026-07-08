"""Debug script: inspect HaluEval dataset + NLI score distribution."""
from __future__ import annotations
import io
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# --- Inspect dataset structure ---
ds = load_from_disk("data/datasets/halueval/qa_samples")
print("Type:", type(ds))
print("Keys:", list(ds.keys()) if hasattr(ds, "keys") else "Dataset (no splits)")

split = ds["data"] if hasattr(ds, "__contains__") and "data" in ds else next(iter(ds.values()))
print("N rows:", len(split))
print("Columns:", list(split.features.keys()))

for i, row in enumerate(split):
    if i >= 3:
        break
    print(f"\nRow {i}:")
    for k, v in row.items():
        val = str(v)[:120]
        print(f"  {k}: {val!r}")

# Count hallucination distribution
counts: dict = {}
for i, row in enumerate(split):
    if i >= 200:
        break
    h = row["hallucination"]
    counts[h] = counts.get(h, 0) + 1
print("\nHallucination distribution (n=200):", counts)

# --- Quick NLI smoke test to check score direction ---
print("\n--- NLI score direction test ---")
from veritascore.verifier.nli_verifier import NLIVerifier
from veritascore.core.types import Claim

verifier = NLIVerifier()

# Test with first few rows
for i, row in enumerate(split):
    if i >= 5:
        break
    context = row.get("knowledge", "")
    answer = row.get("answer", "")
    hallucination = str(row.get("hallucination", "")).strip().lower()
    if not context or not answer:
        continue
    claim = Claim(id=f"c{i}", text=answer, source_span=(0, len(answer)), source_text=answer)
    verdict = verifier.verify([claim], context=context)[0]
    print(f"  [{i}] label={hallucination!r:5} | verdict={verdict.verdict.value:12} | nli_score={verdict.nli_score:.3f} | confidence={verdict.confidence:.3f}")

verifier.unload()
print("\nDone.")
