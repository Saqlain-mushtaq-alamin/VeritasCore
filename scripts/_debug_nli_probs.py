"""Debug: inspect raw NLI probabilities to calibrate hallucination score."""
from __future__ import annotations
import io, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_from_disk
from veritascore.verifier.nli_verifier import NLIVerifier
from veritascore.core.types import Claim

DATA_DIR = Path("data/datasets")
ds = load_from_disk(str(DATA_DIR / "halueval" / "qa_samples"))
split = ds["data"]

verifier = NLIVerifier(entailment_threshold=0.3, contradiction_threshold=0.3)
verifier._load_model()

n = 100
raw_scores = {"yes": [], "no": []}  # label -> list of (entail, contra, neutral)

for i, row in enumerate(split):
    if i >= n:
        break
    context = row.get("knowledge", "")
    question = row.get("question", "")
    answer = row.get("answer", "")
    hallucination = str(row.get("hallucination", "")).strip().lower()
    if not context or not answer:
        continue
    claim_text = f"Q: {question}  A: {answer}" if question else answer
    probs = verifier._run_nli(premise=context[:1500], hypothesis=claim_text)
    # probs order: [contradiction, neutral, entailment]
    raw_scores[hallucination].append(probs)

verifier.unload()

for label in ["yes", "no"]:
    arr = np.array(raw_scores[label])
    print(f"\nLabel={label!r} (n={len(arr)}):")
    print(f"  Contradiction: mean={arr[:,0].mean():.3f}  std={arr[:,0].std():.3f}  max={arr[:,0].max():.3f}")
    print(f"  Neutral:       mean={arr[:,1].mean():.3f}  std={arr[:,1].std():.3f}")
    print(f"  Entailment:    mean={arr[:,2].mean():.3f}  std={arr[:,2].std():.3f}  max={arr[:,2].max():.3f}")

# Best AUROC scoring functions
from sklearn.metrics import roc_auc_score
all_labels = []
scores = {"1-entail": [], "contra": [], "max_ce": [], "contra_minus_entail": [], "neutral_inv": []}
for label, arr_list in raw_scores.items():
    arr = np.array(arr_list)
    lv = 1 if label == "yes" else 0
    for row in arr:
        all_labels.append(lv)
        scores["1-entail"].append(1.0 - row[2])
        scores["contra"].append(row[0])
        scores["max_ce"].append(max(row[0], 1.0 - row[2]))
        scores["contra_minus_entail"].append(row[0] - row[2])
        scores["neutral_inv"].append(1.0 - row[1])

print("\n--- AUROC by scoring function ---")
for name, sc in scores.items():
    try:
        auc = roc_auc_score(all_labels, sc)
        print(f"  {name:30s}: AUROC = {auc:.4f}")
    except Exception as e:
        print(f"  {name}: ERROR {e}")
