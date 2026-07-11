"""Test various unrelated premise/hypothesis pairs for NLI model behavior."""
import numpy as np
from veritascore.verifier.nli_verifier import NLIVerifier

verifier = NLIVerifier()
verifier._load_model()

pairs = [
    ("The stock market fell sharply on Tuesday amid inflation fears.", "Bananas are a good source of potassium."),
    ("Paris is the capital of France.", "The mitochondria is the powerhouse of the cell."),
    ("Water boils at 100 degrees Celsius.", "The Roman Empire fell in 476 AD."),
    ("The Eiffel Tower stands in Paris.", "Photosynthesis uses chlorophyll."),
    ("Mount Everest is the highest peak on Earth.", "Bananas contain potassium."),
]

for premise, hypothesis in pairs:
    probs = verifier._run_nli(premise=premise, hypothesis=hypothesis)
    verdict = "SUPPORTED" if probs[2] >= 0.7 else ("CONTRADICTED" if probs[0] >= 0.5 else "UNSUPPORTED")
    print(f"[contra={probs[0]:.3f}, neutral={probs[1]:.3f}, entail={probs[2]:.3f}] -> {verdict}")
    print(f"  P: {premise[:60]}")
    print(f"  H: {hypothesis[:60]}")

verifier.unload()
