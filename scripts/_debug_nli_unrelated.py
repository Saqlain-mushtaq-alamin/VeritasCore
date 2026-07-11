"""Diagnostic: check NLI score for unrelated context test."""
import numpy as np
from veritascore.verifier.nli_verifier import NLIVerifier
from veritascore.core.types import Claim

verifier = NLIVerifier()
verifier._load_model()

claim = Claim(
    text="Bananas are a good source of potassium.",
    source_span=(0, 41),
    source_text="Bananas are a good source of potassium.",
)
context = "The stock market fell sharply on Tuesday amid inflation fears."
probs = verifier._run_nli(premise=context, hypothesis=claim.text)
print(f"[contradiction={probs[0]:.3f}, neutral={probs[1]:.3f}, entailment={probs[2]:.3f}]")
print(f"Entailment threshold: {verifier.entailment_threshold}")
print(f"Would be SUPPORTED: {probs[2] >= verifier.entailment_threshold}")

# Also check the clear entailment case
probs2 = verifier._run_nli(
    premise="The Eiffel Tower is a famous landmark in Paris, France.",
    hypothesis="The Eiffel Tower is located in Paris."
)
print(f"\nClear entailment: [contra={probs2[0]:.3f}, neutral={probs2[1]:.3f}, entail={probs2[2]:.3f}]")

verifier.unload()
