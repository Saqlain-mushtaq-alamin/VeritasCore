"""Debug what Phi-3 actually outputs for the opinion_filtering test input."""
import sys
sys.path.insert(0, "src")

from veritascore.decomposer.llm_decomposer import LLMDecomposer

decomposer = LLMDecomposer(fallback_on_error=False)

text = "I believe Python is the best language. Python was released in 1991."
print(f"Input: {text!r}\n")

# Patch to capture raw output before parsing
original_parse = LLMDecomposer._parse_numbered_claims
captured_raw = []

@staticmethod
def patched_parse(text):
    captured_raw.append(text)
    return original_parse(text)

LLMDecomposer._parse_numbered_claims = patched_parse

claims = decomposer.decompose(text)

print(f"Raw LLM output: {captured_raw[0]!r}" if captured_raw else "No raw output captured")
print(f"\nExtracted claims ({len(claims)}):")
for c in claims:
    print(f"  - {c.text!r}")

if claims:
    joined = " ".join(c.text for c in claims).lower()
    print(f"\n'1991' in claims: {'1991' in joined}")
    print(f"'best' in claims: {'best' in joined}")
    print(f"'python' in claims: {'python' in joined}")

decomposer.unload()
