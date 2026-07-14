# Domain Profiles — VeritasCore

> How to use built-in profiles and create custom domain configurations.

---

## Overview

Domain profiles allow you to tune VeritasCore's thresholds and behavior for specific use cases without modifying code. A profile controls:

- **Verdict thresholds** — at what trust score a claim is `SUPPORTED` vs `UNSUPPORTED` vs `CONTRADICTED`
- **Minimum confidence** — below this, a claim is flagged as `UNSUPPORTED` regardless of score
- **Required signals** — which verification signals must be present
- **Strictness level** — affects how the Explainer generates reasons

---

## Built-in Profiles

### `general` (default)

Balanced thresholds for everyday LLM output verification.

```yaml
name: general
description: "General-purpose verification for everyday LLM outputs"
thresholds:
  supported_min: 0.65      # score >= 0.65 → SUPPORTED
  contradicted_max: 0.35   # score <= 0.35 → CONTRADICTED
  min_confidence: 0.50     # below this → UNSUPPORTED
strictness: balanced
required_signals: []       # use whatever is available
```

**Use when**: Summarization, Q&A, general chatbots.

---

### `medical`

High-precision thresholds for medical/clinical content. Prioritizes recall of hallucinations (fewer false negatives).

```yaml
name: medical
description: "High-precision medical verification — conservative thresholds"
thresholds:
  supported_min: 0.80      # Requires strong evidence to call SUPPORTED
  contradicted_max: 0.20   # Even slight contradiction flags as CONTRADICTED
  min_confidence: 0.70     # Low-confidence claims → UNSUPPORTED
strictness: strict
required_signals: ["nli"]  # NLI signal required; web retrieval not trusted for medical
```

**Use when**: Clinical decision support, drug information, medical literature review.

> ⚠️ **Warning**: VeritasCore is not a medical device. Always verify critical medical information with qualified professionals.

---

### `legal`

Conservative thresholds for legal content where false confidence is harmful.

```yaml
name: legal
description: "Conservative legal verification — strict evidence requirements"
thresholds:
  supported_min: 0.75      # High bar for SUPPORTED
  contradicted_max: 0.25
  min_confidence: 0.65
strictness: strict
required_signals: []
```

**Use when**: Legal document review, case citation checking, contract analysis.

---

## Using a Profile

### Python Library

```python
from veritascore import VeritasCoreEngine

engine = VeritasCoreEngine()

# Use a built-in profile
report = engine.verify(
    response="Aspirin is contraindicated in children with viral infections.",
    query="Is aspirin safe for children?",
    context="Aspirin is contraindicated in children due to Reye's syndrome risk.",
    domain="medical",
)

print(report.overall_verdict.value)   # e.g. "supported"
```

### REST API

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "response": "...",
    "domain": "medical"
  }'
```

---

## Creating a Custom Profile

### 1. Create a YAML file

Custom profiles live in `configs/profiles/` (or any directory you specify):

```yaml
# configs/profiles/finance.yaml
name: finance
description: "Financial information verification — balanced with source requirements"
thresholds:
  supported_min: 0.70
  contradicted_max: 0.30
  min_confidence: 0.60
strictness: balanced
required_signals: []
notes: "Optimized for earnings reports and financial fact-checking"
```

### 2. Register the profile

```python
from veritascore import VeritasCoreEngine
from veritascore.profiles import DomainProfile

# Load from file
profile = DomainProfile.from_yaml("configs/profiles/finance.yaml")

# Use with the engine
engine = VeritasCoreEngine()
report = engine.verify(
    response="Apple's Q3 revenue was $81.8 billion.",
    query="What was Apple's Q3 2024 revenue?",
    domain=profile,
)
```

### 3. Profile schema reference

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique profile identifier |
| `description` | `str` | Human-readable description |
| `thresholds.supported_min` | `float 0–1` | Minimum trust score for SUPPORTED |
| `thresholds.contradicted_max` | `float 0–1` | Maximum trust score for CONTRADICTED |
| `thresholds.min_confidence` | `float 0–1` | Minimum confidence to avoid UNSUPPORTED |
| `strictness` | `"strict"` \| `"balanced"` \| `"lenient"` | Affects explainer tone |
| `required_signals` | `list[str]` | Signals that must be present (`"nli"`, `"retrieval"`, `"consistency"`) |

**Constraint**: `contradicted_max < supported_min` — there must be an "UNSUPPORTED" gap between the two thresholds.

---

## Profile Comparison Table

| Profile | SUPPORTED ≥ | CONTRADICTED ≤ | Use Case |
|---------|:-----------:|:--------------:|----------|
| `general` | 0.65 | 0.35 | General chatbots, summarization |
| `medical` | 0.80 | 0.20 | Clinical, drug information |
| `legal` | 0.75 | 0.25 | Legal documents, case law |
| Custom `finance` | 0.70 | 0.30 | Financial facts, earnings |

---

## Tips for Threshold Tuning

1. **Start with `general`** and measure precision/recall on your domain's test set.
2. **If too many false positives** (claiming things are SUPPORTED when wrong): raise `supported_min`.
3. **If too many false negatives** (missing hallucinations): lower `contradicted_max`.
4. **Use AUROC** — it's threshold-free and a better tuning metric than F1 for imbalanced data.
5. **Domain-specific fine-tuning**: If you have labeled examples, contribute to `scripts/train_fusion.py` to train a domain-specific fusion model.
