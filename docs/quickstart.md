# Quick Start — VeritasCore

> Get from zero to your first verification report in under 5 minutes.

---

## Prerequisites

- Python 3.10, 3.11, or 3.12
- 8 GB RAM minimum (16 GB recommended)
- GPU optional but recommended for NLI inference speed

---

## 1. Installation

```bash
# From PyPI (recommended)
pip install veritascore

# From source (development)
git clone https://github.com/Saqlain-mushtaq-alamin/VeritasCore.git
cd VeritasCore
pip install -e ".[dev]"
```

---

## 2. Usage Example A — Python Library (Grounded)

Grounded mode: you provide the reference document. This is the fastest and most accurate mode.

```python
from veritascore import VeritasCoreEngine

# Initialize engine (downloads models on first run, ~2 GB)
engine = VeritasCoreEngine()

# Verify an LLM response against a source document
report = engine.verify(
    response="The Eiffel Tower is 350 meters tall and was built in 1887.",
    query="Tell me about the Eiffel Tower",
    context="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, "
            "France. It is 330 metres (1,083 ft) tall and was constructed from 1887 to 1889.",
)

# Print summary
print(f"Trust Score : {report.overall_trust_score:.0%}")
print(f"Verdict     : {report.overall_verdict.value}")
print(f"Mode        : {report.verification_mode.value}")
print(f"Claims      : {len(report.claims)} total")
print(f"Time        : {report.processing_time_ms:.0f} ms")
print()

# Print per-claim details
for cv in report.claims:
    icon = {"supported": "✅", "contradicted": "❌", "unsupported": "⚠️"}.get(cv.verdict.value, "❓")
    print(f"{icon} [{cv.verdict.value.upper():12s}] {cv.claim.text}")
    print(f"   Confidence : {cv.confidence:.0%}")
    print(f"   Reason     : {cv.reason}")
    if cv.evidence:
        print(f"   Evidence   : {cv.evidence[:120]}...")
    print()
```

**Expected output:**

```
Trust Score : 50%
Verdict     : contradicted
Mode        : grounded
Claims      : 2 total
Time        : 340 ms

❌ [CONTRADICTED ] The Eiffel Tower is 350 meters tall.
   Confidence : 87%
   Reason     : Context states the Eiffel Tower is 330 metres tall, contradicting 350 metres.
   Evidence   : It is 330 metres (1,083 ft) tall...

✅ [SUPPORTED    ] The Eiffel Tower was built in 1887.
   Confidence : 91%
   Reason     : Context confirms construction began in 1887.
   Evidence   : was constructed from 1887 to 1889...
```

---

## 3. Usage Example B — Python Library (Ungrounded)

Ungrounded mode: no source document provided. VeritasCore automatically retrieves evidence from the web.

```python
from veritascore import VeritasCoreEngine

engine = VeritasCoreEngine()

report = engine.verify(
    response="Aspirin was first synthesized in 1897 by Felix Hoffmann at Bayer.",
    query="Who invented aspirin and when?",
    # No context → triggers web retrieval automatically
)

print(f"Trust Score: {report.overall_trust_score:.0%}")
for cv in report.claims:
    print(f"  [{cv.verdict.value}] {cv.claim.text}")
```

> **Note:** Ungrounded mode requires an internet connection and is slower (~2 s/claim) than grounded mode (~300 ms/claim).

---

## 4. Usage Example C — REST API + Streaming

**Start the server:**

```bash
uvicorn veritascore.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Single verification (curl):**

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "response": "Mount Everest is 8,849 meters tall.",
    "query": "How tall is Mount Everest?",
    "context": "Mount Everest stands at 8,848.86 metres above sea level."
  }'
```

**Streaming verification (Server-Sent Events):**

```bash
curl -N http://localhost:8000/stream/verify \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "response": "The speed of light is 300,000 km/s in a vacuum.",
    "query": "What is the speed of light?"
  }'
```

Each claim result streams back as it completes, so you get incremental feedback without waiting for the full response.

---

## 5. Domain Profiles

Use domain-specific thresholds for medical or legal content:

```python
report = engine.verify(
    response="Metformin is the first-line treatment for type 2 diabetes.",
    query="What is the first-line treatment for type 2 diabetes?",
    context="...",
    domain="medical",  # Applies stricter confidence thresholds
)
```

Available profiles: `"general"` (default), `"medical"`, `"legal"`.

See [docs/domain_profiles.md](domain_profiles.md) for creating custom profiles.

---

## 6. Next Steps

| Goal | Resource |
|------|----------|
| Understand the system design | [Architecture](architecture.md) |
| Deploy with Docker | [Deployment](deployment.md) |
| Full API reference | [API Reference](api_reference.md) |
| Benchmark results | [Benchmarks](benchmarks.md) |
| Custom domain profiles | [Domain Profiles](domain_profiles.md) |
