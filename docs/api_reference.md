# API Reference — VeritasCore

> Complete reference for the REST API exposed by `veritascore.api`.

---

## Base URL

```
http://localhost:8000
```

---

## Authentication

No authentication required for local deployment. For production, set `API_KEY` in `.env` and pass it as:

```
Authorization: Bearer <API_KEY>
```

---

## Endpoints

### `GET /health`

Health check endpoint. Returns service status and model loading state.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response `200 OK`:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true,
  "uptime_seconds": 42.3
}
```

---

### `POST /verify`

Verify an LLM response and return a full verification report.

**Request body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `response` | `string` | ✅ | The LLM response text to verify |
| `query` | `string` | ❌ | The original query/prompt sent to the LLM |
| `context` | `string` | ❌ | Reference document for grounded verification |
| `domain` | `string` | ❌ | Domain profile: `"general"`, `"medical"`, `"legal"` (default: `"general"`) |
| `max_claims` | `integer` | ❌ | Maximum number of claims to verify (default: 20) |

**Example request:**
```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{
    "response": "The Eiffel Tower is 350 meters tall and was built in 1889.",
    "query": "Tell me about the Eiffel Tower",
    "context": "The Eiffel Tower is 330 metres tall, constructed 1887–1889.",
    "domain": "general"
  }'
```

**Response `200 OK`:**
```json
{
  "overall_trust_score": 0.51,
  "overall_verdict": "contradicted",
  "verification_mode": "grounded",
  "domain": "general",
  "processing_time_ms": 342.7,
  "claims": [
    {
      "id": "claim_0",
      "text": "The Eiffel Tower is 350 meters tall.",
      "verdict": "contradicted",
      "confidence": 0.87,
      "reason": "Context states the Eiffel Tower is 330 metres tall, contradicting 350 metres.",
      "evidence": "The Eiffel Tower is 330 metres tall, constructed 1887–1889.",
      "nli_score": 0.03,
      "contradiction_score": 0.81,
      "reverse_entailment_score": 0.12
    },
    {
      "id": "claim_1",
      "text": "The Eiffel Tower was built in 1889.",
      "verdict": "supported",
      "confidence": 0.91,
      "reason": "Context confirms construction was completed by 1889.",
      "evidence": "constructed 1887–1889.",
      "nli_score": 0.88,
      "contradiction_score": 0.04,
      "reverse_entailment_score": 0.79
    }
  ]
}
```

**Error responses:**

| Code | Meaning |
|------|---------|
| `400` | Invalid request (empty response, unknown domain) |
| `422` | Request validation error (Pydantic schema mismatch) |
| `503` | Models not yet loaded (retry after a few seconds) |
| `500` | Internal server error |

**Error body example (`400`):**
```json
{
  "detail": "response field cannot be empty"
}
```

---

### `POST /stream/verify`

Streaming verification — returns claim verdicts incrementally as Server-Sent Events (SSE).

**Request body:** Same schema as `POST /verify`.

**Response**: `text/event-stream`

Each event is a JSON object. Events are emitted as each claim is verified:

```
data: {"event": "claim", "claim": {"id": "claim_0", "text": "...", "verdict": "contradicted", ...}}

data: {"event": "claim", "claim": {"id": "claim_1", "text": "...", "verdict": "supported", ...}}

data: {"event": "done", "summary": {"overall_trust_score": 0.51, "overall_verdict": "contradicted", "processing_time_ms": 345.2}}
```

**Example (curl):**
```bash
curl -N -X POST http://localhost:8000/stream/verify \
  -H "Content-Type: application/json" \
  -d '{
    "response": "Water boils at 100°C at sea level. It freezes at 0°C.",
    "query": "At what temperature does water boil and freeze?"
  }'
```

**Example (Python):**
```python
import httpx

with httpx.stream("POST", "http://localhost:8000/stream/verify",
                  json={"response": "...", "query": "..."}) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event["event"] == "claim":
                print(event["claim"])
```

---

### `GET /models`

List loaded models and their status.

**Response `200 OK`:**
```json
{
  "nli_model": {
    "name": "cross-encoder/nli-deberta-v3-large",
    "loaded": true,
    "device": "cuda:0"
  },
  "decomposer": {
    "name": "llama3.2",
    "loaded": true,
    "backend": "ollama"
  }
}
```

---

## Python SDK

The same functionality is available as a Python library without running a server:

```python
from veritascore import VeritasCoreEngine

engine = VeritasCoreEngine(domain="general")

# Synchronous
report = engine.verify(response="...", query="...", context="...")

# Async
import asyncio
report = asyncio.run(engine.averify(response="...", query="...", context="..."))

# Streaming
for claim_verdict in engine.stream_verify(response="...", query="..."):
    print(claim_verdict)
```

---

## Request / Response Schema (Pydantic)

```python
class VerifyRequest(BaseModel):
    response: str                          # LLM output to verify
    query: str | None = None               # Original user query
    context: str | None = None             # Reference document
    domain: str = "general"               # Domain profile
    max_claims: int = 20                   # Claim limit

class ClaimVerdictResponse(BaseModel):
    id: str
    text: str
    verdict: Literal["supported", "contradicted", "unsupported"]
    confidence: float
    reason: str
    evidence: str | None
    nli_score: float | None
    contradiction_score: float | None
    reverse_entailment_score: float | None

class VerifyResponse(BaseModel):
    overall_trust_score: float
    overall_verdict: Literal["supported", "contradicted", "unsupported"]
    verification_mode: Literal["grounded", "ungrounded", "consistency"]
    domain: str
    processing_time_ms: float
    claims: list[ClaimVerdictResponse]
```

---

## Interactive API Docs

When the server is running, Swagger UI is available at:

```
http://localhost:8000/docs
```

ReDoc is available at:

```
http://localhost:8000/redoc
```
