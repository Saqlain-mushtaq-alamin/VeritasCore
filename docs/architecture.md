# Architecture — VeritasCore

> Deep-dive into how VeritasCore works internally.

---

## Overview

VeritasCore is a **multi-signal hallucination detection pipeline**. Given an LLM response, it:

1. Decomposes the response into *atomic, independently verifiable claims*
2. Routes each claim to the appropriate verification mode
3. Runs up to three verification signals in parallel (NLI, retrieval, consistency)
4. Fuses signals into a calibrated trust score
5. Generates explanations with evidence and human-readable reasons

---

## Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VeritasCoreEngine                            │
│                                                                     │
│  Input: response, query?, context?, domain?                         │
│                                                                     │
│  ┌────────────┐    ┌───────────┐                                    │
│  │ Decomposer │───►│  Router   │                                    │
│  │ (LLM-based)│    │(mode sel.)│                                    │
│  └────────────┘    └─────┬─────┘                                    │
│                          │                                          │
│           ┌──────────────┼──────────────┐                          │
│           ▼              ▼              ▼                           │
│    ┌─────────────┐ ┌──────────┐ ┌─────────────┐                   │
│    │NLI Verifier │ │Retrieval │ │Consistency  │                   │
│    │(DeBERTa-v3) │ │Verifier  │ │Verifier     │                   │
│    └──────┬──────┘ └────┬─────┘ └──────┬──────┘                   │
│           │             │              │                            │
│           └──────┬───────┘              │                           │
│                  ▼                     │                            │
│          ┌─────────────┐               │                            │
│          │Fusion Scorer│◄──────────────┘                           │
│          │(calibrated) │                                            │
│          └──────┬──────┘                                            │
│                 ▼                                                   │
│          ┌─────────────┐                                            │
│          │  Explainer  │                                            │
│          │(reason+evid)│                                            │
│          └──────┬──────┘                                            │
│                 ▼                                                   │
│        VerificationReport                                           │
│    (overall_trust_score, claims[])                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Decomposer (`veritascore.decomposer`)

**Purpose**: Break an LLM response into atomic, independently verifiable claims.

**Implementation**: Prompt-based extraction using a local LLM (via Ollama) or an OpenAI-compatible endpoint.

**Atomicity criteria**:
- One claim = one verifiable fact
- No compound sentences ("A and B" → two claims)
- No references ("it", "they") — fully self-contained

**Interface**:
```python
class ClaimDecomposer:
    def decompose(self, response: str, query: str | None = None) -> list[Claim]: ...
```

**Model**: Any instruction-following LLM (default: `llama3.2` via Ollama). Falls back to regex-based sentence splitting if no LLM is available.

**Evaluation**: > 90% of claims correctly atomic on manual HaluEval subset (Phase 1 goal).

---

### 2. Router (`veritascore.router`)

**Purpose**: Select the appropriate verification mode based on what inputs are available.

| Condition | Mode Selected |
|-----------|--------------|
| `context` provided | `GROUNDED` |
| No context, internet available | `UNGROUNDED` |
| Offline or no search keys | `CONSISTENCY` |

**Interface**:
```python
class VerificationRouter:
    def route(self, claims: list[Claim], context: str | None, ...) -> VerificationMode: ...
```

---

### 3. NLI Verifier (`veritascore.verifier`)

**Purpose**: Check each claim against a provided context using Natural Language Inference.

**Model**: `cross-encoder/nli-deberta-v3-large` — a fine-tuned DeBERTa-v3-large cross-encoder producing `[contradiction, neutral, entailment]` logits for (premise, hypothesis) pairs.

**Scoring formula** (grid-searched on HaluEval QA, n=200):
```
hallucination_score = fwd_contradiction - 0.2 × fwd_entailment - 0.6 × rev_entailment
```

Where:
- `fwd_contradiction` = P(contradiction | context → claim)
- `fwd_entailment` = P(entailment | context → claim)
- `rev_entailment` = P(entailment | claim → context) — claim implies context = strong support signal

**Context chunking**: Long contexts are split into 512-token chunks; scores are max-pooled across chunks.

**Output fields** on `ClaimVerdict`:
- `nli_score` — forward entailment probability
- `contradiction_score` — forward contradiction probability
- `reverse_entailment_score` — reverse entailment probability

**Benchmark**: AUROC 0.7208 on HaluEval QA (n=200) — meets Quality Gate G2 (≥0.72).

---

### 4. Retrieval Verifier (`veritascore.retriever`)

**Purpose**: Retrieve external evidence passages for ungrounded verification.

**Sources** (in priority order):
1. Wikipedia API (free, no key required)
2. Bing Search API (requires `BING_API_KEY`)
3. Generic web search fallback

**Passage scoring**: Retrieved passages are ranked by semantic similarity to the claim using the NLI verifier's entailment score.

**Interface**:
```python
class RetrievalVerifier:
    async def verify(self, claims: list[Claim], query: str) -> list[ClaimVerdict]: ...
```

---

### 5. Consistency Verifier (`veritascore.scorer`)

**Purpose**: Estimate claim consistency by checking whether the LLM produces the same claim across multiple stochastic re-samples.

**Algorithm**:
1. Re-prompt the LLM `n` times (default n=5) at temperature > 0
2. Extract the target claim from each response
3. Use NLI to compare claim against each re-sampled response
4. Consistency score = mean entailment probability across re-samples

High consistency → claim likely reliable.
Low consistency → LLM is uncertain → possible hallucination.

---

### 6. Fusion Scorer (`veritascore.scorer`)

**Purpose**: Combine multiple verification signals into a single calibrated trust score.

**Method**: Logistic regression over the signal vector `[nli_score, retrieval_score, consistency_score, signal_count]`. Calibrated using Platt scaling.

**Training**: Trained on HaluEval + FEVER labels. Stored as a lightweight pickle model.

**Output**: `overall_trust_score ∈ [0, 1]` where 1.0 = fully supported.

**Threshold** (domain-configurable):
- `score ≥ 0.7` → `SUPPORTED`
- `score ≤ 0.3` → `CONTRADICTED`
- otherwise → `UNSUPPORTED`

---

### 7. Explainer (`veritascore.explainer`)

**Purpose**: Generate human-readable reasons and extract supporting evidence passages.

**Evidence extraction**: Locates the highest-scoring passage from retrieved/provided context and truncates to ≤ 300 characters.

**Reason generation**: Template-based with LLM fallback. Examples:
- "Context states X, contradicting Y in the claim."
- "No evidence found to support this claim."
- "Evidence confirms this claim with high confidence."

---

## Data Types (`veritascore.core.types`)

```python
@dataclass
class Claim:
    id: str
    text: str
    source_span: tuple[int, int]
    source_text: str

class Verdict(Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"

class VerificationMode(Enum):
    GROUNDED = "grounded"
    UNGROUNDED = "ungrounded"
    CONSISTENCY = "consistency"

@dataclass
class ClaimVerdict:
    claim: Claim
    verdict: Verdict
    confidence: float           # 0–1
    reason: str
    evidence: str | None
    nli_score: float | None
    contradiction_score: float | None
    reverse_entailment_score: float | None

@dataclass
class VerificationReport:
    claims: list[ClaimVerdict]
    overall_verdict: Verdict
    overall_trust_score: float  # 0–1
    verification_mode: VerificationMode
    domain: str
    processing_time_ms: float
```

---

## Cross-Phase Contracts

| From Phase | To Phase | Contract |
|-----------|---------|---------|
| Decomposer | Router | `list[Claim]` with `id`, `text`, `source_span` |
| Router | Verifiers | `VerificationMode` + routed `list[Claim]` |
| Verifiers | Fusion | `list[ClaimVerdict]` with signal scores |
| Fusion | Explainer | `ClaimVerdict` with `overall_trust_score` |
| Explainer | API/Library | `VerificationReport` |

---

## Model Specifications

| Model | Task | Size | VRAM |
|-------|------|------|------|
| `cross-encoder/nli-deberta-v3-large` | NLI | ~900 MB | ~2 GB |
| `llama3.2` (Ollama) | Decomposer | ~2 GB | ~4 GB |
| Fusion model | Scoring | < 1 MB | CPU |
