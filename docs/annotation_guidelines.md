# Annotation Guidelines for Decomposition Quality Evaluation

> **Version:** 1.0  
> **Phase:** R6 — Human Evaluation for Decomposition Quality  
> **Purpose:** Inter-annotator agreement study for claim decomposer atomic accuracy  
> **Target:** κ ≥ 0.60 (substantial agreement) on Overall Correctness

---

## Overview

This document provides guidelines for annotating the quality of claim decomposition
produced by VeritasCore's two decomposers:

- **RuleDecomposer** — rule-based, fast, CPU-only fallback
- **LLMDecomposer** — Phi-3-mini-based, higher accuracy, GPU-preferred

Each **decomposed claim** is evaluated against THREE binary criteria. A claim is
**correct** (Overall = ✓) only if **all three** criteria are satisfied.

---

## The Three Criteria

### Criterion 1: Atomicity

> **Question:** Does this claim contain exactly **ONE** independently verifiable fact?

A claim is **atomic** if:
- It asserts one and only one fact that can be verified independently.
- Removing any word or phrase would leave the claim incomplete or change its meaning.

| Label | Example |
|-------|---------|
| ✓ Atomic | "The Eiffel Tower is 330 meters tall." |
| ✗ Not Atomic | "The Eiffel Tower is 330 meters tall and was built in 1889." (two facts) |
| ✗ Not Atomic | "It is tall." (vague — not independently verifiable) |
| ✗ Not Atomic | "Python was created by Guido van Rossum, who was born in the Netherlands, in 1991." (three facts) |

**Edge cases:**
- Numerical claims with units count as one fact: "Water boils at 100°C at sea level." → ✓ Atomic
- Negation counts as one fact: "Pluto is not classified as a planet." → ✓ Atomic
- Multi-hop with who/which: "Guido van Rossum, who created Python, was born in 1956." → ✗ Not Atomic (two facts)

---

### Criterion 2: Self-Containedness

> **Question:** Can this claim be **understood and verified** without reading the original text?

A claim is **self-contained** if:
- It does not use pronouns that refer to something not named in the claim itself.
- It does not use phrases like "the aforementioned", "the above", "as stated before".
- A reader with no context of the original passage can understand what entity is being described.

| Label | Example |
|-------|---------|
| ✓ Self-contained | "The Eiffel Tower was completed in 1889." |
| ✗ Not self-contained | "It was completed in 1889." (dangling pronoun "it") |
| ✗ Not self-contained | "The tower mentioned above is in Paris." (reference to surrounding text) |
| ✗ Not self-contained | "He founded the company in 1994." (unresolved "He") |

---

### Criterion 3: Factuality Filter

> **Question:** Is this claim a **factual statement** (not an opinion, hedge, or meta-commentary)?

A claim **passes** the factuality filter if:
- It asserts a concrete, verifiable fact about the world.
- It is NOT expressing the author's opinion or belief.
- It is NOT a hedge ("I think", "perhaps", "in my view").
- It is NOT a meta-commentary ("Note that...", "It is important to understand...").

| Label | Example |
|-------|---------|
| ✓ Factual | "Water boils at 100°C at standard atmospheric pressure." |
| ✓ Factual | "Pluto was reclassified as a dwarf planet in 2006." |
| ✗ Opinion | "I believe Python is the best programming language." |
| ✗ Hedge | "Perhaps the economy will recover next year." |
| ✗ Meta | "Note that this is an important concept to understand." |

---

## Overall Correctness

A claim is **Overall Correct (✓)** if and only if **all three** criteria are satisfied:

    Overall Correct = Atomic AND Self-Contained AND Factual

---

## Step-by-Step Annotation Process

### Phase 1: Calibration
1. All annotators review these guidelines together.
2. Complete the 10 calibration samples (marked CAL-001 to CAL-010).
3. Compare labels, discuss disagreements.
4. Proceed only after calibration agreement ≥ 80%.

### Phase 2: Independent Annotation
1. Annotate each claim row in the spreadsheet independently.
2. Fill in four binary columns: atomic, self_contained, factual, overall.
3. Add optional notes for borderline cases.
4. Do NOT discuss labels with other annotators during this phase.

### Phase 3: Adjudication
1. Study coordinator generates an IAA report after all annotators finish.
2. Flagged disagreements are discussed.
3. Final adjudicated labels are recorded.

---

## Quality Control Samples

| Control ID | Claim | Atomic | Self-Contained | Factual |
|------------|-------|:------:|:--------------:|:-------:|
| CTRL-001 | The Earth orbits the Sun. | ✓ | ✓ | ✓ |
| CTRL-002 | The Earth orbits the Sun and has one natural satellite. | ✗ | ✓ | ✓ |
| CTRL-003 | It orbits in approximately 365 days. | ✓ | ✗ | ✓ |
| CTRL-004 | I believe the universe is infinite. | ✓ | ✓ | ✗ |
| CTRL-005 | The Titanic sank in 1912 after hitting an iceberg. | ✗ | ✓ | ✓ |
| CTRL-006 | The Titanic sank in 1912. | ✓ | ✓ | ✓ |
| CTRL-007 | He founded Apple in 1976. | ✓ | ✗ | ✓ |
| CTRL-008 | Steve Jobs co-founded Apple Computer Company in 1976. | ✓ | ✓ | ✓ |
| CTRL-009 | Perhaps the vaccine will prove effective. | ✓ | ✓ | ✗ |
| CTRL-010 | The Great Wall of China is approximately 21,196 kilometers long. | ✓ | ✓ | ✓ |

---

## Spreadsheet Column Reference

| Column | Type | Description |
|--------|------|-------------|
| sample_id | string | Unique sample ID (e.g. H001, F001, CAL-001) |
| source | string | Dataset: halueval, fever, or calibration |
| original_text | string | Original text that was decomposed |
| decomposer | string | Which decomposer: rule or llm |
| claim_num | int | Claim index within this sample (1-indexed) |
| claim_text | string | The decomposed claim text to annotate |
| atomic | int | 1 = Atomic, 0 = Not Atomic |
| self_contained | int | 1 = Self-contained, 0 = Not Self-contained |
| factual | int | 1 = Factual, 0 = Not Factual |
| overall | int | 1 = Correct (all three pass), 0 = Incorrect |
| annotator | string | Annotator ID (e.g. A1, A2, A3) |
| notes | string | Optional notes for borderline cases |

---

## Inter-Annotator Agreement Targets

| Criterion | Target κ | Interpretation |
|-----------|----------|----------------|
| Atomicity | ≥ 0.60 | Substantial agreement |
| Self-Containedness | ≥ 0.60 | Substantial agreement |
| Factuality | ≥ 0.60 | Substantial agreement |
| **Overall Correctness** | **≥ 0.60** | **Primary metric** |

Cohen's κ (two annotators) / Fleiss' κ (three or more).

### Landis & Koch (1977) Scale:

| κ Range | Agreement Level |
|---------|-----------------|
| < 0.00 | Poor |
| 0.00 – 0.20 | Slight |
| 0.21 – 0.40 | Fair |
| 0.41 – 0.60 | Moderate |
| 0.61 – 0.80 | Substantial |
| 0.81 – 1.00 | Almost perfect |

---

## FAQ

**Q: What if a claim is almost self-contained but uses a slightly ambiguous pronoun?**
A: If reasonable doubt exists, label as ✗ Not Self-Contained. When in doubt, mark 0.

**Q: What if a claim is factual but clearly wrong (e.g., "The Eiffel Tower is 500m tall")?**
A: Label as ✓ Factual. We annotate FORM not TRUTH VALUE. Verifying truth is the verifier's job.

**Q: What about quotes and reported speech?**
A: Treat as factual claims about what was said. "Einstein said 'God does not play dice'." → ✓ Atomic, ✓ Self-contained, ✓ Factual.

**Q: What if the claim contains a list?**
A: If a single claim contains a list of items, it is ✗ Not Atomic.

**Q: What about negations?**
A: Negations are fine. "Pluto is not a planet." → ✓ Atomic.
