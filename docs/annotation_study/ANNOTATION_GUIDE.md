# Human Annotation Protocol: Atomic Claim Decomposition (Table 4)

## Overview & Objective
You are evaluating the quality of decomposed atomic claims extracted from natural language sentences.
For each row in your spreadsheet (`annotator_X_blinded.csv`), you must evaluate candidate claims against **three binary criteria (0 = No, 1 = Yes)**.

A claim is deemed **Overall Correct** (`overall = 1`) if and only if **all three criteria equal 1**.

---

## The Three Binary Criteria

### 1. Atomicity (`atomic` column: 1 or 0)
* **Definition:** Does the candidate claim assert exactly **ONE** atomic, independently verifiable proposition?
* **Score 1 (Atomic):**
  - Asserts a single fact: *"Marie Curie was born in Warsaw."*
  - Numerical facts with units/dates: *"Python 3.0 was released on December 3, 2008."*
  - Proper noun / entity identification: *"Guido van Rossum is the author of Python."*
* **Score 0 (Not Atomic):**
  - Compound sentences connected by conjunctions ("and", "but", "while"): *"Marie Curie was born in Warsaw and later moved to Paris."* (Two facts)
  - Vague fragments that cannot be evaluated: *"It is significant."*

### 2. Self-Containedness (`self_contained` column: 1 or 0)
* **Definition:** Can the claim be completely understood and verified in isolation **without** reading the original source sentence?
* **Score 1 (Self-Contained):**
  - References explicit entities: *"The Amazon River is the largest river by discharge volume in the world."*
* **Score 0 (Not Self-Contained):**
  - Dangling pronouns: *"He held over 300 patents."* (Who is 'He'?)
  - Contextual deictic references: *"The aforementioned tower was completed in 1889."*

### 3. Factuality (`factual` column: 1 or 0)
* **Definition:** Does the claim represent a factual assertion rather than subjective opinion, advice, or speculation?
* **Score 1 (Factual):**
  - Statements of objective fact: *"The moon orbits Earth."*
* **Score 0 (Not Factual / Opinion):**
  - Subjective opinions: *"Python is the best programming language for beginners."*
  - Speculations / hedges: *"It is possible that life exists on Europa."*

---

## Overall Judgment (`overall` column: 1 or 0)
* **1 (Correct Decomposition):** `atomic == 1` AND `self_contained == 1` AND `factual == 1`.
* **0 (Defective Decomposition):** If any of the three criteria is `0`.

---

## Instructions for Annotators
1. Open your assigned spreadsheet: `annotator_X_blinded.csv`.
2. Do **not** alter the `item_id`, `original_text`, or `claim_text` columns.
3. For each row, enter `0` or `1` in columns `atomic`, `self_contained`, `factual`, and `overall`.
4. Enter any brief notes in the `notes` column if a case is ambiguous.
5. Save the completed file in UTF-8 CSV format.

---

## Inter-Annotator Agreement (IAA) Computation
Once 2 or 3 annotators return their filled spreadsheets, compute Cohen's kappa (2 annotators) or Fleiss' kappa (3 annotators):
```bash
python scripts/compute_iaa.py \
    --csv docs/annotation_study/annotator_1_blinded.csv \
    --csv-a2 docs/annotation_study/annotator_2_blinded.csv
```
Target threshold for publication: $\kappa \ge 0.60$ (Substantial Agreement, Landis & Koch).
