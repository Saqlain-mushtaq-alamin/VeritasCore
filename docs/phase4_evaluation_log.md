# Phase 4 — Quality Gate G4 Evaluation Log




## 1. What Was Validated in This Sandbox

| Check | Method | Result |
|---|---|---|
| Query-claim relevance scoring (on-topic/off-topic/partial) | Stub model, hand-picked 2D unit vectors | Pass (5 tests) |
| Raw-vs-clamped similarity distinction for the off-topic threshold | Stub model, negative-similarity vector | Pass (regression test — see §2) |
| Coherence scoring (consistent vs. inconsistent claim sets, single-claim edge case) | Stub model | Pass (4 tests) |
| Empty claims / empty response / missing query handling | Stub model + pure validation | Pass (5 tests) |
| Response-query relevance | Stub model | Pass (2 tests) |
| `score_claim()` single-claim API | Stub model | Pass (2 tests) |
| Init validation (`relevance_threshold` bounds) | Pure validation | Pass (2 tests) |
| `ConsistencyResult.to_dict()` / `__repr__` | Pure | Pass (2 tests) |
| Lifecycle (`is_available()` failure path, `_resolve_device()` all branches, `unload()`) | Mocked `_load_model`/`torch.cuda.is_available` | Pass (7 tests) |
| Fixture file structural validity (10 queries, 70 claim instances, all required fields) | Pure JSON validation | Pass (3 tests) |
| End-to-end decompose -> consistency-check pipeline | `RuleDecomposer` + stub embedding model | Pass (manual smoke test, see below) |
| Lint (ruff) | `ruff check` | Clean |
| Type check (mypy) | `mypy --ignore-missing-imports` | Clean |
| Unit test coverage | `pytest --cov` | 85% on `consistency.py` (gap = real model-loading code, by design — same pattern as every other phase's model-loading module) |

**Total: 31 new unit tests, 231/231 project-wide unit tests pass.**

### Manual end-to-end smoke test (RuleDecomposer + stub embedding model)

Reproduces the spec's own canonical example exactly:

```
query    = "What is the capital of France?"
response = "Paris is the capital of France. The Eiffel Tower is 330 meters tall."

ConsistencyResult(n_claims=2, coherence=0.06, response_relevance=0.00, n_off_topic=1)
  'Paris is the capital of France.':       relevance=1.00  off_topic=False
  'The Eiffel Tower is 330 meters tall.':  relevance=0.00  off_topic=True
```

The Eiffel-Tower claim is factually true but non-responsive to a capital-city
question — correctly flagged, exactly matching the module's stated purpose.

---

## 2. Deviations From the Reference Spec

The reference implementation in the Phase 4 plan had one subtle correctness gap
and one missing safety check that the spec's own test list implied
(`test_no_query_raises`) but the reference code never actually implemented:

1. **No empty/missing-query validation.** The reference `check_consistency()`
   and `score_claim()` would silently encode an empty string as the query and
   produce a meaningless-but-not-erroring similarity score, rather than
   surfacing the problem. Added an explicit `VerificationError` when `query`
   is `None`, empty, or whitespace-only — directly fulfilling the spec's own
   listed test case `test_no_query_raises`, which the reference implementation
   did not actually satisfy (it only documented "should handle missing query
   gracefully" without raising, contradicting the test name "raises").

2. **Clamping could mask the off-topic threshold check in a refactor-fragile
   way.** The reference code computed
   `claim_scores[claim.id] = max(0.0, min(1.0, sim))` and then checked
   `if sim < self.relevance_threshold` using the same already-computed `sim`
   variable, so this particular ordering happened to be safe. This
   implementation makes the distinction explicit and structural instead:
   `raw_sim` (used for the threshold decision) is a clearly separate variable
   from `clamped` (used for the reported score), with a dedicated regression
   test (`test_clamping_does_not_hide_negative_similarity_from_threshold`)
   asserting a claim with strongly negative similarity is both reported as
   `0.0` (clamped, matching the documented `[0.0, 1.0]` per-claim contract)
   and still correctly appears in `off_topic_claims`.

3. **`coherence_score` and `response_relevance` were clamped to `[0.0, 1.0]`
   in the reference**, silently collapsing genuinely anti-correlated
   (opposite-topic) claim sets to the same `0.0` floor as merely "unrelated"
   claim sets — losing exactly the signal that would make criterion 3
   ("coherence score distinguishes consistent vs inconsistent claim sets")
   most informative. Changed these two fields to clamp to the embeddings'
   true mathematical range `[-1.0, 1.0]` instead (per-claim `claim_scores`
   still clamp to `[0.0, 1.0]`, matching the documented interface contract
   and `ClaimVerdict.consistency_score`'s Pydantic bounds). Covered by
   `test_coherence_mixed_claims` asserting `coherence_score < 0.0` for an
   opposite-topic pair, which would have been impossible to observe under
   the reference's `[0,1]` clamping.

---

## 3. Pending Hardware Validation (Required Before G4 Sign-off)

Run on a machine with the embedding model downloaded
(`python scripts/download_models.py`):

```bash
# Integration tests against the real all-MiniLM-L6-v2 model, including the
# full curated-fixture G4 criteria 1/2/6 evaluation
pytest tests/integration/test_consistency_real_model.py -v -m integration

# Standalone CLI evaluation (same fixture, human-readable pass/fail report)
python scripts/evaluate_consistency.py --verbose
```

### Specific things to confirm on real hardware

1. **Model load time** (<2s, G4 criterion 4): `test_model_loads_quickly`
   asserts this against a fresh (unloaded) checker instance.
2. **Batch encoding speed** (50 claims in <1s, G4 criterion 5):
   `test_batch_encoding_50_claims_under_one_second` constructs exactly 50
   synthetic claims and times a single `check_consistency()` call.
3. **On-topic accuracy** (>0.5 on >90% of samples, G4 criterion 1) and
   **off-topic accuracy** (<0.3 on >80% of samples, G4 criterion 2): both
   evaluated against all 70 claim instances in the curated fixture via
   `TestCuratedFixtureFullEvaluation`. The stub-model unit tests validate
   the logic is correct given known similarity values; only a real model
   run can confirm MiniLM's actual embeddings produce scores in the
   expected ranges for these specific curated examples.
4. **off_topic_claims list detection rate** (>80%, G4 criterion 6):
   `test_off_topic_claims_flagged_in_off_topic_list_80pct` runs the full
   `check_consistency()` path (not just `score_claim()`) per query, mixing
   on-topic and off-topic claims together as `RuleDecomposer` output would,
   to also exercise the batch-encoding code path under realistic conditions.
5. **VRAM coexistence with NLIVerifier** (spec section 4.9: "MiniLM is
   tiny... can coexist with the NLI model in VRAM. No need to unload."):
   not directly tested here — confirm via `nvidia-smi` that loading both
   `NLIVerifier` and `SemanticConsistencyChecker` simultaneously stays well
   under typical 8GB budgets (NLI is about 1.5GB plus MiniLM at about
   80MB, a small fraction of that headroom).

---

## 4. Acceptance Criteria Status (Quality Gate G4)

| # | Criterion | Status |
|---|---|---|
| 1 | On-topic claims score >0.5 relevance on >90% of test samples | Logic verified (stub model); needs real-model hardware run |
| 2 | Off-topic claims score <0.3 relevance on >80% of test samples | Logic verified (stub model); needs real-model hardware run |
| 3 | Coherence score distinguishes consistent vs inconsistent claim sets | Verified — `test_coherence_distinguishes_consistent_vs_inconsistent` asserts a strict ordering between the two cases |
| 4 | Embedding model (MiniLM) loads in <2s and uses <500MB VRAM | Load-time check ready (`test_model_loads_quickly`); needs hardware run. VRAM not directly measurable in this sandbox |
| 5 | Batch encoding processes 50 claims in <1s | Test ready (`test_batch_encoding_50_claims_under_one_second`); needs hardware run |
| 6 | `off_topic_claims` list correctly flags >80% of planted off-topic items | Logic verified (stub model); needs real-model hardware run |
| 7 | All unit tests pass | 231/231 pass |
