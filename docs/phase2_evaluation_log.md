# Phase 2 — Quality Gate G2 Evaluation Log

> This sandbox has no GPU and cannot download the ~700MB NLI model, so the AUROC
> benchmark against HaluEval/FEVER could not be executed here. This log records
> what WAS validated (logic correctness via mocks + a manual mocked pipeline run)
> and what MUST be re-run on real hardware before sign-off.

---

## 1. What Was Validated in This Sandbox

| Check | Method | Result |
|---|---|---|
| `BaseVerifier` abstract contract | pytest | Pass |
| Verdict logic (SUPPORTED/CONTRADICTED/UNSUPPORTED thresholds) | Mocked `_run_nli` | Pass (16 tests) |
| Chunk/probs/evidence consistency (see §2 — bug fix) | Mocked `_run_nli` across multiple chunks | Pass |
| Context chunking (token-bounded, overlapping) | `_FakeTokenizer` | Pass (6 tests) |
| `verify_batch()` == `verify()` equivalence | Mocked tokenizer/model/softmax | Pass |
| Evidence snippet extraction (word-overlap scoring) | Pure function, no model | Pass (4 tests) |
| Model lifecycle (lazy load, `unload()`, label-order discovery) | Mocked model config | Pass (6 tests) |
| End-to-end decompose -> verify pipeline | `RuleDecomposer` + mocked `NLIVerifier` | Pass (manual smoke test, see below) |
| Lint (ruff) | `ruff check` | Clean |
| Type check (mypy) | `mypy --ignore-missing-imports` | Clean |
| Unit test coverage | `pytest --cov` | 76% on `nli_verifier.py` (gap = real model-loading code, by design — same pattern as `llm_decomposer.py` in Phase 1) |

**Total: 41/41 new unit tests pass, 136/136 project-wide unit tests pass.**

### Manual end-to-end smoke test (mocked NLI, real RuleDecomposer)

```
response = "The Eiffel Tower is 330 meters tall. It was completed in 1900."
context  = "The Eiffel Tower stands 330 metres tall in Paris and was completed in 1889."

[supported   ] 'The Eiffel Tower is 330 meters tall.'   nli_score=0.85
[contradicted] 'It was completed in 1900.'              nli_score=0.10
```

The pipeline correctly distinguishes a true claim (height matches context) from
a false claim (wrong completion year) using the same context, confirming the
decompose -> verify wiring is correct end-to-end.

---

## 2. Bug Found and Fixed vs. the Reference Spec

The Phase 2 spec's reference implementation tracked `best_entailment` and
`best_contradiction` as two independent running maxima across context chunks,
each with its own conditionally-updated `best_chunk` / `best_probs`. This can
produce an inconsistent result: e.g. chunk A has entailment=0.6, contradiction=0.1;
chunk B has entailment=0.3, contradiction=0.7. The spec's logic can end up
reporting `best_chunk = B` (since contradiction=0.7 is the global max signal)
while `entail_prob` used for `nli_score` was tracked from chunk A's 0.6 — the
displayed evidence chunk and the score driving the verdict can come from
different chunks.

**Fix applied** (`_verify_single_claim` and `verify_batch`): for each claim, select
the single best chunk by `max(entailment, contradiction)` across all chunks, then
derive verdict, confidence, `nli_score`, and evidence all from that one chunk's
probability vector. This guarantees the three are always mutually consistent.
Regression-covered by `TestNLIVerifierChunkConsistency` (3 tests simulating
multi-chunk contexts where the naive approach would diverge).

A second, smaller deviation from the spec: label order is now discovered from
the model's `config.id2label` at load time (`_discover_label_order()`) rather
than hardcoded, since the developer note in §2.10 explicitly flags this as
something to "verify experimentally before trusting." Falls back to the
documented default `[contradiction, neutral, entailment]` if the model's label
strings don't match the expected vocabulary. Covered by
`test_discover_label_order_from_config` and
`test_discover_label_order_falls_back_on_unknown_labels`.

---

## 3. Pending Hardware Validation (Required Before G2 Sign-off)

Run on a machine with the NLI model downloaded (`python scripts/download_models.py`)
and HaluEval/FEVER cached (`python scripts/download_datasets.py`):

```bash
# Integration tests against the real cross-encoder/nli-deberta-v3-base model
pytest tests/integration/test_grounded_pipeline.py -v -m integration

# AUROC benchmark — the actual G2 acceptance criterion
python scripts/benchmark_nli_verifier.py --dataset halueval --n 200
python scripts/benchmark_nli_verifier.py --dataset fever --n 200
```

**Target**: AUROC >= 0.72 (SummaC NLI-only baseline, Laban et al. 2022).

### Specific things to confirm on real hardware

1. **Label order assumption**: `test_label_order_matches_expected` in the
   integration suite checks that `cross-encoder/nli-deberta-v3-base`'s
   `id2label` actually contains "entailment" and "contradiction" substrings,
   validating the `_discover_label_order()` fallback logic actually engages
   correctly.
2. **Latency target** (<200ms/claim on GPU, §2.8 criterion 10): the integration
   test `test_latency_per_claim` asserts this on GPU and a generous 5s ceiling
   on CPU.
3. **VRAM check** (§2.8 criterion 8: fits in 8GB with batch=16 headroom): not
   testable without a GPU; manually confirm via `nvidia-smi` during a batch run.
4. **`unload()` GPU memory check** (§2.8 criterion 9): `test_unload_frees_gpu_memory`
   skips automatically when no CUDA device is present — will run for real on
   GPU hardware.

---

## 4. Acceptance Criteria Status (Quality Gate G2)

| # | Criterion | Status |
|---|---|---|
| 1 | `BaseVerifier` interface is abstract and importable | Verified in sandbox |
| 2 | `NLIVerifier` loads DeBERTa model successfully | Needs GPU hardware run |
| 3 | All unit tests pass with mocked model | 41/41 pass |
| 4 | AUROC >= 0.72 on HaluEval QA subset | Needs GPU hardware run |
| 5 | Context chunking handles documents >512 tokens | Verified (unit tests) |
| 6 | Batch verification produces identical results to sequential | Verified (mocked); re-confirm on real model via `test_batch_results_match_sequential_real_model` |
| 7 | Evidence snippet is populated for every verdict | Verified |
| 8 | NLI model fits in 8GB VRAM with room for batch=16 | Needs GPU hardware run |
| 9 | `unload()` frees GPU memory | Logic verified; GPU assertion needs hardware |
| 10 | Latency <200ms per claim (grounded mode, GPU) | Needs GPU hardware run |
