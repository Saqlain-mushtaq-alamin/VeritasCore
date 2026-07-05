# Phase 6 — Quality Gate G6 Evaluation Log

All Phase 6 deliverables are logic-only (no ML model required) so this
phase was fully validated in the sandbox. 344/344 unit tests pass, 100%
coverage on all Phase 6 source files.

---

## 1. What Was Validated in This Sandbox

| Check | Method | Result |
|---|---|---|
| `SpanMapper.map_spans()` — exact span, substring fallback, zero-length skip | Pure string ops | Pass |
| `SpanMapper._resolve_overlaps()` — non-overlapping, higher-severity wins, ties by position, 3-way chain | Direct unit tests | Pass |
| `SpanMapper.render_annotated_text()` / `get_verdict_summary()` | String manipulation | Pass |
| `EvidenceLinker.build_chains()` — signals populated, source extraction, Source URL regex | Pure | Pass |
| `EvidenceLinker.validate_traceability()` — G6 criterion 1 and 2 | Pure | Pass |
| `EvidenceLinker.format_chain_text()` — complete and missing evidence paths | Pure | Pass |
| `DomainProfile.from_yaml()` — nested dict to sub-model coercion (regression fix) | YAML round-trip | Pass |
| `DomainProfile.apply_to_nli_verifier()` and `apply_to_consistency_checker()` — G6 criterion 6 | FakeVerifier duck type | Pass |
| `DomainProfile.get_trust_level()` — ok/warning/critical thresholds | Pure | Pass |
| `ProfileRegistry` — scan, get, get_or_default, register, load, missing dir, corrupted YAML | tmp_path-based | Pass |
| Bundled profile YAMLs (general, medical, legal, finance) all parse correctly | Real YAML load | Pass |
| Module-level convenience functions | Registry swap pattern | Pass |
| End-to-end smoke test (span map + evidence chain + profile apply) | All Phase 1-6 modules | Pass |
| Lint (ruff) | `ruff check` | Clean |
| Type check (mypy) | `mypy --ignore-missing-imports` | Clean (0 issues in 6 files) |
| Coverage | `pytest --cov` | 100% on all 6 Phase 6 source files |

**Total: 62 new unit tests, 344/344 project-wide unit tests pass.**

### Manual end-to-end smoke test output

```
Spans: 2
  [supported] 'The Eiffel Tower is 330 meters tall.'
  [contradicted] 'It was completed in 1900.'
Annotated: The Eiffel Tower is 330 meters tall. [CONTRADICTED: "It was completed in 1900."]...
Summary: {'supported': 1, 'contradicted': 1, 'unsupported': 0}
Chains: 2, Issues: []
After profile apply: entailment=0.85
Trust level 0.45: critical
Trust level 0.80: ok
Bundled profiles: ['finance', 'general', 'legal', 'medical']
General profile: entailment_threshold=0.7
ALL PASS
```

---

## 2. Bugs Found and Fixed vs. the Reference Spec

**Bug 1: `from_yaml` used `cls(**data)` instead of `model_validate()`.**
Pydantic v2's `__init__` does NOT recursively coerce nested dict values into
sub-model instances when called via `**kwargs`. A YAML with nested sub-keys
(e.g. `thresholds: {entailment_threshold: 0.85}`) would raise a
`ValidationError` when passed as `DomainProfile(thresholds={"entailment_threshold": 0.85})`
because Pydantic sees a plain dict where it expects a `ThresholdConfig`.
Fixed with `cls.model_validate(data)`, which handles recursive coercion.
Covered by `test_nested_sub_models_coerced_from_dict` and
`test_bundled_profile_yamls_load`.

**Bug 2: `apply_to_nli_verifier()` listed as G6 criterion 6 but never
implemented.** The spec lists criterion 6 as "Profile thresholds correctly
modify verifier behavior" but the reference code has no such method anywhere
on `DomainProfile`. Added `apply_to_nli_verifier()` and
`apply_to_consistency_checker()` to `DomainProfile`. Covered by
`TestApplyToVerifier`.

**Bug 3: `_resolve_overlaps` three-way overlap could silently drop
highest-severity span.** The reference did not sort spans before processing.
In specific orderings, a CONTRADICTED span arriving between two lower-severity
overlaps could be displaced even when it should win. Fixed by sorting all
spans by start position (then descending severity for ties) before resolving,
so the highest-severity span in any group is always the first incumbent
considered. Covered by `test_three_way_overlap_highest_severity_wins`.

**Bug 4: `evidence_source` for ungrounded URL was never extracted.**
The reference spec had a placeholder comment instead of an implementation.
Added `_extract_source()` with the `[Source: url]` regex matching
`RetrievalVerifier`'s annotation format. Covered by `test_source_ungrounded_with_url`.

---

## 3. Acceptance Criteria Status (Quality Gate G6)

| # | Criterion | Status |
|---|---|---|
| 1 | Every flagged claim has an evidence snippet and reason | Verified — `validate_traceability()` checks and flags missing reasons |
| 2 | Contradicted claims always have evidence | Verified — explicit check in `validate_traceability()` |
| 3 | Span mapper correctly maps 100% of test claims | Verified — 13 span mapper tests, all passing |
| 4 | Medical profile uses higher thresholds than general | Verified — `medical.yaml` has entailment=0.85 vs `general.yaml`'s 0.70 |
| 5 | Legal profile has restricted source list | Verified — `legal.yaml` has `blocked_domains` and narrowed `allowed_sources` |
| 6 | Profile thresholds correctly modify verifier behavior | Verified — `apply_to_nli_verifier()` implemented and tested |
| 7 | ProfileRegistry loads all profiles from directory | Verified — `test_scan_loads_yaml_files`; all 4 bundled profiles load |
| 8 | Evidence chains contain source URL or 'provided context' | Verified — `_extract_source()` implemented; 3 source-extraction tests |
