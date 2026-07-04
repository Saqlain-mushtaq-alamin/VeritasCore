# Phase 3 — Quality Gate G3 Evaluation Log

> This sandbox has no GPU, no Tavily/Brave API keys, and no live network access to
> search providers. The F1 benchmark against HaluEval requires live web search, so
> it could not be executed here. This log records what WAS validated (all retriever
> logic, caching, rate limiting, retries, and evidence aggregation, all with mocked
> HTTP/NLI) and what MUST be re-run with real credentials before sign-off.

---

## 1. What Was Validated in This Sandbox

| Check | Method | Result |
|---|---|---|
| `BaseRetriever` abstract contract | pytest | Pass |
| `SearchResult` validation (relevance bounds) | pytest | Pass |
| `SearchCache` (set/get, case-insensitivity, atomic writes, corruption handling, clear) | tmp_path-based, no network | Pass (9 tests) |
| `AsyncRateLimiter` (token bucket, blocks when over limit) | Real asyncio, timed assertions | Pass (4 tests) |
| `retry_with_backoff` (success, retry-then-succeed, exhaustion, non-retryable passthrough) | Real asyncio, no network | Pass (5 tests) |
| `TavilyRetriever` (auth, caching, HTTP errors, retry-on-503) | Mocked `httpx.AsyncClient.post` | Pass (8 tests) |
| `BraveRetriever` (auth, caching, response parsing) | Mocked `httpx.AsyncClient.get` | Pass (4 tests) |
| `OfflineRetriever` (cache-only behavior, never raises) | tmp_path-based | Pass (4 tests) |
| Retriever auto-selection (Tavily/Brave/Offline based on configured keys, both provider orderings) | Config-driven, dummy NLI | Pass (5 tests) |
| `verify()` event-loop guard + retriever-exception handling | pytest-asyncio | Pass (2 tests) |
| Query formulation (prefix stripping, length cap) | Pure function | Pass (5 tests) |
| Evidence aggregation (majority support/contradict, no-agreement, no-results, relevance weighting) | Stub NLI with programmed probs | Pass (12 tests) |
| Evidence/score consistency (same bug class as Phase 2's chunk-consistency fix) | Stub NLI, multi-source scenario | Pass |
| End-to-end decompose -> retrieve -> verify pipeline | `RuleDecomposer` + `OfflineRetriever` (seeded cache) + stub NLI | Pass (manual smoke test, see below) |
| Lint (ruff) | `ruff check` | Clean |
| Type check (mypy) | `mypy --ignore-missing-imports` | Clean (2 real type errors found and fixed — see §3) |
| Unit test coverage | `pytest --cov` | retriever/ 91-100%, retrieval_verifier.py 99% |

**Total: 64 new unit tests (45 in test_retrievers.py + 19 in test_retrieval_verifier.py), 200/200 project-wide unit tests pass.**

### Manual end-to-end smoke test (seeded OfflineRetriever, stub NLI, real RuleDecomposer)

```
response = "The Eiffel Tower stands 330 meters tall. It was completed in 1700."

[supported   ] 'The Eiffel Tower stands 330 meters tall.'
    retrieval_score=0.77 nli_score=0.85
    evidence='The Eiffel Tower is 330 metres tall in Paris, France. [Source: https://en.wikipedia.org/wiki/Eiffel_Tower]'

[contradicted] 'It was completed in 1700.'
    retrieval_score=0.09 nli_score=0.10
    evidence='The tower was completed in 1889 for the World Fair. [Source: https://en.wikipedia.org/wiki/Eiffel_Tower]'
```

The pipeline correctly distinguishes a true claim from a false one, with evidence
and source URL correctly attached to each verdict.

---

## 2. Deviations From the Reference Spec (Bugs Found and Fixed)

The reference implementation in the Phase 3 plan was missing two of its own listed
deliverables and contradicted its own developer notes:

1. **No rate limiting was implemented**, despite G3 criterion 6 ("Rate limiting
   prevents API quota exhaustion in benchmarks") and "Rate limiting and retry logic"
   being listed as a deliverable. Added `retriever/rate_limit.py` with
   `AsyncRateLimiter` (token bucket, 5 calls/sec default) wired into both
   `TavilyRetriever` and `BraveRetriever`.

2. **No retry logic was implemented**, despite being a listed deliverable. Added
   `retry_with_backoff()` (exponential backoff + jitter) wired into both retrievers'
   HTTP calls, retrying on `httpx.TransportError` and `httpx.HTTPStatusError`.

3. **Sequential `for claim in claims: ... await self.retriever.search(...)` inside
   `_verify_async`** directly contradicted the spec's own developer note 3.9
   ("Async is important... gather multiple searches concurrently"). Replaced with
   `asyncio.gather()` over all claims, bounded by an `asyncio.Semaphore` (default
   `max_concurrent_claims=8`) to avoid unbounded concurrent connections.

4. **Evidence/score inconsistency risk** — the same bug class fixed in Phase 2's
   `NLIVerifier`. The reference selected `best_idx` by `max(entailment, contradiction)`
   across sources (correct), but used a loosely-typed `dict[str, object]` for the
   per-source result, which mypy flagged as unsafe. Replaced with an explicit
   `_EvidenceNLIResult` NamedTuple so verdict, `nli_score`, and `evidence` are always
   read from the exact same source object, with a dedicated regression test
   (`test_evidence_and_score_consistency`) constructing a weak-then-strong-evidence
   scenario where a naive implementation could diverge.

5. **`verify()` silently called `asyncio.new_event_loop()` and `loop.run_until_complete()`
   unconditionally** in the spec, which raises a `DeprecationWarning` in modern Python
   and crashes outright if called from code that's already inside a running event loop
   (e.g. a FastAPI async handler in Phase 7). Replaced with `asyncio.run()` plus an
   explicit guard that raises a clear `RuntimeError` directing the caller to use
   `await verify_async(...)` instead, when already inside a running loop. Covered by
   `test_verify_raises_inside_running_loop`.

6. **mypy caught two real type errors** during development: a `dict[str, str | None]`
   passed where `httpx` expects `Mapping[str, str]` (the Brave API key could be `None`
   per the type checker even though `is_available()` was already checked — fixed with
   an explicit local type-narrowed variable), and `float(object)` calls on the loosely
   typed dict values from the original spec's aggregation dict (fixed by the
   `_EvidenceNLIResult` NamedTuple in point 4).

---

## 3. Pending Hardware/Credentials Validation (Required Before G3 Sign-off)

Run with a real Tavily or Brave API key configured (`.env`) and the NLI model +
HaluEval dataset downloaded:

```bash
# Live search integration tests (auto-skip if no key configured)
TAVILY_API_KEY=... pytest tests/integration/test_ungrounded_pipeline.py -v -m integration

# F1 benchmark — the actual G3 acceptance criterion
python scripts/benchmark_retrieval_verifier.py --dataset halueval --n 50
```

**Target**: F1 >= your chosen retrieval-only baseline (the spec does not name a
specific baseline paper/score for this criterion — pick one, e.g. a BM25-retrieval
+ lexical-overlap baseline, or the retrieval-only ablation referenced in the Phase 2
literature review, and record the result in this file).

### Specific things to confirm with real credentials

1. **Live Tavily/Brave response shape**: the mocked tests assume Tavily's
   `results[].{title,url,content,score}` and Brave's `web.results[].{title,url,description}`
   schemas based on the spec's documented API; confirm these match the current
   live API response shape (search APIs occasionally change field names).
2. **Cache effectiveness in a real benchmark run**: re-run
   `benchmark_retrieval_verifier.py` twice with the same `--n` and confirm the
   second run completes near-instantly (all cache hits, per
   `test_second_identical_search_uses_cache`'s unit-level guarantee, now validated
   at benchmark scale).
3. **Rate limiter tuning**: the default 5 calls/sec is conservative; if Tavily/Brave
   allow higher throughput, tune `AsyncRateLimiter(max_calls=..., period_seconds=...)`
   per-provider and re-benchmark latency.
4. **Concurrent claim verification**: `max_concurrent_claims=8` (default) bounds
   simultaneous in-flight searches; confirm this doesn't trip provider-side
   concurrent-connection limits during a real benchmark run.

---

## 4. Acceptance Criteria Status (Quality Gate G3)

| # | Criterion | Status |
|---|---|---|
| 1 | All retriever implementations conform to `BaseRetriever` interface | Verified in sandbox |
| 2 | Search caching prevents duplicate API calls (test with counter) | Verified — `test_second_identical_search_uses_cache` asserts call_count == 1 across 3 identical searches |
| 3 | `RetrievalVerifier` produces valid `ClaimVerdict` for all test cases | Verified (19 evidence-aggregation/init tests) |
| 4 | F1 >= baseline retrieval-only method on HaluEval subset | Needs live API key + hardware run |
| 5 | Offline mode returns graceful "unsupported" verdicts (no crashes) | Verified — `OfflineRetriever` never raises; `test_retriever_exception_yields_unsupported_not_crash` confirms graceful handling even when the retriever itself throws |
| 6 | Rate limiting prevents API quota exhaustion in benchmarks | Implemented (was missing from the reference spec) — `AsyncRateLimiter`, unit-tested; real-world quota behavior needs a live benchmark run |
| 7 | All verdicts include evidence snippet and source URL when available | Verified — `test_evidence_includes_source_url` |
| 8 | Auto-selection of retriever works based on available API keys | Verified — 5 tests covering both provider orderings and the offline fallback |
