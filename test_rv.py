import time
print("1. Importing modules...")
from veritascore.core.types import Claim
from veritascore.retriever.base import SearchResult
from veritascore.retriever.cache import SearchCache
from veritascore.verifier.retrieval_verifier import RetrievalVerifier

print("2. Instantiating RetrievalVerifier...")
v = RetrievalVerifier()
print("3. Pre-populating cache...")
cache = SearchCache(cache_dir=".cache/search")
cache.set(
    "The Eiffel Tower is located in Paris.",
    [
        SearchResult(
            title="Eiffel Tower - Wikipedia",
            url="https://en.wikipedia.org/wiki/Eiffel_Tower",
            snippet="The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
            relevance_score=0.95,
        )
    ],
)
v.retriever.cache = cache
claim = Claim(
    id="ret_bench",
    text="The Eiffel Tower is located in Paris.",
    source_span=(0, 38),
    source_text="The Eiffel Tower is located in Paris.",
)

# Test synchronous path:
v.nli_verifier._load_model()
q = v._formulate_query(claim, None)
res = v.retriever.cache.get(q)
t0 = time.perf_counter()
ans = v._verify_against_evidence(claim, res)
dt = (time.perf_counter() - t0) * 1000
print(f"Sync verify done in {dt:.1f}ms: verdict={ans.verdict}, conf={ans.confidence:.2f}")
