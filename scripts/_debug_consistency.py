"""Debug consistency checker import and basic operation."""
import io, sys, traceback
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "src")

print("Testing import...")
try:
    from veritascore.verifier.consistency import SemanticConsistencyChecker
    print("Import OK")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)

print("Creating checker...")
try:
    checker = SemanticConsistencyChecker()
    print("Created OK")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)

print("Loading model...")
try:
    import time
    t0 = time.time()
    checker._load_model()
    print(f"Model loaded in {time.time()-t0:.2f}s")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)

print("Testing score_claim...")
try:
    from veritascore.core.types import Claim
    c = Claim(id="c1", text="The Eiffel Tower is in Paris.", source_span=(0, 30), source_text="The Eiffel Tower is in Paris.")
    score = checker.score_claim(c, "Where is the Eiffel Tower?")
    print(f"score: {score}")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)

checker.unload()
print("ALL OK")
