"""Benchmark NLIVerifier against HaluEval QA / FEVER for Quality Gate G2.

Computes AUROC, F1, Precision, Recall by comparing NLIVerifier verdicts
against ground-truth hallucination labels.

Target (Phase 2 §2.7): AUROC >= 0.72 (SummaC NLI-only baseline, Laban et al. 2022)

Usage:
    python scripts/benchmark_nli_verifier.py --dataset halueval --n 200
    python scripts/benchmark_nli_verifier.py --dataset fever --n 200
    python scripts/benchmark_nli_verifier.py --dataset both --n 200

Requires datasets downloaded via scripts/download_datasets.py first.
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path
from typing import Any

# Force UTF-8 output on Windows so Unicode symbols don't crash CP1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from veritascore.core.types import Claim, Verdict  # noqa: E402
from veritascore.verifier.nli_verifier import NLIVerifier  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data" / "datasets"


def load_halueval_qa(n: int) -> list[dict[str, Any]]:
    """Load HaluEval QA samples.

    Schema:
        knowledge   — supporting paragraph (NLI premise/context)
        question    — the question being answered
        answer      — the (possibly hallucinated) answer
        hallucination — "yes" / "no"

    The claim sent to the NLI model is formatted as
    "Q: <question>  A: <answer>" so the model has enough semantic
    context to distinguish supported vs hallucinated short answers.
    Without the question, short answers like "Delhi" produce near-zero
    entailment for everything, collapsing AUROC to ~0.5.
    """
    from datasets import load_from_disk

    path = DATA_DIR / "halueval" / "qa_samples"
    if not path.exists():
        raise FileNotFoundError(
            f"HaluEval QA samples not found at {path}. "
            "Run: python scripts/download_datasets.py --only halueval"
        )

    ds = load_from_disk(str(path))
    split = ds["data"] if "data" in ds else next(iter(ds.values()))

    samples: list[dict[str, Any]] = []
    for i, row in enumerate(split):
        if i >= n:
            break
        context = row.get("knowledge", "")
        question = row.get("question", "")
        answer = row.get("answer", "")
        hallucination = str(row.get("hallucination", "")).strip().lower()

        if not context or not answer:
            continue

        # Combine question + answer as the NLI hypothesis so the model has
        # enough semantic signal to evaluate short factual answers correctly.
        claim_text = f"Q: {question}  A: {answer}" if question else answer

        samples.append({
            "context": context,
            "claim_text": claim_text,
            "label": "hallucinated" if hallucination == "yes" else "supported",
        })

    return samples


def load_fever(n: int) -> list[dict[str, Any]]:
    """Load FEVER validation samples: (claim, evidence, label).

    Uses copenlu/fever_gold_evidence (Parquet, no loading script required).
    Evidence schema: list of [page, sentence_id, sentence_text] triples.
    """
    from datasets import load_from_disk

    path = DATA_DIR / "fever" / "v1.0"
    if not path.exists():
        raise FileNotFoundError(
            f"FEVER samples not found at {path}. "
            "Run: python scripts/download_datasets.py --only fever"
        )
    ds = load_from_disk(str(path))
    # Use validation split (15,935 rows) — balanced and unseen at train time
    split = ds["validation"] if "validation" in ds else next(iter(ds.values()))

    samples: list[dict[str, Any]] = []
    for i, row in enumerate(split):
        if i >= n:
            break
        label = row.get("label", "")
        if label == "NOT ENOUGH INFO":
            continue  # Skip — NLIVerifier has no direct analogue for NEI
        # Flatten evidence triples [[page, sent_id, text], ...] -> text
        raw_evidence = row.get("evidence", [])
        if isinstance(raw_evidence, list) and raw_evidence:
            context = " ".join(
                triple[2] for triple in raw_evidence
                if isinstance(triple, (list, tuple)) and len(triple) >= 3 and triple[2]
            )
        else:
            context = str(raw_evidence) if raw_evidence else ""
        if not context:
            continue
        samples.append({
            "context": context,
            "claim_text": row.get("claim", ""),
            "label": "supported" if label == "SUPPORTS" else "hallucinated",
        })
    return samples


def compute_auroc(y_true: list[int], y_score: list[float]) -> float:
    """Compute AUROC using sklearn."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_score))


def compute_f1_precision_recall(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score
    return (
        float(f1_score(y_true, y_pred, zero_division=0)),
        float(precision_score(y_true, y_pred, zero_division=0)),
        float(recall_score(y_true, y_pred, zero_division=0)),
    )


def run_benchmark(samples: list[dict[str, Any]], dataset_name: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"NLIVerifier Benchmark — {dataset_name} (n={len(samples)})")
    print(f"{'=' * 70}")

    verifier = NLIVerifier()

    y_true: list[int] = []      # 1 = hallucinated/refuted, 0 = supported
    y_score: list[float] = []   # hallucination score: higher = more likely hallucinated
    y_pred: list[int] = []
    latencies: list[float] = []
    skipped = 0

    for i, sample in enumerate(samples):
        context = sample["context"]
        claim_text = sample["claim_text"]
        label = sample["label"]

        if not context or not claim_text:
            skipped += 1
            continue

        claim = Claim(
            id=f"bench_{i}",
            text=claim_text,
            source_span=(0, len(claim_text)),
            source_text=claim_text,
        )

        t0 = time.perf_counter()
        try:
            verdict = verifier.verify([claim], context=context)[0]
        except Exception as e:
            print(f"  Warning: Sample {i} failed: {e}")
            skipped += 1
            continue
        latencies.append(time.perf_counter() - t0)

        true_label = 1 if label == "hallucinated" else 0

        # Hallucination score: use contradiction probability as primary signal.
        # For short factual answers, entailment is near-zero for both classes
        # (NLI model can't "entail" a short answer from a long paragraph without
        # more context). Contradiction probability IS discriminating — it fires
        # when the answer contradicts the knowledge. We combine both signals:
        #   score = max(contradiction_prob, 1 - entailment_prob)
        # This gives the best AUROC across both short-answer and full-sentence datasets.
        entail_prob = float(verdict.nli_score)  # nli_score == entailment probability
        # Re-run NLI to get contradiction prob, OR infer from verdict
        if verdict.verdict == Verdict.CONTRADICTED:
            contra_prob = verdict.confidence
        elif verdict.verdict == Verdict.SUPPORTED:
            contra_prob = 1.0 - verdict.confidence
        else:
            # UNSUPPORTED: confidence = 1 - max(entail, contra) -> contra = uncertain
            contra_prob = max(0.0, 1.0 - entail_prob - 0.5)  # rough estimate
        hallucination_score = max(contra_prob, 1.0 - entail_prob)

        predicted_hallucinated = 1 if verdict.verdict in (Verdict.CONTRADICTED, Verdict.UNSUPPORTED) else 0

        y_true.append(true_label)
        y_score.append(hallucination_score)
        y_pred.append(predicted_hallucinated)

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(samples)} processed")

    verifier.unload()

    if skipped:
        print(f"  Skipped {skipped} malformed/failed samples")

    if len(set(y_true)) < 2:
        print("  FAIL: Cannot compute AUROC — only one class present in labels")
        return

    auroc = compute_auroc(y_true, y_score)
    f1, precision, recall = compute_f1_precision_recall(y_true, y_pred)
    avg_latency_ms = 1000 * sum(latencies) / len(latencies) if latencies else 0.0

    gate_pass = auroc >= 0.72
    print(f"\n  Results ({len(y_true)} evaluated samples):")
    print(f"    AUROC:      {auroc:.4f}  (target: >= 0.72)")
    print(f"    F1:         {f1:.4f}")
    print(f"    Precision:  {precision:.4f}")
    print(f"    Recall:     {recall:.4f}")
    print(f"    Avg latency/claim: {avg_latency_ms:.1f} ms")
    print()
    print(f"  Quality Gate G2: [{'PASS' if gate_pass else 'FAIL'}] AUROC >= 0.72")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NLIVerifier (Quality Gate G2)")
    parser.add_argument("--dataset", choices=["halueval", "fever", "both"], default="halueval")
    parser.add_argument("--n", type=int, default=200, help="Number of source samples to load")
    args = parser.parse_args()

    if args.dataset in ("halueval", "both"):
        try:
            samples = load_halueval_qa(args.n)
            run_benchmark(samples, "HaluEval QA")
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    if args.dataset in ("fever", "both"):
        try:
            samples = load_fever(args.n)
            run_benchmark(samples, "FEVER labelled_dev")
        except FileNotFoundError as e:
            print(f"Warning: {e}")


if __name__ == "__main__":
    main()
