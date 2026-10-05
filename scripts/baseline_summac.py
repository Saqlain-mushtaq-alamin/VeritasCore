"""SummaC ZS baseline — reimplemented using DeBERTa-v3 NLI (Phase R4).

The original `summac` package (v0.0.4) pins huggingface_hub==0.17.0, which is
incompatible with the modern `datasets` library (>=0.25.0 required). Installing
it breaks the entire pipeline. We therefore reimplement the SummaC ZS algorithm
directly using the NLI cross-encoder that VeritasCore already has installed.

Algorithm (SummaC ZS — Laban et al. 2022, §3.1):
    For each (document D, hypothesis h):
      1. Split D into sentences s_1, ..., s_k using NLTK punkt tokenizer.
      2. For each sentence s_i, compute P(entailment | s_i, h) with an NLI model.
      3. SummaC consistency score = max_i P(entailment | s_i, h).
      4. This score is in [0, 1] — HIGH means claim is consistent with document.
    Hallucination score = 1.0 - consistency_score  (invert for our convention).

Model:
    We use `cross-encoder/nli-deberta-v3-base` (already installed by VeritasCore)
    instead of Laban et al.'s `vitc` model.  This is noted clearly in results:
    "SummaC ZS (reimplemented, DeBERTa-v3-base)" with a paper footnote.
    The algorithmic comparison (sentence-level max vs. bidirectional scoring) is
    still valid because both use the same model backbone.

Reference:
    Laban et al. (2022). SummaC: Re-visiting NLI-based Models for
    Inconsistency Detection in Summarization. TACL.
    https://arxiv.org/abs/2111.09525

Usage:
    python scripts/baseline_summac.py --dataset both --n 1000 --bootstrap-ci
    python scripts/baseline_summac.py --dataset halueval --n 200
    python scripts/baseline_summac.py --dataset fever  --n 200 \\
        --bootstrap-ci --output tests/benchmarks/results/
"""
# ruff: noqa: E501

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPTS_DIR = Path(__file__).parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))           # for stats_utils
sys.path.insert(0, str(REPO_ROOT / "src"))     # for veritascore

from stats_utils import bootstrap_auroc_ci, bootstrap_f1_ci, format_ci  # noqa: E402

DATA_DIR = REPO_ROOT / "data" / "datasets"

# NLI model to use — same as VeritasCore's NLI verifier for fair hardware comparison
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

# Module-level stubs — populated lazily on first SummaCZSReimplemented init.
# Defined here so unit tests can patch them with patch('baseline_summac.AutoTokenizer') etc.
AutoTokenizer = None  # type: ignore[assignment]
AutoModelForSequenceClassification = None  # type: ignore[assignment]
torch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Sentence splitter (SummaC ZS step 1)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Uses NLTK punkt tokenizer if available; falls back to a simple
    regex-based splitter. SummaC ZS requires sentence-level splitting.
    """
    try:
        import nltk  # type: ignore[import-untyped]
        try:
            return nltk.sent_tokenize(text)
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            return nltk.sent_tokenize(text)
    except ImportError:
        pass

    # Fallback: split on '. ', '! ', '? ' boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s.strip()]


# ---------------------------------------------------------------------------
# SummaC ZS scorer
# ---------------------------------------------------------------------------

class SummaCZSReimplemented:
    """SummaC ZS algorithm reimplemented with DeBERTa-v3 cross-encoder.

    Implements the sentence-level max-entailment scoring from Laban et al. 2022
    §3.1 without requiring the `summac` package.

    Args:
        model_name: HuggingFace model for NLI cross-encoder.
        device:     'cuda' (GPU) or 'cpu'. Auto-detected if None.
        batch_size: Cross-encoder pairs per forward pass.
    """

    def __init__(
        self,
        model_name: str = NLI_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        # Lazy imports — populate the module-level stubs so tests can patch them.
        import torch as _torch
        from transformers import (  # type: ignore[import-untyped]  # noqa: PLC0415
            AutoModelForSequenceClassification as _AMSC,
            AutoTokenizer as _AT,
        )
        import baseline_summac as _m
        _m.torch = _torch
        _m.AutoTokenizer = _AT
        _m.AutoModelForSequenceClassification = _AMSC

        _torch_mod = _torch
        _at_mod = _AT
        _amsc_mod = _AMSC

        if device is None:
            device = "cuda" if _torch_mod.cuda.is_available() else "cpu"

        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name

        print(f"  Loading NLI model {model_name!r} on {device}...")
        self.tokenizer = _at_mod.from_pretrained(model_name)
        self.model = _amsc_mod.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        # DeBERTa NLI label order: 0=contradiction, 1=neutral, 2=entailment
        # (confirmed from cross-encoder/nli-deberta-v3-base config)
        self._entailment_idx = 2

        print(f"  SummaC ZS model ready on {device}.")

    def _nli_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Run NLI on a batch of (premise, hypothesis) pairs.

        Returns P(entailment) for each pair.
        """
        import torch

        entailment_probs: list[float] = []

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            premises, hypotheses = zip(*batch)

            enc = self.tokenizer(
                list(premises),
                list(hypotheses),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**enc).logits  # (B, 3)
                probs = torch.softmax(logits, dim=-1)[:, self.entailment_idx]

            entailment_probs.extend(probs.cpu().tolist())

        return entailment_probs

    @property
    def entailment_idx(self) -> int:
        return self._entailment_idx

    def score(self, document: str, hypothesis: str) -> float:
        """Compute SummaC ZS consistency score for (document, hypothesis).

        Algorithm (Laban et al. 2022 §3.1):
          - Split document into sentences.
          - For each sentence, compute P(entailment | sentence, hypothesis).
          - Return max over all sentences.

        Returns:
            Consistency score in [0, 1].
            HIGH = claim is consistent with document (NOT hallucinated).
            Invert (1 - score) to get hallucination score.
        """
        sentences = _split_sentences(document)
        if not sentences:
            return 0.5  # no sentences → uncertain

        pairs = [(sent, hypothesis) for sent in sentences]
        entailment_probs = self._nli_batch(pairs)

        # SummaC ZS: max entailment probability across all document sentences
        return float(max(entailment_probs))

    def unload(self) -> None:
        """Release GPU memory."""
        import torch
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Dataset loaders (mirrors benchmark_nli_verifier.py exactly)
# ---------------------------------------------------------------------------

def load_halueval_qa(n: int) -> list[dict[str, Any]]:
    """Load exactly n HaluEval QA samples with oracle knowledge paragraphs."""
    from datasets import load_from_disk  # type: ignore[import-untyped]

    path = DATA_DIR / "halueval" / "qa_samples"
    if not path.exists():
        raise FileNotFoundError(
            f"HaluEval QA not found at {path}. "
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
        claim_text = f"Q: {question}  A: {answer}" if question else answer
        samples.append({
            "context": context,
            "claim_text": claim_text,
            "label": "hallucinated" if hallucination == "yes" else "supported",
        })

    return samples


def load_fever(n: int) -> list[dict[str, Any]]:
    """Load exactly n non-NEI FEVER validation samples with gold evidence."""
    from datasets import load_from_disk  # type: ignore[import-untyped]

    path = DATA_DIR / "fever" / "v1.0"
    if not path.exists():
        raise FileNotFoundError(
            f"FEVER not found at {path}. "
            "Run: python scripts/download_datasets.py --only fever"
        )

    ds = load_from_disk(str(path))
    raw_split = ds["validation"] if "validation" in ds else next(iter(ds.values()))

    samples: list[dict[str, Any]] = []
    skipped_nei = 0
    skipped_no_evidence = 0

    for row in raw_split:
        if len(samples) >= n:
            break
        label = row.get("label", "")
        if label == "NOT ENOUGH INFO":
            skipped_nei += 1
            continue
        raw_evidence = row.get("evidence", [])
        if isinstance(raw_evidence, list) and raw_evidence:
            context = " ".join(
                triple[2] for triple in raw_evidence
                if isinstance(triple, (list, tuple)) and len(triple) >= 3 and triple[2]
            )
        else:
            context = str(raw_evidence) if raw_evidence else ""
        if not context:
            skipped_no_evidence += 1
            continue
        samples.append({
            "context": context,
            "claim_text": row.get("claim", ""),
            "label": "supported" if label == "SUPPORTS" else "hallucinated",
        })

    print(
        f"  [FEVER] Collected {len(samples)} non-NEI samples "
        f"(skipped NEI={skipped_nei}, no-evidence={skipped_no_evidence})"
    )
    return samples


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def build_summac_scores(
    samples: list[dict[str, Any]],
    scorer: SummaCZSReimplemented,
) -> tuple[list[int], list[float], list[float]]:
    """Run SummaC ZS on all samples.

    Returns:
        (y_true, hallucination_scores, raw_consistency_scores)

    Score inversion:
        SummaC consistency score is HIGH for consistent/supported claims.
        We invert: hallucination_score = 1.0 - consistency_score.
    """
    y_true: list[int] = []
    y_score: list[float] = []
    raw_scores: list[float] = []

    for i, sample in enumerate(samples):
        context = sample.get("context", "")
        claim = sample.get("claim_text", "")
        label = sample.get("label", "")

        if not context or not claim:
            continue

        consistency = scorer.score(context, claim)
        hallucination_score = 1.0 - consistency

        y_true.append(1 if label == "hallucinated" else 0)
        y_score.append(hallucination_score)
        raw_scores.append(consistency)

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(samples)} processed")

    return y_true, y_score, raw_scores


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_summac_benchmark(
    samples: list[dict[str, Any]],
    dataset_name: str,
    scorer: SummaCZSReimplemented,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 2000,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run SummaC ZS and return structured results."""
    import numpy as np
    from sklearn.metrics import (  # type: ignore[import-untyped]
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    print(f"\n{'=' * 70}")
    print(f"SummaC ZS (reimplemented) — {dataset_name} (n={len(samples)})")
    print(f"Protocol: Oracle Evidence (same gold passages as VeritasCore NLI)")
    print(f"Model:    {scorer.model_name}  |  Device: {scorer.device}")
    print(f"{'=' * 70}")

    t_start = time.perf_counter()
    y_true, y_score, raw_scores = build_summac_scores(samples, scorer)
    elapsed = time.perf_counter() - t_start

    if len(set(y_true)) < 2:
        print("  FAIL: Only one class present — cannot compute AUROC")
        return {
            "mode": "summac_zs_reimplemented",
            "dataset": dataset_name,
            "n_samples": len(y_true),
            "auroc": None,
            "error": "single_class",
        }

    auroc = float(roc_auc_score(y_true, y_score))
    y_pred = [1 if s >= 0.5 else 0 for s in y_score]
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    avg_latency_ms = (elapsed / len(y_true)) * 1000 if y_true else 0.0

    print(f"\n  Results ({len(y_true)} samples, {elapsed:.1f}s total):")
    print(f"    AUROC:     {auroc:.4f}")
    print(f"    F1:        {f1:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall:    {recall:.4f}")
    print(f"    Avg latency/claim: {avg_latency_ms:.1f} ms")

    result: dict[str, Any] = {
        "mode": "summac_zs_reimplemented",
        "protocol": "oracle_evidence",
        "dataset": dataset_name,
        "nli_model": scorer.model_name,
        "device": scorer.device,
        "algorithm": "SummaC ZS: max sentence-level P(entailment)",
        "n_samples": len(y_true),
        "auroc": auroc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "avg_latency_ms": avg_latency_ms,
        "total_elapsed_s": elapsed,
        "score_formula": "1.0 - max(P(entailment | sentence_i, claim))",
        "threshold": 0.5,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "_y_true": [int(v) for v in y_true],
        "_y_score": [float(v) for v in y_score],
        "_y_pred": [int(v) for v in y_pred],
    }

    if bootstrap_ci and len(y_true) >= 10:
        print(f"\n  Computing bootstrapped CIs (n_bootstrap={n_bootstrap})...")
        auc_mean, auc_lo, auc_hi = bootstrap_auroc_ci(y_true, y_score, n_bootstrap, seed=seed)
        f1_mean, f1_lo, f1_hi = bootstrap_f1_ci(y_true, y_pred, n_bootstrap, seed=seed)
        result["ci_auroc_mean"] = auc_mean
        result["ci_auroc_lower"] = auc_lo
        result["ci_auroc_upper"] = auc_hi
        result["ci_f1_mean"] = f1_mean
        result["ci_f1_lower"] = f1_lo
        result["ci_f1_upper"] = f1_hi
        result["ci_n_bootstrap"] = n_bootstrap
        result["ci_level"] = 0.95
        print(f"    AUROC 95% CI: {format_ci(auc_mean, auc_lo, auc_hi)}")
        print(f"    F1    95% CI: {format_ci(f1_mean, f1_lo, f1_hi)}")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = dataset_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_result = {k: v for k, v in result.items() if not k.startswith("_")}
        out_path = output_dir / f"{safe_name}_summac_oracle_results.json"
        out_path.write_text(json.dumps(save_result, indent=2))
        print(f"\n  Results saved → {out_path}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run SummaC ZS baseline under oracle-evidence conditions (Phase R4).\n\n"
            "Reimplements SummaC ZS using DeBERTa-v3-base (already installed).\n"
            "GPU (RTX 4060) strongly recommended — ~280ms/claim.\n\n"
            "Produces Table 1 data: NLI Quality comparison (apples-to-apples)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["halueval", "fever", "both"],
        default="both",
    )
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-ci", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="NLI cross-encoder batch size (default: 32, reduce if OOM)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Compute device (default: auto-detect CUDA)",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Phase R4: SummaC ZS Oracle Baseline (Reimplemented)")
    print("  Algorithm: sentence-level max entailment (Laban et al. 2022)")
    print(f"  Model:     {NLI_MODEL_NAME}")
    print("  Protocol:  Oracle Evidence  →  Table 1: NLI Quality")
    print("=" * 70)

    # Load scorer once, share across datasets
    scorer = SummaCZSReimplemented(
        model_name=NLI_MODEL_NAME,
        device=args.device,
        batch_size=args.batch_size,
    )

    results = []

    if args.dataset in ("halueval", "both"):
        try:
            print(f"\n[HaluEval] Loading {args.n} samples...")
            samples = load_halueval_qa(args.n)
            r = run_summac_benchmark(
                samples, "HaluEval QA", scorer,
                bootstrap_ci=args.bootstrap_ci,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                output_dir=args.output,
            )
            results.append(r)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    if args.dataset in ("fever", "both"):
        try:
            print(f"\n[FEVER] Loading {args.n} non-NEI samples...")
            samples = load_fever(args.n)
            r = run_summac_benchmark(
                samples, "FEVER (validation)", scorer,
                bootstrap_ci=args.bootstrap_ci,
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
                output_dir=args.output,
            )
            results.append(r)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")

    scorer.unload()

    # Summary table
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY — SummaC ZS (reimplemented, DeBERTa-v3-base)")
        print(f"{'Dataset':<30} {'AUROC':>10} {'F1':>10} {'Avg ms/claim':>15}")
        print("-" * 70)
        for r in results:
            if r.get("auroc") is None:
                continue
            auroc_str = f"{r['auroc']:.4f}"
            if "ci_auroc_mean" in r:
                auroc_str = (
                    f"{r['ci_auroc_mean']:.4f} "
                    f"({r['ci_auroc_lower']:.4f}–{r['ci_auroc_upper']:.4f})"
                )
            print(f"  {r['dataset']:<28} {auroc_str:>10} {r['f1']:>10.4f} {r['avg_latency_ms']:>15.1f}")
        print("=" * 70)

    print("\n✓ Phase R4 SummaC oracle baseline complete.")
    print("  Results feed into Table 1 (NLI Quality, Oracle Evidence) in the paper.")
    print("  Paper note: 'SummaC ZS reimplemented with DeBERTa-v3-base;")
    print("  original vitc model incompatible with current dependency stack.'\n")


if __name__ == "__main__":
    main()
