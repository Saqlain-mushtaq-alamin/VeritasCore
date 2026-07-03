"""Download and validate all required ML models for VeritasCore.

Downloads models to the HuggingFace cache (~/.cache/huggingface/).
Run this once before using VeritasCore for the first time.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --skip-decomposer   # Skip the large LLM

Exit codes:
    0 — All models downloaded successfully
    1 — One or more downloads failed
"""

from __future__ import annotations

import argparse
import sys
import time


def _fmt_params(model: object) -> str:
    """Format parameter count as human-readable string."""
    n = sum(p.numel() for p in model.parameters())  # type: ignore[union-attr]
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    return f"{n/1e6:.0f}M"


def check_gpu() -> bool:
    """Report GPU status and return True if CUDA is available."""
    import torch

    if not torch.cuda.is_available():
        print("⚠  No CUDA GPU detected. Models will run on CPU (slower).")
        return False

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    cuda_ver = torch.version.cuda
    print(f"✓  GPU: {gpu_name} ({vram_gb:.1f} GB VRAM, CUDA {cuda_ver})")
    return True


def download_nli_model() -> None:
    """Download the NLI cross-encoder model (Phase 2: grounded verification)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "cross-encoder/nli-deberta-v3-base"
    print(f"\n[1/3] NLI model: {model_name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Quick smoke test
    inputs = tokenizer(
        "The sky is blue.", "The sky is not blue.",
        return_tensors="pt", truncation=True,
    )
    outputs = model(**inputs)
    assert outputs.logits.shape == (1, 3), f"Unexpected output shape: {outputs.logits.shape}"

    elapsed = time.time() - t0
    print(f"  ✓ Loaded {_fmt_params(model)} params | Labels: {list(model.config.id2label.values())}")
    print(f"  ✓ Smoke test passed | {elapsed:.1f}s")
    del model, tokenizer


def download_embedding_model() -> None:
    """Download the sentence embedding model (Phases 4, 5: consistency + fusion)."""
    from sentence_transformers import SentenceTransformer

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"\n[2/3] Embedding model: {model_name}")
    t0 = time.time()

    model = SentenceTransformer(model_name)

    # Smoke test: encode two sentences, check shape and similarity
    sentences = ["The Eiffel Tower is in Paris.", "Paris is home to the Eiffel Tower."]
    embeddings = model.encode(sentences)
    assert embeddings.shape == (2, 384), f"Unexpected shape: {embeddings.shape}"

    from numpy.linalg import norm
    import numpy as np
    cos_sim = np.dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
    assert cos_sim > 0.8, f"Unexpectedly low similarity for paraphrase: {cos_sim:.3f}"

    elapsed = time.time() - t0
    print(f"  ✓ Embedding dim: {embeddings.shape[1]} | Paraphrase cosine sim: {cos_sim:.3f}")
    print(f"  ✓ Smoke test passed | {elapsed:.1f}s")
    del model


def download_decomposer_model() -> None:
    """Download the claim decomposition LLM (Phase 1: claim extraction)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "microsoft/Phi-3-mini-4k-instruct"
    print(f"\n[3/3] Decomposer model: {model_name}")
    print("  ⚡ This is the largest download (~7GB). Please wait...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Half precision to fit in 8GB VRAM
        device_map="auto",          # Automatically place on GPU/CPU
        trust_remote_code=True,
    )

    # Smoke test: generate a short response
    prompt = "List one fact about Paris:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    assert len(response.strip()) > 0, "Model generated empty response"

    elapsed = time.time() - t0
    print(f"  ✓ {_fmt_params(model)} params | Device: {model.device}")
    print(f"  ✓ Smoke test: '{response.strip()[:60]}...'")
    print(f"  ✓ Download & load complete | {elapsed:.1f}s")
    del model, tokenizer

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VeritasCore ML models")
    parser.add_argument("--skip-decomposer", action="store_true",
                        help="Skip downloading the large decomposer LLM")
    parser.add_argument("--skip-nli", action="store_true", help="Skip NLI model")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip embedding model")
    args = parser.parse_args()

    print("=" * 60)
    print("VeritasCore — Model Download & Validation")
    print("=" * 60)

    check_gpu()

    failed: list[str] = []

    if not args.skip_nli:
        try:
            download_nli_model()
        except Exception as e:
            print(f"  ✗ NLI model failed: {e}")
            failed.append("NLI")

    if not args.skip_embeddings:
        try:
            download_embedding_model()
        except Exception as e:
            print(f"  ✗ Embedding model failed: {e}")
            failed.append("Embeddings")

    if not args.skip_decomposer:
        try:
            download_decomposer_model()
        except Exception as e:
            print(f"  ✗ Decomposer model failed: {e}")
            failed.append("Decomposer")

    print("\n" + "=" * 60)
    if failed:
        print(f"✗ {len(failed)} model(s) failed: {', '.join(failed)}")
        print("  Check your internet connection and disk space, then retry.")
        sys.exit(1)
    else:
        print("✓ All models downloaded and validated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
