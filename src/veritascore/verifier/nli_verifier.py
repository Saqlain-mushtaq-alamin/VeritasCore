"""NLI-based grounded claim verification using a cross-encoder.

Verifies atomic claims against a provided source context using Natural
Language Inference (NLI). This is the highest-confidence verification mode
in VeritasCore because it checks claims against ground-truth evidence
supplied by the caller, rather than retrieved or self-consistency signals.

Model: cross-encoder/nli-deberta-v3-base (default, ~180M params, ~700MB fp32)
Output: [contradiction, neutral, entailment] probabilities per (premise, hypothesis) pair
    premise = context chunk
    hypothesis = claim text

Memory note:
    Cannot fit simultaneously with the Phase 1 decomposer LLM on an 8GB GPU.
    Call LLMDecomposer.unload() before constructing/loading NLIVerifier.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from veritascore.core.config import EngineConfig
from veritascore.core.exceptions import ModelLoadError, VerificationError
from veritascore.core.types import Claim, ClaimVerdict, Verdict, VerificationMode
from veritascore.verifier.base import BaseVerifier
from veritascore.verifier.utils import chunk_context, extract_evidence_snippet

logger = logging.getLogger(__name__)

# cross-encoder/nli-deberta-v3-base label order — verified against the
# model's config.id2label at load time (see _load_model). This constant
# documents the EXPECTED order; _label_order holds the ACTUAL order used.
EXPECTED_NLI_LABEL_ORDER = ("contradiction", "neutral", "entailment")

_DEFAULT_MAX_CONTEXT_TOKENS = 512
_DEFAULT_CHUNK_OVERLAP_TOKENS = 50
_RESERVED_TOKENS_FOR_HYPOTHESIS = 50


class NLIVerifier(BaseVerifier):
    """Verify claims against provided context using an NLI cross-encoder.

    For each claim, the verifier checks every context chunk and selects the
    chunk that gives the strongest signal (either entailment or
    contradiction, whichever is larger). The verdict is then derived from
    that single chunk's full probability vector, ensuring the reported
    nli_score, verdict, and evidence are always mutually consistent.

    Args:
        config: EngineConfig instance. Defaults to EngineConfig.default().
        entailment_threshold: Minimum entailment probability for SUPPORTED.
        contradiction_threshold: Minimum contradiction probability for CONTRADICTED.
        max_context_tokens: Max total sequence length (premise + hypothesis).
        chunk_overlap_tokens: Overlap between consecutive context chunks.

    Example:
        >>> verifier = NLIVerifier()
        >>> verdicts = verifier.verify(
        ...     claims=[Claim(text="The tower is 330m tall.", ...)],
        ...     context="The Eiffel Tower stands 330 metres tall in Paris.",
        ... )
        >>> verdicts[0].verdict
        <Verdict.SUPPORTED: 'supported'>
        >>> verifier.unload()
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        entailment_threshold: float = 0.7,
        contradiction_threshold: float = 0.5,
        max_context_tokens: int = _DEFAULT_MAX_CONTEXT_TOKENS,
        chunk_overlap_tokens: int = _DEFAULT_CHUNK_OVERLAP_TOKENS,
    ) -> None:
        if not 0.0 <= entailment_threshold <= 1.0:
            raise ValueError("entailment_threshold must be in [0, 1]")
        if not 0.0 <= contradiction_threshold <= 1.0:
            raise ValueError("contradiction_threshold must be in [0, 1]")

        self.config = config or EngineConfig.default()
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.max_context_tokens = max_context_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"
        self._loaded: bool = False

        # Index of [contradiction, neutral, entailment] within the model's
        # raw logits — discovered from the model config at load time, since
        # different NLI checkpoints order labels differently.
        self._label_index: dict[str, int] = {"contradiction": 0, "neutral": 1, "entailment": 2}

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        import torch

        device = self.config.models.device
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _discover_label_order(self) -> None:
        """Inspect model.config.id2label to determine actual label→index mapping.

        Falls back to EXPECTED_NLI_LABEL_ORDER if the model's labels don't
        match the expected vocabulary (e.g. a differently-trained checkpoint).
        """
        id2label = getattr(self._model.config, "id2label", None)
        if not id2label:
            logger.warning(
                "Model has no id2label config; assuming default order %s",
                EXPECTED_NLI_LABEL_ORDER,
            )
            return

        discovered: dict[str, int] = {}
        for idx, label in id2label.items():
            normalized = str(label).lower()
            for expected in EXPECTED_NLI_LABEL_ORDER:
                if expected in normalized:
                    discovered[expected] = int(idx)

        if len(discovered) == 3:
            self._label_index = discovered
            logger.info("NLI label order discovered from model config: %s", discovered)
        else:
            logger.warning(
                "Could not fully resolve NLI label order from id2label=%s; using default %s",
                id2label,
                EXPECTED_NLI_LABEL_ORDER,
            )

    def _load_model(self) -> None:
        """Lazy-load the NLI model (idempotent)."""
        if self._loaded:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_name = self.config.models.nli_model
        logger.info("Loading NLI model: %s", model_name)
        t0 = time.time()

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)

            self._device = self._resolve_device()
            self._model = self._model.to(self._device)
            self._model.eval()

            self._discover_label_order()

            elapsed = time.time() - t0
            logger.info("NLI model loaded on %s in %.1fs", self._device, elapsed)
        except Exception as e:
            raise ModelLoadError(f"Failed to load NLI model '{model_name}': {e}") from e

        self._loaded = True

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_nli(self, premise: str, hypothesis: str) -> Any:
        """Run NLI inference for a single (premise, hypothesis) pair.

        Returns:
            numpy array of shape (3,): [contradiction, neutral, entailment]
            probabilities, reordered according to the discovered label index.
        """
        import numpy as np
        import torch

        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            max_length=self.max_context_tokens,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            raw_probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        return self._reorder_probs(raw_probs, np)

    def _reorder_probs(self, raw_probs: Any, np_module: Any) -> Any:
        """Reorder raw model probs into [contradiction, neutral, entailment]."""
        return np_module.array(
            [
                raw_probs[self._label_index["contradiction"]],
                raw_probs[self._label_index["neutral"]],
                raw_probs[self._label_index["entailment"]],
            ]
        )

    # ── Verdict Logic ─────────────────────────────────────────────────────────

    def _probs_to_verdict(self, probs: Any) -> tuple[Verdict, float, float, str]:
        """Convert a [contradiction, neutral, entailment] vector to a verdict.

        Args:
            probs: Array-like [contradiction, neutral, entailment].

        Returns:
            Tuple of (verdict, confidence, entailment_prob, reason).
        """
        contra_prob = float(probs[0])
        entail_prob = float(probs[2])

        if entail_prob >= self.entailment_threshold:
            return (
                Verdict.SUPPORTED,
                entail_prob,
                entail_prob,
                f"Entailed by context (confidence: {entail_prob:.2f})",
            )
        if contra_prob >= self.contradiction_threshold:
            return (
                Verdict.CONTRADICTED,
                contra_prob,
                entail_prob,
                f"Contradicted by context (confidence: {contra_prob:.2f})",
            )
        confidence = 1.0 - max(entail_prob, contra_prob)
        return (
            Verdict.UNSUPPORTED,
            confidence,
            entail_prob,
            (
                f"Not sufficiently supported or contradicted by context "
                f"(entailment: {entail_prob:.2f}, contradiction: {contra_prob:.2f})"
            ),
        )

    # ── Public Interface ──────────────────────────────────────────────────────

    def verify(
        self,
        claims: list[Claim],
        context: str | None = None,
        query: str | None = None,
    ) -> list[ClaimVerdict]:
        """Verify claims against context using NLI.

        Args:
            claims: Atomic claims to verify.
            context: Source context/document. Required — raises if missing.
            query: Ignored by NLIVerifier (no query-aware behavior).

        Returns:
            List of ClaimVerdict, one per claim, in input order.

        Raises:
            VerificationError: If context is None/empty.
        """
        if context is None or not context.strip():
            raise VerificationError(
                "NLIVerifier requires context for grounded verification. "
                "Use a retrieval-based verifier for ungrounded mode."
            )
        if not claims:
            return []

        self._load_model()
        context_chunks = self._chunk_context(context)

        return [self._verify_single_claim(claim, context_chunks) for claim in claims]

    def _verify_single_claim(self, claim: Claim, context_chunks: list[str]) -> ClaimVerdict:
        """Verify a single claim against all context chunks.

        Selects the chunk with the strongest signal — whichever of
        (entailment, contradiction) is larger across all chunks — and
        derives the verdict from THAT chunk's full probability vector.
        This guarantees verdict, nli_score, and evidence are always
        mutually consistent (a bug present in naive implementations that
        track entailment and contradiction maxima independently).
        """
        import numpy as np

        if not context_chunks:
            probs = np.array([0.0, 1.0, 0.0])
            best_chunk = ""
        else:
            best_chunk = context_chunks[0]
            probs = None
            best_signal = -1.0

            for chunk in context_chunks:
                chunk_probs = self._run_nli(premise=chunk, hypothesis=claim.text)
                signal = max(float(chunk_probs[0]), float(chunk_probs[2]))
                if signal > best_signal:
                    best_signal = signal
                    probs = chunk_probs
                    best_chunk = chunk

            if probs is None:
                probs = np.array([0.0, 1.0, 0.0])

        verdict, confidence, entail_prob, reason = self._probs_to_verdict(probs)
        evidence_snippet = extract_evidence_snippet(best_chunk, claim.text)

        return ClaimVerdict(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            nli_score=entail_prob,
            evidence=evidence_snippet,
            reason=reason,
            verification_mode=VerificationMode.GROUNDED,
        )

    def verify_batch(
        self,
        claims: list[Claim],
        context: str,
        batch_size: int = 16,
    ) -> list[ClaimVerdict]:
        """Batch-verify claims for better GPU utilization.

        Builds all (chunk, claim) pairs up front and runs NLI inference in
        batches of `batch_size`, then aggregates per claim by selecting the
        chunk with the strongest signal — identical selection logic to
        verify()/_verify_single_claim(), guaranteeing equivalent results.

        Args:
            claims: Atomic claims to verify.
            context: Source context/document.
            batch_size: Number of (premise, hypothesis) pairs per forward pass.

        Returns:
            List of ClaimVerdict, one per claim, in input order. Equivalent
            to calling verify() but with batched inference for speed.

        Raises:
            VerificationError: If context is empty.
        """
        import numpy as np
        import torch

        if not context or not context.strip():
            raise VerificationError("NLIVerifier requires non-empty context.")
        if not claims:
            return []

        self._load_model()
        context_chunks = self._chunk_context(context)

        if not context_chunks:
            return [
                ClaimVerdict(
                    claim=claim,
                    verdict=Verdict.UNSUPPORTED,
                    confidence=1.0,
                    nli_score=0.0,
                    evidence=None,
                    reason="No context available after chunking.",
                    verification_mode=VerificationMode.GROUNDED,
                )
                for claim in claims
            ]

        # Build all (premise, hypothesis) pairs with index bookkeeping
        pairs: list[tuple[str, str]] = []
        pair_claim_idx: list[int] = []
        pair_chunk_idx: list[int] = []
        for ci, claim in enumerate(claims):
            for chi, chunk in enumerate(context_chunks):
                pairs.append((chunk, claim.text))
                pair_claim_idx.append(ci)
                pair_chunk_idx.append(chi)

        # Run NLI in batches, reordering probs consistently with _run_nli
        all_probs: list[Any] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            premises = [p[0] for p in batch]
            hypotheses = [p[1] for p in batch]

            inputs = self._tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                max_length=self.max_context_tokens,
                truncation=True,
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                raw_probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()

            for row in raw_probs:
                all_probs.append(self._reorder_probs(row, np))

        # Aggregate per claim: pick chunk with strongest signal
        verdicts: list[ClaimVerdict] = []
        for ci, claim in enumerate(claims):
            relevant_indices = [j for j in range(len(pairs)) if pair_claim_idx[j] == ci]

            best_idx = max(
                relevant_indices,
                key=lambda j: max(float(all_probs[j][0]), float(all_probs[j][2])),
            )
            best_probs = all_probs[best_idx]
            best_chunk = context_chunks[pair_chunk_idx[best_idx]]

            verdict, confidence, entail_prob, reason = self._probs_to_verdict(best_probs)
            evidence = extract_evidence_snippet(best_chunk, claim.text)

            verdicts.append(
                ClaimVerdict(
                    claim=claim,
                    verdict=verdict,
                    confidence=confidence,
                    nli_score=entail_prob,
                    evidence=evidence,
                    reason=reason,
                    verification_mode=VerificationMode.GROUNDED,
                )
            )

        return verdicts

    def _chunk_context(self, context: str) -> list[str]:
        """Split context into overlapping, token-bounded chunks.

        Delegates to verifier.utils.chunk_context with this instance's
        configured token budget and overlap.
        """
        return chunk_context(
            context,
            self._tokenizer,
            max_tokens=self.max_context_tokens,
            overlap_tokens=self.chunk_overlap_tokens,
            reserved_tokens=_RESERVED_TOKENS_FOR_HYPOTHESIS,
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if the NLI model can be loaded successfully."""
        try:
            self._load_model()
            return True
        except ModelLoadError:
            return False

    def unload(self) -> None:
        """Unload the model from memory, freeing GPU VRAM.

        Example:
            >>> verifier = NLIVerifier()
            >>> verdicts = verifier.verify(claims, context=doc)
            >>> verifier.unload()
        """
        if not self._loaded:
            return

        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        self._loaded = False

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(
                    "GPU memory after unload: %.0f MB allocated",
                    torch.cuda.memory_allocated() / 1e6,
                )
        except ImportError:
            pass

        logger.info("NLI model unloaded")
