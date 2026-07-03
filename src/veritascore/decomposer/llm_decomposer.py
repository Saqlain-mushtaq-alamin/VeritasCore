"""LLM-based claim decomposition using a local language model.

Primary decomposer for VeritasCore. Uses a local causal LLM (default:
Phi-3-mini-4k-instruct, fallback: Qwen2.5-3B-Instruct) to extract
atomic claims from LLM responses.

Key design decisions:
    - Lazy model loading (load only when first decompose() is called)
    - Supports Ollama as an alternative backend (no HuggingFace Transformers)
    - unload() frees GPU memory so the NLI model (Phase 2) can be loaded
    - Falls back to RuleDecomposer on LLM failure if fallback_on_error=True
    - Temperature=0.1 for deterministic, consistent extraction

Memory note:
    Phi-3-mini-4k-instruct in float16 ≈ 4GB VRAM.
    cross-encoder/nli-deberta-v3-base ≈ 1.5GB VRAM.
    Both cannot fit simultaneously on an 8GB GPU.
    Call unload() after decomposition, before loading NLI model (Phase 2).
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from veritascore.core.config import EngineConfig
from veritascore.core.exceptions import DecompositionError, ModelLoadError
from veritascore.core.types import Claim
from veritascore.decomposer.base import BaseDecomposer
from veritascore.decomposer.prompts import (
    DECOMPOSITION_VERSION,
    build_decomposition_prompt,
    get_system_prompt,
)
from veritascore.decomposer.rule_decomposer import RuleDecomposer
from veritascore.decomposer.span_mapper import map_claims_to_spans

logger = logging.getLogger(__name__)

# Regex to parse numbered claim lines: "1. text" or "1) text"
_NUMBERED_LINE = re.compile(r'^\s*\d+[.)]\s*(.+)$')

# Maximum tokens to generate for the claim list
_MAX_NEW_TOKENS = 1024

# Generation parameters
_GENERATION_KWARGS: dict[str, Any] = {
    "max_new_tokens": _MAX_NEW_TOKENS,
    "temperature": 0.1,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
}


class LLMDecomposer(BaseDecomposer):
    """Decompose LLM responses into atomic claims using a local language model.

    Supports two backends:
        - "transformers" (default): HuggingFace Transformers pipeline
        - "ollama": Ollama local server (requires `pip install ollama`)

    Args:
        config: EngineConfig instance. Defaults to EngineConfig.default().
        backend: "transformers" or "ollama".
        fallback_on_error: If True, fall back to RuleDecomposer on LLM failure.
        prompt_version: Which prompt template version to use.

    Example:
        >>> decomposer = LLMDecomposer(fallback_on_error=True)
        >>> claims = decomposer.decompose(
        ...     "Einstein developed relativity and won the Nobel Prize in 1921.",
        ...     query="Tell me about Einstein.",
        ... )
        >>> decomposer.unload()  # Free GPU memory before loading NLI model
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        backend: str = "transformers",
        fallback_on_error: bool = True,
        prompt_version: str = DECOMPOSITION_VERSION,
    ) -> None:
        if backend not in ("transformers", "ollama"):
            raise ValueError(f"backend must be 'transformers' or 'ollama', got '{backend}'")

        self.config = config or EngineConfig.default()
        self.backend = backend
        self.fallback_on_error = fallback_on_error
        self.prompt_version = prompt_version

        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str = "cpu"
        self._loaded: bool = False
        self._fallback = RuleDecomposer()

    # ── Model Loading ─────────────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        """Determine the inference device."""
        import torch
        device = self.config.models.device
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _load_transformers(self) -> None:
        """Load model via HuggingFace Transformers."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.config.models.decomposer_model
        logger.info("Loading decomposer model (transformers): %s", model_name)
        t0 = time.time()

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )

            self._device = self._resolve_device()
            dtype = torch.float16 if self._device != "cpu" else torch.float32

            load_kwargs: dict[str, Any] = {
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }
            if self._device == "cpu":
                load_kwargs["device_map"] = None
            else:
                load_kwargs["device_map"] = "auto"

            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )

            if self._device == "cpu":
                self._model = self._model.to("cpu")

            elapsed = time.time() - t0
            n_params = sum(p.numel() for p in self._model.parameters()) / 1e6
            logger.info(
                "Decomposer loaded: %.0fM params on %s in %.1fs",
                n_params, self._device, elapsed,
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load decomposer model '{model_name}': {e}"
            ) from e

    def _load_ollama(self) -> None:
        """Verify Ollama server is reachable and model is available."""
        try:
            import ollama  # type: ignore[import]
        except ImportError as e:
            raise ModelLoadError(
                "Ollama backend requires 'ollama' package: pip install ollama[dev]"
            ) from e

        model_name = self.config.models.decomposer_model
        logger.info("Using Ollama backend: %s", model_name)
        try:
            # Pull model if not already downloaded
            ollama.pull(model_name)
        except Exception as e:
            raise ModelLoadError(
                f"Failed to pull Ollama model '{model_name}': {e}"
            ) from e

        self._device = "ollama"

    def _load_model(self) -> None:
        """Lazy-load the decomposition model (idempotent)."""
        if self._loaded:
            return
        if self.backend == "transformers":
            self._load_transformers()
        else:
            self._load_ollama()
        self._loaded = True

    # ── Inference ─────────────────────────────────────────────────────────────

    def _generate_transformers(self, messages: list[dict[str, str]]) -> str:
        """Run inference with HuggingFace Transformers."""
        import torch

        inputs = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )

        if self._device != "cpu":
            inputs = inputs.to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(inputs, **_GENERATION_KWARGS)

        # Decode only the newly generated tokens
        generated_ids = outputs[0][inputs.shape[1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _generate_ollama(self, messages: list[dict[str, str]]) -> str:
        """Run inference with Ollama."""
        import ollama  # type: ignore[import]

        response = ollama.chat(
            model=self.config.models.decomposer_model,
            messages=messages,
            options={
                "temperature": _GENERATION_KWARGS["temperature"],
                "top_p": _GENERATION_KWARGS["top_p"],
                "num_predict": _MAX_NEW_TOKENS,
            },
        )
        return response["message"]["content"]

    def _generate(self, messages: list[dict[str, str]]) -> str:
        """Route to the appropriate backend for generation."""
        if self.backend == "transformers":
            return self._generate_transformers(messages)
        return self._generate_ollama(messages)

    # ── Output Parsing ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_numbered_claims(text: str) -> list[str]:
        """Parse numbered claim lines from LLM output.

        Accepts formats: "1. claim text", "1) claim text", " 1. claim text"
        Filters trivially short outputs (< 8 chars).

        Args:
            text: Raw LLM generation string.

        Returns:
            List of clean claim strings.
        """
        claims: list[str] = []
        for line in text.strip().splitlines():
            match = _NUMBERED_LINE.match(line)
            if match:
                claim_text = match.group(1).strip()
                # Strip trailing period added inconsistently by some models
                claim_text = claim_text.rstrip(".")
                if len(claim_text) >= 8:
                    claims.append(claim_text)
        return claims

    # ── Public Interface ──────────────────────────────────────────────────────

    def decompose(
        self,
        response_text: str,
        query: str | None = None,
    ) -> list[Claim]:
        """Decompose an LLM response into atomic factual claims.

        Args:
            response_text: Full LLM response to decompose.
            query: Optional original user query for context.

        Returns:
            List of Claim objects with text, id, source_span, source_text.

        Raises:
            DecompositionError: If decomposition fails and fallback_on_error=False.
        """
        if not response_text or not response_text.strip():
            return []

        # Enforce max-claims limit from config
        max_claims = self.config.max_claims_per_response

        try:
            self._load_model()

            t0 = time.time()

            system_prompt = get_system_prompt(self.prompt_version)
            user_prompt = build_decomposition_prompt(
                response_text, query, version=self.prompt_version
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            raw_output = self._generate(messages)
            elapsed = time.time() - t0

            # Parse numbered claims from the output
            raw_claims = self._parse_numbered_claims(raw_output)

            if not raw_claims:
                logger.warning(
                    "LLM produced no parseable claims (%.1fs). "
                    "Raw output snippet: %s",
                    elapsed,
                    raw_output[:200],
                )
                if self.fallback_on_error:
                    logger.info("Falling back to RuleDecomposer")
                    return self._fallback.decompose(response_text, query)
                raise DecompositionError(
                    "LLM decomposer produced no parseable claims"
                )

            # Truncate to max_claims
            if len(raw_claims) > max_claims:
                logger.warning(
                    "Truncating %d claims to max_claims=%d",
                    len(raw_claims), max_claims,
                )
                raw_claims = raw_claims[:max_claims]

            # Map claim texts to source spans
            span_mappings = map_claims_to_spans(raw_claims, response_text)

            claims: list[Claim] = []
            for i, (claim_text, (span, source_text)) in enumerate(
                zip(raw_claims, span_mappings, strict=True)
            ):
                claims.append(Claim(
                    id=f"c{i + 1:03d}",
                    text=claim_text,
                    source_span=span,
                    source_text=source_text,
                ))

            logger.info(
                "LLMDecomposer: %d claims in %.2fs (backend=%s)",
                len(claims), elapsed, self.backend,
            )
            return claims

        except (ModelLoadError, DecompositionError):
            raise
        except Exception as e:
            logger.error("LLMDecomposer unexpected error: %s", e, exc_info=True)
            if self.fallback_on_error:
                logger.info("Falling back to RuleDecomposer due to error: %s", e)
                return self._fallback.decompose(response_text, query)
            raise DecompositionError(
                f"LLM decomposition failed unexpectedly: {e}"
            ) from e

    def is_available(self) -> bool:
        """Return True if the model can be loaded successfully."""
        try:
            self._load_model()
            return True
        except (ModelLoadError, Exception):
            return False

    def unload(self) -> None:
        """Unload the model from memory, freeing GPU VRAM.

        Call this after decomposition is complete and before loading
        the NLI model (Phase 2). On an 8GB GPU, both models cannot
        fit simultaneously.

        Example:
            >>> decomposer = LLMDecomposer()
            >>> claims = decomposer.decompose(response)
            >>> decomposer.unload()  # ← free 4GB VRAM
            >>> # Now safe to load NLI model
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

        logger.info("LLMDecomposer unloaded")
