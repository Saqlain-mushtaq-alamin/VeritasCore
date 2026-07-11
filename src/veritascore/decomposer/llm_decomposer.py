"""LLM-based claim decomposition using a local language model.

Primary decomposer for VeritasCore. Uses a local causal LLM (default:
Phi-3-mini-4k-instruct, fallback: Qwen2.5-3B-Instruct) to extract
atomic claims from LLM responses.

Key design decisions:
    - Lazy model loading (load only when first decompose() is called)
    - Supports Ollama as an alternative backend (no HuggingFace Transformers)
    - unload() frees GPU memory so the NLI model (Phase 2) can be loaded
    - Falls back to RuleDecomposer on LLM failure if fallback_on_error=True
    - Temperature=0.0 for deterministic, consistent extraction
    - Pre-filtering skips non-factual inputs before invoking the LLM
    - Post-filtering removes opinion/hedge claims from LLM output

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

# Regex for dash/bullet list lines: "- text" or "* text"
_BULLET_LINE = re.compile(r'^\s*[-•]\s*(.+)$')

# Maximum tokens to generate for the claim list.
# 384 tokens is sufficient for ~15 atomic claims with pronoun-resolved text.
_MAX_NEW_TOKENS = 384

# Generation parameters — deterministic (temperature=0)
_GENERATION_KWARGS: dict[str, Any] = {
    "max_new_tokens": _MAX_NEW_TOKENS,
    "do_sample": False,
    "repetition_penalty": 1.1,
}

# ── Pre-filtering constants ──────────────────────────────────────────────────

# Minimum text length to send to LLM (shorter inputs bypass to rule-based)
_MIN_LLM_INPUT_LENGTH = 5

# Pattern to detect code blocks
_CODE_PATTERN = re.compile(
    r'^\s*(?:def |class |import |from |if |for |while |return |'
    r'print\(|console\.|var |let |const |function |public |private |'
    r'#include|#define|package |using |namespace )',
    re.MULTILINE,
)

# Pattern to detect if text is purely a question
_PURE_QUESTION = re.compile(
    r'^\s*(?:who|what|when|where|why|how|is|are|was|were|do|does|did'
    r'|can|could|would|should|will|shall|have|has|had)\b.*\?\s*$',
    re.IGNORECASE,
)

# ── Post-filtering constants ─────────────────────────────────────────────────

# Hedge/opinion words that indicate a claim is subjective ONLY when they are
# at the BEGINNING of the claim (sentence-level hedges), NOT when embedded
# inside a factual statement (e.g. "approximately 300,000 km/s" is factual).
#
# We match these only at claim-start to avoid falsely dropping legitimate
# claims like "Solar panel costs dropped by about 90%".
_SENTENCE_HEDGE_STARTERS: tuple[str, ...] = (
    "probably ", "perhaps ", "possibly ", "maybe ", "arguably ",
    "supposedly ", "presumably ",
)

# Claim-level opinion prefixes (case-insensitive start-of-claim)
_OPINION_CLAIM_PREFIXES: tuple[str, ...] = (
    "i think", "i believe", "i feel", "in my opinion",
    "in my view", "personally", "it seems", "it appears",
)

# Preamble / apology / meta-commentary patterns the model sometimes outputs.
# We strip lines matching these before attempting to parse numbered items.
# Phi-3 commonly generates explanations like:
#   "Since this instruction requires us only to provide..."
#   "Since we need to extract only factual claims..."
#   "Based on the instruction, I will only extract..."
_APOLOGY_LINE = re.compile(
    r'^\s*(?:i apologize|i\'m sorry|since i am|since we need|since this |since the |'
    r'based upon your|based on (the|your|this)|here (?:is|are) the|to address your|'
    r'since your instruction|please note|note that|'
    r'unfortunately|as per your|i cannot provide|'
    r'the following (?:are|is)|as instructed|per (the|your)|'
    r'following (the|your)|in accordance|as per (the|your))',
    re.IGNORECASE,
)

# Lines that look like model meta-commentary rather than facts:
# e.g. "*Adjusted Response Based On Ruleset Constraints:**"
_META_LINE = re.compile(
    r'^\s*\*+\s*(?:adjusted|note|output|response|based|according|per rule)',
    re.IGNORECASE,
)


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
        """Load model via HuggingFace Transformers.

        GPU strategy:
          - If CUDA is available AND free VRAM >= 4 GB, load all weights onto
            cuda:0 with float16. This keeps 100% GPU utilisation and avoids the
            PCIe bottleneck from CPU offloading (which causes ~50% GPU util).
          - If VRAM is tight, fall back to device_map='auto' which splits across
            GPU+CPU.  Slower, but prevents OOM.
          - CPU-only: float32, no device_map.

        Attention implementation:
          - Uses 'sdpa' (PyTorch Scaled Dot Product Attention) on CUDA. This is
            2-3× faster than 'eager' on RTX 4060 (Ampere/Ada) with CUDA ≥11.8
            and automatically exploits Flash Attention 2 kernels when available.
          - Falls back to 'eager' if 'sdpa' is not supported by the model.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.config.models.decomposer_model
        logger.info("Loading decomposer model (transformers): %s", model_name)
        t0 = time.time()

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)

            self._device = self._resolve_device()

            load_kwargs: dict[str, Any] = {}

            if self._device != "cpu":
                load_kwargs["torch_dtype"] = torch.float16

                # Check free VRAM — if ≥ 4 GB available, load entirely on GPU
                # to avoid the CPU offload that throttles utilisation to ~50%.
                free_vram_gb = 0.0
                try:
                    free_bytes, _ = torch.cuda.mem_get_info(0)
                    free_vram_gb = free_bytes / 1e9
                except Exception:
                    pass

                if free_vram_gb >= 4.0:
                    # All layers on GPU — fastest path, 100% GPU utilisation
                    load_kwargs["device_map"] = {"": 0}
                    logger.info(
                        "Forcing full GPU load (%.1f GB free VRAM)", free_vram_gb
                    )
                else:
                    # Fall back to auto-split if VRAM is tight
                    load_kwargs["device_map"] = "auto"
                    logger.info(
                        "Low VRAM (%.1f GB free) — using device_map='auto'",
                        free_vram_gb,
                    )

                # Prefer SDPA for 2-3× speedup on Ampere/Ada GPUs
                try:
                    load_kwargs["attn_implementation"] = "sdpa"
                except Exception:
                    load_kwargs["attn_implementation"] = "eager"
            else:
                load_kwargs["torch_dtype"] = torch.float32
                load_kwargs["device_map"] = None
                load_kwargs["attn_implementation"] = "eager"

            self._model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            )

            if self._device == "cpu":
                self._model = self._model.to("cpu")

            elapsed = time.time() - t0
            n_params = sum(p.numel() for p in self._model.parameters()) / 1e6
            logger.info(
                "Decomposer loaded: %.0fM params on %s in %.1fs (attn=%s)",
                n_params, self._device, elapsed,
                load_kwargs.get("attn_implementation", "?"),
            )
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load decomposer model '{model_name}': {e}"
            ) from e

    def _load_ollama(self) -> None:
        """Verify Ollama server is reachable and model is available."""
        try:
            import ollama  # noqa: F811
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

    # ── Pre-filtering ─────────────────────────────────────────────────────────

    @staticmethod
    def _should_skip_input(text: str) -> bool:
        """Check if the input should be skipped entirely (no LLM call needed).

        Returns True for:
            - Very short text (< 5 chars)
            - Code blocks / programming syntax
            - Pure questions with no factual content
            - Single non-factual words (e.g. "Yes.", "No.", "OK.")

        Args:
            text: Input text to check.

        Returns:
            True if the input has no factual content worth decomposing.
        """
        stripped = text.strip()

        # Too short to contain a meaningful fact
        if len(stripped) < _MIN_LLM_INPUT_LENGTH:
            return True

        # Pure code block
        if _CODE_PATTERN.search(stripped) and not any(
            c.isalpha() and c.isupper() for c in stripped[:50]
            if not stripped[:50].startswith(("def ", "class ", "import "))
        ):
            # Heuristic: if it looks like code throughout, skip
            lines = stripped.splitlines()
            code_lines = sum(1 for line in lines if _CODE_PATTERN.match(line))
            if code_lines >= len(lines) * 0.5:
                return True

        # Pure question (single sentence ending with ?)
        if _PURE_QUESTION.match(stripped):
            return True

        # Single word / very short non-factual response
        words = stripped.rstrip(".!?").split()
        if len(words) <= 1:
            return True

        return False

    # ── Inference ─────────────────────────────────────────────────────────────

    def _generate_transformers(self, messages: list[dict[str, str]]) -> str:
        """Run inference with HuggingFace Transformers."""
        import torch

        # Two-step approach: render chat template to text, then tokenize.
        # This is more reliable across transformers versions than using
        # apply_chat_template with return_tensors directly.
        prompt_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        encoded = self._tokenizer(
            prompt_text,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_ids = encoded.input_ids
        attention_mask = encoded.attention_mask

        if self._device != "cpu":
            # When device_map='auto' (accelerate splits layers across CPU+GPU),
            # self._device may be 'cuda' but the embedding layer may live on
            # 'cpu'. Move inputs to the device of the first model parameter
            # (the embedding layer) to avoid the "tensors on two devices" error.
            try:
                first_param = next(self._model.parameters())
                input_device = first_param.device
            except StopIteration:
                input_device = torch.device(self._device)
            input_ids = input_ids.to(input_device)
            attention_mask = attention_mask.to(input_device)

        prompt_len = input_ids.shape[1]

        gen_kwargs = dict(_GENERATION_KWARGS)
        gen_kwargs["attention_mask"] = attention_mask

        # Suppress pad_token warning — use eos as pad for open-ended generation
        eos_id = self._tokenizer.eos_token_id
        if eos_id is not None:
            gen_kwargs["pad_token_id"] = eos_id

        with torch.no_grad():
            outputs = self._model.generate(input_ids, **gen_kwargs)

        # Decode only the newly generated tokens
        generated_ids = outputs[0][prompt_len:]
        result: str = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return result

    def _generate_ollama(self, messages: list[dict[str, str]]) -> str:
        """Run inference with Ollama."""
        import ollama  # noqa: F811

        response = ollama.chat(
            model=self.config.models.decomposer_model,
            messages=messages,
            options={
                "temperature": 0.0,
                "top_p": 0.9,
                "num_predict": _MAX_NEW_TOKENS,
            },
        )
        result: str = response["message"]["content"]
        return result

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
        Also accepts dash/bullet lists as a fallback: "- claim", "* claim"
        Filters trivially short outputs (< 8 chars).
        Handles the "NONE" sentinel for no-claims case.
        Strips markdown code fences that some models wrap output in.
        Strips leading apology/preamble lines before looking for numbered items.

        Args:
            text: Raw LLM generation string.

        Returns:
            List of clean claim strings.
        """
        # Strip markdown code fences (```plaintext, ```text, ``` etc.)
        stripped = re.sub(
            r'```(?:plaintext|text|markdown)?\s*\n?', '', text
        ).strip()
        # Strip closing fence
        stripped = re.sub(r'\n?```\s*$', '', stripped).strip()

        # Check for the "NONE" sentinel (no factual claims)
        if stripped.upper() in ("NONE", "NONE."):
            return []

        # Remove leading apology / preamble / meta lines so numbered items reachable
        lines_raw = stripped.splitlines()
        lines: list[str] = []
        found_first_claim = False
        for line in lines_raw:
            if not found_first_claim:
                # Skip apology/meta lines before the first numbered claim
                if _APOLOGY_LINE.match(line) or _META_LINE.match(line):
                    logger.debug("Parser: stripped preamble line: %s", line[:80])
                    continue
                # Once we see a numbered line, start collecting
                if _NUMBERED_LINE.match(line):
                    found_first_claim = True
            # Always drop meta-commentary lines even after first claim found
            if _META_LINE.match(line):
                logger.debug("Parser: stripped meta line: %s", line[:80])
                continue
            lines.append(line)

        # ── Pass 1: numbered lines ────────────────────────────────────────────
        claims: list[str] = []
        for line in lines:
            match = _NUMBERED_LINE.match(line)
            if match:
                claim_text = match.group(1).strip()
                # Strip trailing period added inconsistently by some models
                claim_text = claim_text.rstrip(".")
                # Skip lines that are clearly meta-commentary masquerading as claims
                if _META_LINE.match(claim_text):
                    continue
                if len(claim_text) >= 6:
                    claims.append(claim_text)

        if claims:
            return claims

        # ── Pass 2: bullet / dash list fallback ──────────────────────────────
        for line in lines:
            match = _BULLET_LINE.match(line)
            if match:
                claim_text = match.group(1).strip().rstrip(".")
                if _META_LINE.match(claim_text):
                    continue
                if len(claim_text) >= 6:
                    claims.append(claim_text)

        return claims

    # ── Post-filtering ────────────────────────────────────────────────────────

    @staticmethod
    def _filter_claims(raw_claims: list[str]) -> list[str]:
        """Post-filter claims to remove opinions and sentence-level hedges.

        KEY CHANGE from v3: hedge words (probably, perhaps, etc.) are only
        filtered when they appear at the BEGINNING of a claim, indicating a
        sentence-level hedge. Embedded qualifiers like "approximately",
        "about", "commonly", "generally", "typically" are preserved because
        they are part of verifiable factual statements.

        Args:
            raw_claims: Claims parsed from LLM output.

        Returns:
            Filtered list of factual claims.
        """
        filtered: list[str] = []
        for claim in raw_claims:
            claim_lower = claim.lower().strip()

            # Skip opinion-prefixed claims
            if any(claim_lower.startswith(prefix) for prefix in _OPINION_CLAIM_PREFIXES):
                logger.debug("Post-filter: skipped opinion claim: %s", claim[:60])
                continue

            # Skip claims that START with a sentence-level hedge word
            # (e.g. "Probably the most important..." or "Perhaps Einstein...")
            # but NOT claims that merely CONTAIN embedded qualifiers
            # (e.g. "Solar costs dropped by approximately 90%")
            if any(claim_lower.startswith(hedge) for hedge in _SENTENCE_HEDGE_STARTERS):
                logger.debug("Post-filter: skipped sentence-hedge claim: %s", claim[:60])
                continue

            # Skip claims that are absurdly long (hallucinated/rambling)
            # A well-formed atomic claim should rarely exceed 300 chars
            if len(claim) > 350:
                logger.debug("Post-filter: skipped over-long claim: %s", claim[:60])
                continue

            filtered.append(claim)

        if len(filtered) < len(raw_claims):
            logger.info(
                "Post-filter: %d → %d claims (removed %d)",
                len(raw_claims), len(filtered), len(raw_claims) - len(filtered),
            )

        return filtered

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

        # Pre-filter: skip inputs that clearly have no factual content
        if self._should_skip_input(response_text):
            logger.info(
                "Pre-filter: input skipped (non-factual or too short): '%s'",
                response_text[:60],
            )
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

            # Post-filter: remove opinions and sentence-level hedges
            raw_claims = self._filter_claims(raw_claims)

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

        except ModelLoadError:
            if self.fallback_on_error:
                logger.info(
                    "Model load failed, falling back to RuleDecomposer"
                )
                return self._fallback.decompose(response_text, query)
            raise
        except DecompositionError:
            raise
        except Exception as e:
            logger.error(
                "LLMDecomposer unexpected error: %s", e, exc_info=True,
            )
            if self.fallback_on_error:
                logger.info(
                    "Falling back to RuleDecomposer due to error: %s", e,
                )
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
