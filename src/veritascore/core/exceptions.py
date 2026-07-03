"""Custom exception hierarchy for VeritasCore.

All exceptions inherit from VeritasCoreError for easy catch-all handling.

Exception hierarchy:
    VeritasCoreError
    ├── ModelLoadError       — Failed to load an ML model
    ├── DecompositionError   — Failed to decompose response into claims
    ├── VerificationError    — Failed to verify a claim
    ├── RetrievalError       — Failed to retrieve evidence from search
    ├── ConfigurationError   — Invalid configuration
    └── HardwareError        — Hardware requirements not met

Example:
    >>> from veritascore.core.exceptions import VeritasCoreError, ModelLoadError
    >>> try:
    ...     engine.load_model("bad-model-name")
    ... except ModelLoadError as e:
    ...     print(f"Model load failed: {e}")
    ... except VeritasCoreError as e:
    ...     print(f"General error: {e}")
"""


class VeritasCoreError(Exception):
    """Base exception for all VeritasCore errors.

    Catch this to handle any VeritasCore-specific error.
    """


class ModelLoadError(VeritasCoreError):
    """Failed to load an ML model.

    Raised when a model cannot be loaded from disk or HuggingFace Hub,
    e.g. due to insufficient VRAM, corrupted weights, or missing files.
    """


class DecompositionError(VeritasCoreError):
    """Failed to decompose an LLM response into atomic claims.

    Raised when the claim decomposer produces invalid output or fails
    to parse the LLM response.
    """


class VerificationError(VeritasCoreError):
    """Failed to verify a claim.

    Raised when NLI inference, retrieval verification, or consistency
    checking fails for a specific claim.
    """


class RetrievalError(VeritasCoreError):
    """Failed to retrieve evidence from a search provider.

    Raised when web search fails due to network issues, invalid API keys,
    rate limiting, or provider unavailability.
    """


class ConfigurationError(VeritasCoreError):
    """Invalid or missing configuration.

    Raised when required configuration values are missing or invalid,
    e.g. missing API key for ungrounded mode.
    """


class HardwareError(VeritasCoreError):
    """Hardware requirements not met.

    Raised when the current hardware cannot satisfy the engine's
    requirements, e.g. insufficient VRAM for the selected model.
    """
