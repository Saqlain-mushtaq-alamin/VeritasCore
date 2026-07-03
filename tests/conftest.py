"""Shared test fixtures for VeritasCore.

All fixtures here are available to every test file without importing.
Add only fixtures that are genuinely shared across multiple test modules.
Phase-specific fixtures belong in the unit test file for that phase.
"""

from __future__ import annotations

import pytest

from veritascore.core.config import EngineConfig, ModelConfig, SearchConfig
from veritascore.core.types import (
    Claim,
    ClaimVerdict,
    Verdict,
    VerificationMode,
)

# ── Config Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def default_config() -> EngineConfig:
    """Default engine configuration for testing (no GPU assumed)."""
    return EngineConfig(
        models=ModelConfig(device="cpu"),
        search=SearchConfig(provider="none"),
        log_level="DEBUG",
    )


@pytest.fixture
def offline_config() -> EngineConfig:
    """Offline-only config for unit tests that must not touch the network."""
    return EngineConfig(
        models=ModelConfig(device="cpu"),
        search=SearchConfig(provider="none", cache_enabled=False),
        verification_mode="offline",
    )


# ── Sample Text Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_response() -> str:
    """A sample LLM response for testing the full pipeline."""
    return (
        "The Eiffel Tower is located in Paris, France. "
        "It was completed in 1889 as the centerpiece of the World's Fair. "
        "The tower stands 330 meters tall and attracts millions of visitors annually."
    )


@pytest.fixture
def sample_context() -> str:
    """Sample grounding context for grounded verification tests."""
    return (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars "
        "in Paris, France. It was constructed from 1887 to 1889 as the centerpiece "
        "of the 1889 World's Fair. The tower is 330 metres (1,083 ft) tall, including "
        "antennas at the top."
    )


@pytest.fixture
def sample_query() -> str:
    """A sample user query."""
    return "Tell me about the Eiffel Tower."


# ── Claim Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def claim_supported() -> Claim:
    """A claim that is factually correct (Eiffel Tower height)."""
    return Claim(
        id="c001",
        text="The Eiffel Tower stands 330 meters tall.",
        source_span=(0, 40),
        source_text="The Eiffel Tower stands 330 meters tall.",
    )


@pytest.fixture
def claim_contradicted() -> Claim:
    """A claim that is factually incorrect (wrong height)."""
    return Claim(
        id="c002",
        text="The Eiffel Tower is 500 meters tall.",
        source_span=(0, 36),
        source_text="The Eiffel Tower is 500 meters tall.",
    )


@pytest.fixture
def claim_unsupported() -> Claim:
    """A claim that cannot be verified from the sample context."""
    return Claim(
        id="c003",
        text="The Eiffel Tower was designed by Gustave Eiffel himself.",
        source_span=(0, 55),
        source_text="The Eiffel Tower was designed by Gustave Eiffel himself.",
    )


@pytest.fixture
def sample_claims(
    claim_supported: Claim,
    claim_contradicted: Claim,
    claim_unsupported: Claim,
) -> list[Claim]:
    """A list of diverse sample claims for pipeline testing."""
    return [claim_supported, claim_contradicted, claim_unsupported]


# ── ClaimVerdict Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def verdict_supported(claim_supported: Claim) -> ClaimVerdict:
    """A supported claim verdict."""
    return ClaimVerdict(
        claim=claim_supported,
        verdict=Verdict.SUPPORTED,
        confidence=0.92,
        nli_score=0.91,
        reason="The provided context explicitly states the tower is 330 metres tall.",
        verification_mode=VerificationMode.GROUNDED,
        evidence="The tower is 330 metres (1,083 ft) tall, including antennas at the top.",
    )


@pytest.fixture
def verdict_contradicted(claim_contradicted: Claim) -> ClaimVerdict:
    """A contradicted claim verdict."""
    return ClaimVerdict(
        claim=claim_contradicted,
        verdict=Verdict.CONTRADICTED,
        confidence=0.88,
        nli_score=0.05,
        reason="Context states 330m; claim states 500m — direct contradiction.",
        verification_mode=VerificationMode.GROUNDED,
        evidence="The tower is 330 metres (1,083 ft) tall.",
    )


@pytest.fixture
def verdict_unsupported(claim_unsupported: Claim) -> ClaimVerdict:
    """An unsupported claim verdict."""
    return ClaimVerdict(
        claim=claim_unsupported,
        verdict=Verdict.UNSUPPORTED,
        confidence=0.60,
        nli_score=0.45,
        reason="The provided context does not mention the designer by name.",
        verification_mode=VerificationMode.GROUNDED,
        evidence=None,
    )
