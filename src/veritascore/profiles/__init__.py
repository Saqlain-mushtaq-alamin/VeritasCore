"""Domain profiles module — configurable per-domain verification behaviour."""

from veritascore.profiles.base import (
    DomainProfile,
    SeverityWeights,
    ThresholdConfig,
    VerificationConstraints,
)
from veritascore.profiles.registry import (
    ProfileRegistry,
    get_profile,
    get_profile_or_default,
    get_registry,
)

__all__ = [
    "DomainProfile",
    "VerificationConstraints",
    "SeverityWeights",
    "ThresholdConfig",
    "ProfileRegistry",
    "get_profile",
    "get_profile_or_default",
    "get_registry",
]
