"""Domain profile registry — loads, caches, and resolves profiles by name."""

from __future__ import annotations

import logging
from pathlib import Path

from veritascore.profiles.base import DomainProfile

logger = logging.getLogger(__name__)

DEFAULT_PROFILES_DIR = Path(__file__).parent.parent.parent.parent.parent / "configs" / "profiles"


class ProfileRegistry:
    """Singleton-ish registry that loads and caches DomainProfile instances.

    Example:
        >>> registry = ProfileRegistry()
        >>> profile = registry.get("medical")
        >>> registry.list_profiles()
        ['finance', 'general', 'legal', 'medical']
    """

    def __init__(self, profiles_dir: Path | None = None) -> None:
        self._dir = profiles_dir or DEFAULT_PROFILES_DIR
        self._cache: dict[str, DomainProfile] = {}
        self._scanned: bool = False

    def _scan(self) -> None:
        """Scan profiles_dir for *.yaml files and load them into the cache."""
        if self._scanned:
            return
        self._scanned = True
        if not self._dir.exists():
            logger.warning("Profiles directory not found: %s", self._dir)
            return
        for yaml_path in sorted(self._dir.glob("*.yaml")):
            try:
                profile = DomainProfile.from_yaml(yaml_path)
                self._cache[profile.name] = profile
                logger.debug("Loaded profile '%s' from %s", profile.name, yaml_path)
            except Exception as e:
                logger.warning("Failed to load profile from %s: %s", yaml_path, e)

    def get(self, name: str) -> DomainProfile:
        """Return a profile by name.

        Raises:
            KeyError: If the named profile is not found.
        """
        self._scan()
        if name not in self._cache:
            raise KeyError(f"Profile '{name}' not found. Available: {sorted(self._cache.keys())}")
        return self._cache[name]

    def get_or_default(self, name: str) -> DomainProfile:
        """Return a profile by name, falling back to 'general' on miss."""
        self._scan()
        if name in self._cache:
            return self._cache[name]
        logger.warning("Profile '%s' not found; falling back to 'general'.", name)
        if "general" in self._cache:
            return self._cache["general"]
        return DomainProfile(
            name="general",
            display_name="General",
            description="Default profile — sensible defaults for all domains.",
        )

    def register(self, profile: DomainProfile) -> None:
        """Register a profile programmatically (without a YAML file)."""
        self._cache[profile.name] = profile

    def list_profiles(self) -> list[str]:
        """Return sorted list of all registered profile names."""
        self._scan()
        return sorted(self._cache.keys())

    def load_from_yaml(self, path: str | Path) -> DomainProfile:
        """Load a profile from a YAML file and register it."""
        profile = DomainProfile.from_yaml(path)
        self._cache[profile.name] = profile
        return profile


_default_registry: ProfileRegistry | None = None


def get_registry() -> ProfileRegistry:
    """Return the module-level default registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ProfileRegistry()
    return _default_registry


def get_profile(name: str) -> DomainProfile:
    """Convenience: get a profile from the default registry."""
    return get_registry().get(name)


def get_profile_or_default(name: str) -> DomainProfile:
    """Convenience: get a profile or fall back to 'general'."""
    return get_registry().get_or_default(name)
