"""Unit tests for Phase 6 domain profiles (DomainProfile + ProfileRegistry)."""
# ruff: noqa: E501

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from veritascore.profiles import (
    DomainProfile,
    ProfileRegistry,
    SeverityWeights,
    ThresholdConfig,
    VerificationConstraints,
)


class TestDomainProfileDefaults:
    def test_minimal_required_fields(self) -> None:
        p = DomainProfile(name="test", display_name="Test", description="A test profile.")
        assert p.name == "test"
        assert p.thresholds.entailment_threshold == pytest.approx(0.70)
        assert p.thresholds.contradiction_threshold == pytest.approx(0.50)
        assert p.constraints.required_source_count == 1
        assert p.severity.contradicted_weight == pytest.approx(2.0)

    def test_default_sub_models_are_distinct_instances(self) -> None:
        p1 = DomainProfile(name="a", display_name="A", description="A")
        p2 = DomainProfile(name="b", display_name="B", description="B")
        p1.thresholds.entailment_threshold = 0.99
        assert p2.thresholds.entailment_threshold != 0.99

    def test_threshold_bounds_enforced(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DomainProfile(
                name="bad", display_name="Bad", description="Bad",
                thresholds=ThresholdConfig(entailment_threshold=1.5),
            )


class TestDomainProfileFromYaml:
    def test_loads_complete_profile(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""
            name: general
            display_name: General
            description: General profile.
            thresholds:
              entailment_threshold: 0.70
              contradiction_threshold: 0.50
              consistency_threshold: 0.30
              overall_trust_warning: 0.60
              overall_trust_critical: 0.30
            constraints:
              blocked_domains:
                - reddit.com
              required_source_count: 1
              max_search_results: 5
            severity:
              contradicted_weight: 2.0
        """)
        f = tmp_path / "general.yaml"
        f.write_text(yaml_content)
        p = DomainProfile.from_yaml(f)
        assert p.name == "general"
        assert p.thresholds.entailment_threshold == pytest.approx(0.70)
        assert "reddit.com" in p.constraints.blocked_domains

    def test_nested_sub_models_coerced_from_dict(self, tmp_path: Path) -> None:
        """Regression: from_yaml must use model_validate(), not cls(**data),
        to correctly coerce nested dict fields into Pydantic sub-models."""
        yaml_content = textwrap.dedent("""
            name: medical
            display_name: Medical
            description: Medical profile.
            thresholds:
              entailment_threshold: 0.85
              contradiction_threshold: 0.60
              consistency_threshold: 0.40
              overall_trust_warning: 0.75
              overall_trust_critical: 0.50
            severity:
              contradicted_weight: 4.0
        """)
        f = tmp_path / "medical.yaml"
        f.write_text(yaml_content)
        p = DomainProfile.from_yaml(f)
        assert isinstance(p.thresholds, ThresholdConfig)
        assert isinstance(p.severity, SeverityWeights)
        assert p.thresholds.entailment_threshold == pytest.approx(0.85)
        assert p.severity.contradicted_weight == pytest.approx(4.0)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DomainProfile.from_yaml(tmp_path / "nonexistent.yaml")

    def test_partial_yaml_uses_defaults(self, tmp_path: Path) -> None:
        f = tmp_path / "partial.yaml"
        f.write_text("name: partial\ndisplay_name: Partial\ndescription: Minimal.\n")
        p = DomainProfile.from_yaml(f)
        assert p.thresholds.entailment_threshold == pytest.approx(0.70)
        assert p.constraints.required_source_count == 1


class TestApplyToVerifier:
    """G6 criterion 6: Profile thresholds correctly modify verifier behaviour."""

    def test_apply_to_nli_verifier(self) -> None:
        class FakeNLI:
            entailment_threshold = 0.70
            contradiction_threshold = 0.50

        verifier = FakeNLI()
        p = DomainProfile(
            name="m", display_name="M", description="",
            thresholds=ThresholdConfig(entailment_threshold=0.85, contradiction_threshold=0.60),
        )
        p.apply_to_nli_verifier(verifier)
        assert verifier.entailment_threshold == pytest.approx(0.85)
        assert verifier.contradiction_threshold == pytest.approx(0.60)

    def test_apply_to_consistency_checker(self) -> None:
        class FakeChecker:
            relevance_threshold = 0.30

        checker = FakeChecker()
        p = DomainProfile(
            name="m", display_name="M", description="",
            thresholds=ThresholdConfig(consistency_threshold=0.45),
        )
        p.apply_to_consistency_checker(checker)
        assert checker.relevance_threshold == pytest.approx(0.45)

    def test_general_and_medical_thresholds_differ(self) -> None:
        general = DomainProfile(name="g", display_name="G", description="")
        medical = DomainProfile(
            name="m", display_name="M", description="",
            thresholds=ThresholdConfig(entailment_threshold=0.85),
        )
        assert general.thresholds.entailment_threshold != medical.thresholds.entailment_threshold


class TestGetTrustLevel:
    def test_ok(self) -> None:
        p = DomainProfile(name="g", display_name="G", description="")
        assert p.get_trust_level(0.9) == "ok"
        assert p.get_trust_level(0.6) == "ok"

    def test_warning(self) -> None:
        p = DomainProfile(name="g", display_name="G", description="")
        assert p.get_trust_level(0.59) == "warning"
        assert p.get_trust_level(0.31) == "warning"

    def test_critical(self) -> None:
        p = DomainProfile(name="g", display_name="G", description="")
        assert p.get_trust_level(0.29) == "critical"
        assert p.get_trust_level(0.0) == "critical"

    def test_boundary_warning_is_ok(self) -> None:
        p = DomainProfile(
            name="g", display_name="G", description="",
            thresholds=ThresholdConfig(overall_trust_warning=0.60, overall_trust_critical=0.30),
        )
        assert p.get_trust_level(0.60) == "ok"
        assert p.get_trust_level(0.599) == "warning"

    def test_medical_higher_thresholds(self) -> None:
        p = DomainProfile(
            name="m", display_name="M", description="",
            thresholds=ThresholdConfig(overall_trust_warning=0.75, overall_trust_critical=0.50),
        )
        assert p.get_trust_level(0.7) == "warning"


class TestProfileRegistry:
    def _make_registry(self, tmp_path: Path) -> tuple[ProfileRegistry, Path]:
        profiles_dir = tmp_path / "profiles"
        profiles_dir.mkdir()
        return ProfileRegistry(profiles_dir=profiles_dir), profiles_dir

    def _write_profile(self, profiles_dir: Path, name: str) -> None:
        content = f"name: {name}\ndisplay_name: {name.capitalize()}\ndescription: A {name} profile.\n"
        (profiles_dir / f"{name}.yaml").write_text(content)

    def test_empty_registry(self, tmp_path: Path) -> None:
        registry, _ = self._make_registry(tmp_path)
        assert registry.list_profiles() == []

    def test_scan_loads_yaml_files(self, tmp_path: Path) -> None:
        registry, d = self._make_registry(tmp_path)
        self._write_profile(d, "general")
        self._write_profile(d, "medical")
        profiles = registry.list_profiles()
        assert "general" in profiles
        assert "medical" in profiles

    def test_get_existing_profile(self, tmp_path: Path) -> None:
        registry, d = self._make_registry(tmp_path)
        self._write_profile(d, "general")
        assert registry.get("general").name == "general"

    def test_get_missing_raises(self, tmp_path: Path) -> None:
        registry, _ = self._make_registry(tmp_path)
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_get_or_default_falls_back_to_general(self, tmp_path: Path) -> None:
        registry, d = self._make_registry(tmp_path)
        self._write_profile(d, "general")
        p = registry.get_or_default("nonexistent")
        assert p.name == "general"

    def test_get_or_default_no_general_returns_constructed_default(self, tmp_path: Path) -> None:
        registry, _ = self._make_registry(tmp_path)
        p = registry.get_or_default("nonexistent")
        assert p.name == "general"
        assert isinstance(p, DomainProfile)

    def test_register_programmatic(self, tmp_path: Path) -> None:
        registry, _ = self._make_registry(tmp_path)
        p = DomainProfile(name="custom", display_name="Custom", description="Custom.")
        registry.register(p)
        assert "custom" in registry.list_profiles()
        assert registry.get("custom").display_name == "Custom"

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        registry, d = self._make_registry(tmp_path)
        self._write_profile(d, "finance")
        p = registry.load_from_yaml(d / "finance.yaml")
        assert p.name == "finance"
        assert registry.get("finance").name == "finance"

    def test_missing_profiles_dir_does_not_crash(self, tmp_path: Path) -> None:
        registry = ProfileRegistry(profiles_dir=tmp_path / "nonexistent")
        assert registry.list_profiles() == []

    def test_corrupted_yaml_is_skipped(self, tmp_path: Path) -> None:
        registry, d = self._make_registry(tmp_path)
        self._write_profile(d, "good")
        (d / "bad.yaml").write_text("{not: valid: yaml: ]]]")
        assert "good" in registry.list_profiles()

    def test_bundled_profile_yamls_load(self) -> None:
        """Regression: all bundled configs/profiles/*.yaml files must parse
        correctly with model_validate() (tests the from_yaml fix)."""
        bundled_dir = Path(__file__).parent.parent.parent / "configs" / "profiles"
        if not bundled_dir.exists():
            pytest.skip("configs/profiles not found")
        registry = ProfileRegistry(profiles_dir=bundled_dir)
        profiles = registry.list_profiles()
        assert len(profiles) >= 3
        for name in profiles:
            p = registry.get(name)
            assert isinstance(p.thresholds, ThresholdConfig)
            assert isinstance(p.constraints, VerificationConstraints)


class TestModuleLevelConvenience:
    def test_get_registry_returns_registry_instance(self) -> None:
        from veritascore.profiles.registry import get_registry
        r = get_registry()
        assert isinstance(r, ProfileRegistry)

    def test_get_profile_or_default_module_level(self, tmp_path: Path) -> None:
        import veritascore.profiles.registry as reg_module
        from veritascore.profiles.registry import get_profile_or_default

        # Swap in a registry pointed at tmp_path so we don't pollute state
        old = reg_module._default_registry
        reg_module._default_registry = ProfileRegistry(profiles_dir=tmp_path / "empty")
        try:
            p = get_profile_or_default("nonexistent")
            assert isinstance(p, DomainProfile)
        finally:
            reg_module._default_registry = old



def test_get_or_default_when_name_exists(tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "medical.yaml").write_text(
        "name: medical\ndisplay_name: Medical\ndescription: Medical.\n"
    )
    registry = ProfileRegistry(profiles_dir=profiles_dir)
    p = registry.get_or_default("medical")
    assert p.name == "medical"


def test_get_profile_module_level(tmp_path: Path) -> None:
    import veritascore.profiles.registry as reg_module
    old = reg_module._default_registry
    (tmp_path / "empty").mkdir()
    reg_module._default_registry = ProfileRegistry(profiles_dir=tmp_path / "empty")
    p = DomainProfile(name="x", display_name="X", description="X")
    reg_module._default_registry.register(p)
    from veritascore.profiles.registry import get_profile
    result = get_profile("x")
    assert result.name == "x"
    reg_module._default_registry = old
