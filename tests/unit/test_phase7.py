"""Unit tests for Phase 7 — engine, mode router, and API models.
All ML calls are mocked; no models are loaded.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from veritascore.api.models import (
    HealthResponse,
    ProfilesResponse,
    VerifyRequest,
    VerifyResponse,
    report_to_response,
)
from veritascore.core.config import EngineConfig, SearchConfig
from veritascore.core.types import (
    Claim,
    ClaimVerdict,
    Verdict,
    VerificationMode,
    VerificationReport,
)
from veritascore.router.mode_router import ModeRouter

# ── Shared fixtures ───────────────────────────────────────────────────────────

def make_claim(cid: str = "c1", text: str = "Paris is the capital of France.") -> Claim:
    return Claim(id=cid, text=text, source_span=(0, len(text)), source_text=text)


def make_verdict(
    cid: str = "c1",
    verdict: Verdict = Verdict.SUPPORTED,
    nli_score: float = 0.85,
) -> ClaimVerdict:
    return ClaimVerdict(
        claim=make_claim(cid),
        verdict=verdict,
        confidence=0.85,
        nli_score=nli_score,
        reason="Test reason.",
        verification_mode=VerificationMode.GROUNDED,
    )


def make_report(verdicts: list[ClaimVerdict] | None = None) -> VerificationReport:
    v = verdicts or [make_verdict()]
    return VerificationReport(
        query="What is the capital of France?",
        response_text="Paris is the capital of France.",
        claims=v,
        overall_trust_score=0.85,
        overall_verdict=Verdict.SUPPORTED,
        verification_mode=VerificationMode.GROUNDED,
        domain_profile="general",
        processing_time_ms=120.0,
        metadata={"num_claims": len(v)},
    )


# ── ModeRouter ────────────────────────────────────────────────────────────────

class TestModeRouter:
    def test_explicit_grounded_mode(self) -> None:
        router = ModeRouter()
        assert router.resolve("grounded") == VerificationMode.GROUNDED

    def test_explicit_ungrounded_mode(self) -> None:
        router = ModeRouter()
        assert router.resolve("ungrounded") == VerificationMode.UNGROUNDED

    def test_explicit_offline_mode(self) -> None:
        router = ModeRouter()
        assert router.resolve("offline") == VerificationMode.OFFLINE

    def test_invalid_mode_raises(self) -> None:
        router = ModeRouter()
        with pytest.raises(ValueError):
            router.resolve("nonsense")

    def test_auto_with_context_resolves_grounded(self) -> None:
        router = ModeRouter()
        mode = router.resolve("auto", has_context=True)
        assert mode == VerificationMode.GROUNDED

    def test_auto_no_context_no_internet_resolves_offline(self) -> None:
        router = ModeRouter()
        with patch.object(router, "_has_internet", return_value=False):
            mode = router.resolve("auto", has_context=False)
        assert mode == VerificationMode.OFFLINE

    def test_auto_no_context_internet_no_key_resolves_offline(self) -> None:
        config = EngineConfig(search=SearchConfig())  # no API keys
        router = ModeRouter(config=config)
        with patch.object(router, "_has_internet", return_value=True):
            mode = router.resolve("auto", has_context=False)
        assert mode == VerificationMode.OFFLINE

    def test_auto_no_context_internet_with_key_resolves_ungrounded(self, tmp_path: Path) -> None:
        config = EngineConfig(search=SearchConfig(tavily_api_key="fake-key"))
        router = ModeRouter(config=config)
        with patch.object(router, "_has_internet", return_value=True):
            mode = router.resolve("auto", has_context=False)
        assert mode == VerificationMode.UNGROUNDED

    def test_has_search_api_with_tavily_key(self) -> None:
        config = EngineConfig(search=SearchConfig(tavily_api_key="tvly-abc"))
        router = ModeRouter(config=config)
        assert router._has_search_api() is True

    def test_has_search_api_with_brave_key(self) -> None:
        config = EngineConfig(search=SearchConfig(brave_api_key="BSA-abc"))
        router = ModeRouter(config=config)
        assert router._has_search_api() is True

    def test_has_search_api_no_keys(self) -> None:
        config = EngineConfig(search=SearchConfig())
        router = ModeRouter(config=config)
        assert router._has_search_api() is False

    def test_has_internet_connection_error_returns_false(self) -> None:
        router = ModeRouter()
        with patch("socket.create_connection", side_effect=OSError("refused")):
            assert router._has_internet() is False


# ── API Models ────────────────────────────────────────────────────────────────

class TestVerifyRequest:
    def test_defaults(self) -> None:
        req = VerifyRequest(response="Some text.")
        assert req.mode == "auto"
        assert req.domain == "general"
        assert req.query is None
        assert req.context is None

    def test_empty_response_raises(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            VerifyRequest(response="")

    def test_all_fields(self) -> None:
        req = VerifyRequest(
            response="Text.", query="Query?", context="Context.",
            mode="grounded", domain="medical",
        )
        assert req.mode == "grounded"
        assert req.domain == "medical"


class TestReportToResponse:
    def test_basic_conversion(self) -> None:
        report = make_report()
        resp = report_to_response(report)
        assert isinstance(resp, VerifyResponse)
        assert resp.overall_verdict == "supported"
        assert resp.n_claims == 1
        assert resp.n_supported == 1
        assert len(resp.claims) == 1

    def test_claim_fields_mapped(self) -> None:
        report = make_report()
        resp = report_to_response(report)
        claim_resp = resp.claims[0]
        assert claim_resp.claim_id == "c1"
        assert claim_resp.verdict == "supported"
        assert claim_resp.nli_score == pytest.approx(0.85)

    def test_multiple_verdicts(self) -> None:
        verdicts = [
            make_verdict("c1", Verdict.SUPPORTED),
            make_verdict("c2", Verdict.CONTRADICTED),
            make_verdict("c3", Verdict.UNSUPPORTED),
        ]
        report = make_report(verdicts)
        resp = report_to_response(report)
        assert resp.n_claims == 3

    def test_metadata_preserved(self) -> None:
        report = make_report()
        resp = report_to_response(report)
        assert "num_claims" in resp.metadata

    def test_response_serializable(self) -> None:
        report = make_report()
        resp = report_to_response(report)
        data = resp.model_dump()
        json.dumps(data)  # Must not raise


class TestOtherModels:
    def test_health_response(self) -> None:
        h = HealthResponse(status="ok", version="0.1.0", mode="ready")
        assert h.status == "ok"

    def test_profiles_response(self) -> None:
        p = ProfilesResponse(profiles=["general", "medical"])
        assert "medical" in p.profiles


# ── VeritasCoreEngine (heavily mocked) ────────────────────────────────────────

class TestVeritasCoreEngineAggregation:
    """Test the pure-logic methods of the engine without loading any models."""

    def _make_engine(self) -> Any:
        from veritascore.core.engine import VeritasCoreEngine
        return VeritasCoreEngine()

    def test_aggregate_verdict_all_supported(self) -> None:
        engine = self._make_engine()
        verdicts = [make_verdict("c1", Verdict.SUPPORTED), make_verdict("c2", Verdict.SUPPORTED)]
        assert engine._aggregate_verdict(verdicts) == Verdict.SUPPORTED

    def test_aggregate_verdict_any_contradicted(self) -> None:
        engine = self._make_engine()
        verdicts = [make_verdict("c1", Verdict.SUPPORTED), make_verdict("c2", Verdict.CONTRADICTED)]
        assert engine._aggregate_verdict(verdicts) == Verdict.CONTRADICTED

    def test_aggregate_verdict_all_unsupported(self) -> None:
        engine = self._make_engine()
        verdicts = [make_verdict("c1", Verdict.UNSUPPORTED), make_verdict("c2", Verdict.UNSUPPORTED)]
        assert engine._aggregate_verdict(verdicts) == Verdict.UNSUPPORTED

    def test_aggregate_verdict_mixed_supported_unsupported(self) -> None:
        engine = self._make_engine()
        verdicts = [make_verdict("c1", Verdict.SUPPORTED), make_verdict("c2", Verdict.UNSUPPORTED)]
        assert engine._aggregate_verdict(verdicts) == Verdict.UNSUPPORTED

    def test_aggregate_verdict_empty_list(self) -> None:
        engine = self._make_engine()
        assert engine._aggregate_verdict([]) == Verdict.UNSUPPORTED

    def test_empty_report_returns_unsupported(self) -> None:
        import time
        engine = self._make_engine()
        report = engine._empty_report("text", "query", VerificationMode.OFFLINE, "general", time.perf_counter())
        assert report.overall_verdict == Verdict.UNSUPPORTED
        assert report.n_claims == 0
        assert report.overall_trust_score == pytest.approx(0.5)


class TestVeritasCoreEngineVerify:
    """Test the full verify() method with all ML components mocked."""

    def test_verify_grounded_mode(self) -> None:
        from veritascore.core.engine import VeritasCoreEngine

        engine = VeritasCoreEngine()

        # Mock all ML components
        mock_verdict = make_verdict("c1", Verdict.SUPPORTED)
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = [make_claim()]

        mock_nli = MagicMock()
        mock_nli.verify.return_value = [mock_verdict]

        mock_scorer = MagicMock()
        mock_scorer.score_claim.return_value = 0.87
        mock_scorer.score_response.return_value = 0.87

        from veritascore.verifier.consistency import ConsistencyResult
        mock_checker = MagicMock()
        mock_checker.check_consistency.return_value = ConsistencyResult(
            claim_scores={"c1": 0.88}, coherence_score=0.9, response_relevance=0.9, off_topic_claims=[]
        )
        engine._decomposer = mock_decomposer
        engine._nli_verifier = mock_nli
        engine._scorer = mock_scorer
        engine._consistency_checker = mock_checker

        report = engine.verify(
            response="Paris is the capital of France.",
            query="What is the capital of France?",
            context="Paris is the capital and largest city of France.",
        )

        assert report.overall_trust_score == pytest.approx(0.87)
        assert report.verification_mode == VerificationMode.GROUNDED
        assert report.n_claims == 1

    def test_verify_empty_response_no_decomposition(self) -> None:
        from veritascore.core.engine import VeritasCoreEngine

        engine = VeritasCoreEngine()

        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = []  # empty decomposition
        engine._decomposer = mock_decomposer

        report = engine.verify(response="OK.", context="Some context.")
        assert report.n_claims == 0
        assert report.overall_trust_score == pytest.approx(0.5)

    def test_verify_uses_domain_profile_thresholds(self) -> None:
        """G7 criterion 4: domain profile is actually applied to the verifier."""
        from veritascore.core.engine import VeritasCoreEngine

        engine = VeritasCoreEngine()
        applied_thresholds: dict[str, float] = {}

        mock_verdict = make_verdict()
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = [make_claim()]
        mock_nli = MagicMock()

        def capture_apply(verifier: Any) -> None:
            applied_thresholds["entailment"] = verifier.entailment_threshold

        mock_nli.verify.return_value = [mock_verdict]
        mock_nli.entailment_threshold = 0.7
        mock_nli.contradiction_threshold = 0.5
        engine._decomposer = mock_decomposer
        engine._nli_verifier = mock_nli

        mock_scorer = MagicMock()
        mock_scorer.score_claim.return_value = 0.8
        mock_scorer.score_response.return_value = 0.8
        engine._scorer = mock_scorer

        # Register a custom medical profile so the registry can find it
        from veritascore.profiles import DomainProfile, ThresholdConfig
        from veritascore.profiles.registry import ProfileRegistry
        medical = DomainProfile(
            name="medical", display_name="Medical", description="Test",
            thresholds=ThresholdConfig(entailment_threshold=0.85),
        )
        registry = ProfileRegistry()
        registry.register(medical)
        engine._profile_registry = registry

        engine.verify(
            response="Some claim.", context="Some context.", domain="medical"
        )

        # After verification, the NLI verifier should have received the medical threshold
        assert mock_nli.entailment_threshold == pytest.approx(0.85)

    def test_verify_consistency_uses_claim_scores_attribute(self) -> None:
        """Bug fix regression: _apply_consistency_scores accesses .claim_scores
        as an attribute (ConsistencyResult), not via dict subscript."""
        from veritascore.core.engine import VeritasCoreEngine
        from veritascore.verifier.consistency import ConsistencyResult

        engine = VeritasCoreEngine()

        claim = make_claim("c1")
        mock_verdict = make_verdict("c1", Verdict.SUPPORTED)
        mock_decomposer = MagicMock()
        mock_decomposer.decompose.return_value = [claim]
        mock_nli = MagicMock()
        mock_nli.verify.return_value = [mock_verdict]
        mock_scorer = MagicMock()
        mock_scorer.score_claim.return_value = 0.8
        mock_scorer.score_response.return_value = 0.8

        # consistency checker returns a ConsistencyResult (not a dict)
        consistency_result = ConsistencyResult(
            claim_scores={"c1": 0.92},
            coherence_score=0.8,
            response_relevance=0.85,
            off_topic_claims=[],
        )
        mock_checker = MagicMock()
        mock_checker.check_consistency.return_value = consistency_result

        engine._decomposer = mock_decomposer
        engine._nli_verifier = mock_nli
        engine._scorer = mock_scorer
        engine._consistency_checker = mock_checker

        # Should NOT raise AttributeError or TypeError accessing .claim_scores
        report = engine.verify(
            response="Paris is the capital of France.",
            query="What is the capital?",
            context="Paris is in France.",
        )
        assert report.claims[0].consistency_score == pytest.approx(0.92)

    def test_unload_clears_components(self) -> None:
        from veritascore.core.engine import VeritasCoreEngine

        engine = VeritasCoreEngine()
        engine._nli_verifier = MagicMock()
        engine._nli_verifier.unload = MagicMock()
        engine._consistency_checker = MagicMock()
        engine._consistency_checker.unload = MagicMock()
        engine._scorer = MagicMock()

        engine.unload()

        assert engine._nli_verifier is None
        assert engine._consistency_checker is None
        assert engine._scorer is None


# ── FastAPI App (mocked engine) ────────────────────────────────────────────────

class TestFastAPIApp:
    """Test the FastAPI endpoints using create_app with a mocked engine injected
    directly via the _engine module global, bypassing lifespan."""

    @pytest.fixture
    def client(self) -> Any:
        import importlib

        from fastapi.testclient import TestClient

        # importlib avoids the name collision where
        # `import veritascore.api.app as m` returns the FastAPI *object*
        # (because veritascore.api.__init__ also exports `app`).
        app_module = importlib.import_module("veritascore.api.app")

        mock_report = make_report()
        mock_engine = MagicMock()
        mock_engine.verify.return_value = mock_report

        original_engine = app_module._engine
        original_registry = app_module._profile_registry
        mock_registry = MagicMock()
        mock_registry.list_profiles.return_value = ["general", "medical"]
        app_module._engine = mock_engine
        app_module._profile_registry = mock_registry

        test_app = app_module.create_app()
        with TestClient(test_app, raise_server_exceptions=False) as c:
            yield c

        app_module._engine = original_engine
        app_module._profile_registry = original_registry

    def test_health_endpoint(self, client: Any) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_profiles_endpoint(self, client: Any) -> None:
        resp = client.get("/profiles")
        assert resp.status_code == 200
        data = resp.json()
        assert "profiles" in data
        assert isinstance(data["profiles"], list)

    def test_modes_endpoint(self, client: Any) -> None:
        resp = client.get("/modes")
        assert resp.status_code == 200
        data = resp.json()
        assert "auto" in data["modes"]
        assert "grounded" in data["modes"]

    def test_verify_post(self, client: Any) -> None:
        resp = client.post("/verify", json={
            "response": "Paris is the capital of France.",
            "query": "What is the capital of France?",
            "context": "Paris is the capital of France.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_trust_score" in data
        assert "claims" in data
        assert data["overall_verdict"] == "supported"

    def test_verify_post_empty_response_rejected(self, client: Any) -> None:
        resp = client.post("/verify", json={"response": ""})
        assert resp.status_code == 422

    def test_verify_stream_endpoint_returns_sse(self, client: Any) -> None:
        resp = client.get("/verify/stream", params={
            "response": "Paris is the capital of France.",
        })
        assert resp.status_code == 200
        content = resp.text
        assert "data:" in content

    def test_verify_stream_contains_complete_event(self, client: Any) -> None:
        resp = client.get("/verify/stream", params={
            "response": "The Eiffel Tower is in Paris.",
            "query": "Where is the Eiffel Tower?",
        })
        assert "complete" in resp.text

    def test_verify_stream_progress_events_for_each_claim(self, client: Any) -> None:
        resp = client.get("/verify/stream", params={
            "response": "Paris is in France.",
        })
        events = [
            json.loads(line[6:])
            for line in resp.text.strip().split("\n\n")
            if line.startswith("data: ")
        ]
        progress_events = [e for e in events if e.get("type") == "progress"]
        assert len(progress_events) == 1
        assert progress_events[0]["claim_id"] == "c1"
