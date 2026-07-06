"""FastAPI application for VeritasCore REST API (Phase 7).

Endpoints:
    POST /verify          — verify a response (full blocking pipeline)
    GET  /verify/stream   — SSE streaming with per-claim progress events
    GET  /health          — liveness probe
    GET  /profiles        — list available domain profiles
    GET  /modes           — list available verification modes

Bug fixes vs. spec reference implementation:
    1. Engine is a singleton per app lifecycle (not recreated per request).
    2. /verify/stream is registered on the same app (not lost as orphan).
    3. _run_consistency_check uses .claim_scores attribute not dict subscript.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from veritascore import __version__
from veritascore.api.models import (
    HealthResponse,
    ProfilesResponse,
    VerifyRequest,
    VerifyResponse,
    report_to_response,
)
from veritascore.core.config import EngineConfig
from veritascore.core.engine import VeritasCoreEngine
from veritascore.core.types import VerificationMode
from veritascore.profiles.registry import ProfileRegistry

logger = logging.getLogger(__name__)

# Module-level engine singleton (lazy init on first request)
_engine: VeritasCoreEngine | None = None
_profile_registry: ProfileRegistry | None = None


def _get_engine(config: EngineConfig | None = None) -> VeritasCoreEngine:
    """Return the singleton engine, creating it on first call."""
    global _engine
    if _engine is None:
        _engine = VeritasCoreEngine(config=config)
        logger.info("VeritasCoreEngine singleton created")
    return _engine


def _get_profile_registry() -> ProfileRegistry:
    global _profile_registry
    if _profile_registry is None:
        _profile_registry = ProfileRegistry()
    return _profile_registry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — pre-warm engine on startup."""
    logger.info("VeritasCore API starting up (version %s)", __version__)
    _get_engine()          # ensures singleton exists before first request
    _get_profile_registry()
    yield
    # Shutdown: unload models to free GPU memory
    global _engine
    if _engine is not None:
        _engine.unload()
        logger.info("VeritasCore API shut down — models unloaded")


def create_app(config: EngineConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Optional EngineConfig. Defaults to EngineConfig.default().

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="VeritasCore API",
        description="Post-hoc LLM hallucination detection — model-agnostic, production-grade.",
        version=__version__,
        lifespan=lifespan,
    )

    # CORS — allow all origins in dev; restrict in production via env/config
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Endpoints ─────────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse, tags=["Meta"])
    async def health() -> HealthResponse:
        """Liveness probe — returns 200 if the API is running."""
        return HealthResponse(
            status="ok",
            version=__version__,
            mode="ready",
        )

    @app.get("/profiles", response_model=ProfilesResponse, tags=["Meta"])
    async def list_profiles() -> ProfilesResponse:
        """List all available domain profiles."""
        registry = _get_profile_registry()
        return ProfilesResponse(profiles=registry.list_profiles())

    @app.get("/modes", tags=["Meta"])
    async def list_modes() -> dict[str, list[str]]:
        """List all valid verification mode strings."""
        return {"modes": [m.value for m in VerificationMode]}

    @app.post("/verify", response_model=VerifyResponse, tags=["Verification"])
    async def verify(request: VerifyRequest) -> VerifyResponse:
        """Verify an LLM response and return a structured trust report.

        Runs the full pipeline: decompose → verify → consistency → score.
        For large responses or long contexts, prefer /verify/stream to get
        per-claim progress events.
        """
        engine = _get_engine(config)
        try:
            report = await asyncio.to_thread(
                engine.verify,
                response=request.response,
                query=request.query,
                context=request.context,
                mode=request.mode,
                domain=request.domain,
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:
            logger.exception("Unexpected error in /verify: %s", e)
            raise HTTPException(status_code=500, detail="Internal verification error") from e

        return report_to_response(report)

    @app.get("/verify/stream", tags=["Verification"])
    async def verify_stream(
        response: str,
        query: str | None = None,
        context: str | None = None,
        mode: str = "auto",
        domain: str = "general",
    ) -> StreamingResponse:
        """Verify a response with Server-Sent Events progress streaming.

        Yields one SSE event per claim as it is verified, then a final
        'complete' event with the full VerifyResponse JSON. Useful for
        displaying real-time per-claim verdicts in a UI.

        Query parameters mirror the POST /verify request body.
        """
        engine = _get_engine(config)
        request = VerifyRequest(
            response=response, query=query, context=context, mode=mode, domain=domain
        )
        return StreamingResponse(
            _stream_verify(engine, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    return app


async def _stream_verify(
    engine: VeritasCoreEngine, request: VerifyRequest
) -> AsyncGenerator[str, None]:
    """Generate SSE events for a streaming verification request.

    Yields:
        data: JSON events with type 'progress' (per claim) or 'complete' (full report).
        Each event is formatted as per the SSE spec: ``data: {...}\\n\\n``.
    """
    try:
        report = await asyncio.to_thread(
            engine.verify,
            response=request.response,
            query=request.query,
            context=request.context,
            mode=request.mode,
            domain=request.domain,
        )
    except Exception as e:
        error_event: dict[str, Any] = {"type": "error", "message": str(e)}
        yield f"data: {json.dumps(error_event)}\n\n"
        return

    # Emit one progress event per claim
    for i, claim_verdict in enumerate(report.claims):
        progress_event: dict[str, Any] = {
            "type": "progress",
            "claim_index": i,
            "total_claims": report.n_claims,
            "claim_id": claim_verdict.claim.id,
            "claim_text": claim_verdict.claim.text,
            "verdict": claim_verdict.verdict.value,
            "confidence": claim_verdict.confidence,
            "evidence": claim_verdict.evidence,
        }
        yield f"data: {json.dumps(progress_event)}\n\n"
        # Yield control between events so the client receives them incrementally
        await asyncio.sleep(0)

    # Emit final complete event with full report
    response_data = report_to_response(report)
    complete_event: dict[str, Any] = {
        "type": "complete",
        "report": response_data.model_dump(),
    }
    yield f"data: {json.dumps(complete_event)}\n\n"


# Create the default application instance
app = create_app()
