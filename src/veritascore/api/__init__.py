"""VeritasCore REST API (Phase 7)."""

from veritascore.api.app import app, create_app
from veritascore.api.models import VerifyRequest, VerifyResponse

__all__ = ["app", "create_app", "VerifyRequest", "VerifyResponse"]
