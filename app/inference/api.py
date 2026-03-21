from __future__ import annotations

import os
import importlib.util
from uuid import uuid4

from fastapi import Depends, FastAPI, Request

from .auth import require_api_key
from .router import parse_manifest_request, process_manifest
from .schemas import CompressionResponse, HealthResponse


def create_app() -> FastAPI:
    app = FastAPI(title="Pied Piper Inference", version="0.1.0")

    @app.get("/")
    async def root():
        return {"service": "pied-piper-inference", "version": "0.1.0"}

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        text_backend = (
            "llmlingua"
            if importlib.util.find_spec("llmlingua") is not None
            else "unavailable"
        )

        return HealthResponse(
            auth_configured=bool(os.environ.get("PIED_PIPER_API_KEY")),
            text_backend=text_backend,
        )

    @app.post("/v1/compress", response_model=CompressionResponse, dependencies=[Depends(require_api_key)])
    async def compress(request: Request) -> CompressionResponse:
        manifest, uploads = await parse_manifest_request(request)
        return await process_manifest(
            manifest,
            uploads,
            request_id=f"req_{uuid4().hex}",
        )

    return app
