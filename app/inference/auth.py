from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, status


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("PIED_PIPER_API_KEY")
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server authentication is not configured.",
        )

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

