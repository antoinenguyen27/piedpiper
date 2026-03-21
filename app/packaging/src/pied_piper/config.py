from __future__ import annotations

import os

import httpx

from .exceptions import ConfigurationError

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, write=60.0, read=300.0, pool=10.0)


def resolve_base_url(base_url: str | None) -> str:
    value = base_url or os.environ.get("PIED_PIPER_BASE_URL")
    if not value:
        raise ConfigurationError(
            "Missing base URL. Set PIED_PIPER_BASE_URL or pass base_url explicitly."
        )
    return value.rstrip("/")


def resolve_api_key(api_key: str | None) -> str:
    value = api_key or os.environ.get("PIED_PIPER_API_KEY")
    if not value:
        raise ConfigurationError(
            "Missing API key. Set PIED_PIPER_API_KEY or pass api_key explicitly."
        )
    return value


def resolve_timeout(timeout: float | httpx.Timeout | None) -> httpx.Timeout:
    if timeout is None:
        return DEFAULT_TIMEOUT
    if isinstance(timeout, httpx.Timeout):
        return timeout
    return httpx.Timeout(timeout)

