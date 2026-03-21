from __future__ import annotations

from .client import Client
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    PiedPiperError,
    RequestError,
    ServerError,
)
from .models import CompressionItemResult, CompressionResult

_default_client: Client | None = None


def _get_default_client() -> Client:
    global _default_client
    if _default_client is None:
        _default_client = Client()
    return _default_client


def compress(input):
    return _get_default_client().compress(input)


__all__ = [
    "AuthenticationError",
    "Client",
    "CompressionItemResult",
    "CompressionResult",
    "ConfigurationError",
    "PiedPiperError",
    "RequestError",
    "ServerError",
    "compress",
]

