from __future__ import annotations

import json
from contextlib import ExitStack
from typing import Any

import httpx

from .config import resolve_api_key, resolve_base_url, resolve_timeout
from .exceptions import AuthenticationError, RequestError, ServerError
from .inputs import normalize_input
from .models import CompressionResult, compression_result_from_dict


class Client:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | httpx.Timeout | None = None,
        *,
        http_client: httpx.Client | None = None,
    ):
        self.base_url = resolve_base_url(base_url)
        self.api_key = resolve_api_key(api_key)
        self.timeout = resolve_timeout(timeout)
        self._client = http_client or httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def compress(self, input_value: Any, *, fidelity: float = 0.33) -> CompressionResult:
        normalized = normalize_input(input_value, fidelity=fidelity)
        headers = {"Authorization": f"Bearer {self.api_key}"}

        with ExitStack() as stack:
            files = []
            for upload in normalized.uploads:
                file_handle = stack.enter_context(upload.path.open("rb"))
                files.append(
                    (
                        upload.field_name,
                        (upload.path.name, file_handle, upload.content_type),
                    )
                )

            try:
                response = self._client.post(
                    "/v1/compress",
                    headers=headers,
                    data={"manifest": json.dumps(normalized.manifest)},
                    files=files,
                )
            except httpx.HTTPError as exc:
                raise RequestError(f"Request failed: {exc}") from exc

        if response.status_code == 401:
            raise AuthenticationError("Authentication failed.")
        if response.status_code >= 500:
            raise ServerError(f"Server error: {response.status_code} {response.text}")
        if response.status_code >= 400:
            raise RequestError(f"Request failed: {response.status_code} {response.text}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise RequestError("Server returned invalid JSON.") from exc

        try:
            return compression_result_from_dict(payload)
        except Exception as exc:
            raise RequestError(f"Malformed response payload: {exc}") from exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
