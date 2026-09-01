from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from .config import RuntimeSettings


@dataclass(frozen=True, slots=True)
class GenerationResult:
    model: str
    data: dict[str, Any]


def _exception_chain(exc: Exception) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in chain and len(chain) < 8:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _error_code(exc: Exception) -> int | None:
    for current in _exception_chain(exc):
        for attribute in ("code", "status_code"):
            value = getattr(current, attribute, None)
            if callable(value):
                value = value()
            value = getattr(value, "value", value)
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        match = re.search(
            r"(?:\b|['\"]code['\"]\s*:\s*)(401|403|404|408|429|500|502|503|504)\b",
            str(current),
        )
        if match:
            return int(match.group(1))
    return None


def _is_transport_error(exc: Exception) -> bool:
    return any(
        isinstance(current, (TimeoutError, ConnectionError))
        or type(current).__name__ in {"TransportError", "ConnectError", "ReadTimeout"}
        for current in _exception_chain(exc)
    )


def is_model_access_error(exc: Exception) -> bool:
    if _is_transport_error(exc):
        return True
    return _error_code(exc) in {401, 403, 404, 408, 429, 500, 502, 503, 504}


def is_retryable_error(exc: Exception) -> bool:
    if _is_transport_error(exc):
        return True
    return _error_code(exc) in {408, 429, 500, 502, 503, 504}


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        delay = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, delay)


class GeminiAdapter:
    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
        before_call: Callable[[], None] | None = None,
        allow_fallback: bool = True,
    ) -> None:
        self.settings = settings
        self.client = client or genai.Client(
            vertexai=True,
            project=settings.project_id,
            location=settings.location,
            http_options=types.HttpOptions(
                api_version="v1",
                timeout=settings.request_timeout_ms,
            ),
        )
        self.sleep = sleep
        self.jitter = jitter
        self.before_call = before_call
        self.allow_fallback = allow_fallback

    def generate_json(
        self,
        *,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> GenerationResult:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        return self._generate_json(
            contents=prompt,
            response_schema=response_schema,
        )

    def generate_multimodal_json(
        self,
        *,
        prompt: str,
        images: list[tuple[bytes, str]],
        response_schema: dict[str, Any],
    ) -> GenerationResult:
        """Generate schema-constrained JSON from text and in-memory images."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not images:
            return self.generate_json(
                prompt=prompt,
                response_schema=response_schema,
            )
        parts: list[Any] = [types.Part.from_text(text=prompt)]
        for image_bytes, mime_type in images:
            if not image_bytes:
                raise ValueError("image bytes must not be empty")
            if not mime_type.startswith("image/"):
                raise ValueError(f"unsupported image mime type: {mime_type}")
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
        return self._generate_json(
            contents=[types.Content(role="user", parts=parts)],
            response_schema=response_schema,
        )

    def _generate_json(
        self,
        *,
        contents: Any,
        response_schema: dict[str, Any],
    ) -> GenerationResult:
        last_access_error: Exception | None = None

        for primary_attempt in range(1, self.settings.primary_access_attempts + 1):
            try:
                return self._call(
                    model=self.settings.primary_model,
                    contents=contents,
                    response_schema=response_schema,
                )
            except Exception as exc:
                if not is_model_access_error(exc):
                    raise
                last_access_error = exc
                if not is_retryable_error(exc):
                    break
                if primary_attempt < self.settings.primary_access_attempts:
                    retry_after = _retry_after_seconds(exc)
                    delay = (
                        retry_after
                        if retry_after is not None
                        else min(2 ** (primary_attempt - 1), 8) + self.jitter(0.0, 0.25)
                    )
                    self.sleep(delay)

        if not self.allow_fallback:
            assert last_access_error is not None
            raise last_access_error

        try:
            return self._call(
                model=self.settings.fallback_model,
                contents=contents,
                response_schema=response_schema,
            )
        except Exception as fallback_error:
            if last_access_error is not None:
                fallback_error.add_note(
                    "Primary model failed "
                    f"{self.settings.primary_access_attempts} access attempts: "
                    f"{type(last_access_error).__name__}"
                )
            raise

    def _call(
        self,
        *,
        model: str,
        contents: Any,
        response_schema: dict[str, Any],
    ) -> GenerationResult:
        if self.before_call is not None:
            self.before_call()
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=response_schema,
                max_output_tokens=self.settings.max_output_tokens,
                thinking_config=types.ThinkingConfig(thinking_level=self.settings.thinking_level),
            ),
        )
        data = self._extract_json(response)
        return GenerationResult(
            model=model,
            data=data,
        )

    @staticmethod
    def _extract_json(response: Any) -> dict[str, Any]:
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, dict):
            return parsed
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Gemini response did not contain JSON text")
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("Gemini response JSON must be an object")
        return value
