from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from .config import RuntimeSettings
from .pricing import TokenUsage
from .usage import UsageLedger


@dataclass(frozen=True, slots=True)
class GenerationResult:
    request_id: str
    model: str
    data: dict[str, Any]
    usage: TokenUsage
    duration_ms: int
    attempts: int
    finish_reason: str | None


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _error_code(exc: Exception) -> int | None:
    for attribute in ("code", "status_code"):
        value = getattr(exc, attribute, None)
        if callable(value):
            value = value()
        value = getattr(value, "value", value)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def is_model_access_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return _error_code(exc) in {401, 403, 404, 408, 429, 500, 502, 503, 504}


class GeminiAdapter:
    def __init__(
        self,
        settings: RuntimeSettings,
        ledger: UsageLedger,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.client = client or genai.Client(
            vertexai=True,
            project=settings.project_id,
            location=settings.location,
            http_options=types.HttpOptions(api_version="v1"),
        )
        self.sleep = sleep

    def generate_json(
        self,
        *,
        prompt: str,
        response_schema: dict[str, Any],
        request_id: str | None = None,
    ) -> GenerationResult:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        return self._generate_json(
            contents=prompt,
            projected_input_bytes=len(prompt.encode("utf-8")),
            response_schema=response_schema,
            request_id=request_id,
        )

    def generate_multimodal_json(
        self,
        *,
        prompt: str,
        images: list[tuple[bytes, str]],
        response_schema: dict[str, Any],
        request_id: str | None = None,
    ) -> GenerationResult:
        """Generate schema-constrained JSON from text and in-memory images."""
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if not images:
            return self.generate_json(
                prompt=prompt,
                response_schema=response_schema,
                request_id=request_id,
            )
        parts: list[Any] = [types.Part.from_text(text=prompt)]
        projected_input_bytes = len(prompt.encode("utf-8"))
        for image_bytes, mime_type in images:
            if not image_bytes:
                raise ValueError("image bytes must not be empty")
            if not mime_type.startswith("image/"):
                raise ValueError(f"unsupported image mime type: {mime_type}")
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
            projected_input_bytes += len(image_bytes)
        return self._generate_json(
            contents=[types.Content(role="user", parts=parts)],
            projected_input_bytes=projected_input_bytes,
            response_schema=response_schema,
            request_id=request_id,
        )

    def _generate_json(
        self,
        *,
        contents: Any,
        projected_input_bytes: int,
        response_schema: dict[str, Any],
        request_id: str | None,
    ) -> GenerationResult:

        request_id = request_id or str(uuid.uuid4())
        attempt = 0
        last_access_error: Exception | None = None

        for primary_attempt in range(1, self.settings.primary_access_attempts + 1):
            attempt += 1
            try:
                return self._call(
                    model=self.settings.primary_model,
                    contents=contents,
                    projected_input_bytes=projected_input_bytes,
                    response_schema=response_schema,
                    request_id=request_id,
                    attempt=attempt,
                )
            except Exception as exc:
                if not is_model_access_error(exc):
                    raise
                last_access_error = exc
                if primary_attempt < self.settings.primary_access_attempts:
                    self.sleep(min(2 ** (primary_attempt - 1), 8))

        attempt += 1
        try:
            return self._call(
                model=self.settings.fallback_model,
                contents=contents,
                projected_input_bytes=projected_input_bytes,
                response_schema=response_schema,
                request_id=request_id,
                attempt=attempt,
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
        projected_input_bytes: int,
        response_schema: dict[str, Any],
        request_id: str,
        attempt: int,
    ) -> GenerationResult:
        # UTF-8 byte length deliberately overestimates most text token counts,
        # including Korean prompts, for the preflight budget guard.
        projected_prompt_tokens = max(1, projected_input_bytes)
        self.ledger.assert_can_spend(
            model,
            TokenUsage(
                prompt_tokens=projected_prompt_tokens,
                output_tokens=self.settings.max_output_tokens,
            ),
        )

        started = time.perf_counter()
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=response_schema,
                    max_output_tokens=self.settings.max_output_tokens,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=self.settings.thinking_level
                    ),
                ),
            )
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000)
            self.ledger.append(
                self.ledger.build_record(
                    request_id=request_id,
                    attempt=attempt,
                    model=model,
                    status="error",
                    duration_ms=duration_ms,
                    usage=TokenUsage(),
                    error_type=type(exc).__name__,
                )
            )
            raise

        duration_ms = round((time.perf_counter() - started) * 1000)
        usage = self._extract_usage(response)
        finish_reason = self._extract_finish_reason(response)
        data = self._extract_json(response)
        self.ledger.append(
            self.ledger.build_record(
                request_id=request_id,
                attempt=attempt,
                model=model,
                status="success",
                duration_ms=duration_ms,
                usage=usage,
                finish_reason=finish_reason,
            )
        )
        return GenerationResult(
            request_id=request_id,
            model=model,
            data=data,
            usage=usage,
            duration_ms=duration_ms,
            attempts=attempt,
            finish_reason=finish_reason,
        )

    @staticmethod
    def _extract_usage(response: Any) -> TokenUsage:
        metadata = getattr(response, "usage_metadata", None)
        return TokenUsage(
            prompt_tokens=int(getattr(metadata, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(metadata, "candidates_token_count", 0) or 0),
            thinking_tokens=int(getattr(metadata, "thoughts_token_count", 0) or 0),
            cached_tokens=int(getattr(metadata, "cached_content_token_count", 0) or 0),
        )

    @staticmethod
    def _extract_finish_reason(response: Any) -> str | None:
        candidates = getattr(response, "candidates", None) or []
        return _enum_value(getattr(candidates[0], "finish_reason", None)) if candidates else None

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
