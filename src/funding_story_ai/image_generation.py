from __future__ import annotations

import os
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from google.genai import types


@dataclass(frozen=True, slots=True)
class ImageSettings:
    """Bounded Nano Banana execution policy for the MVP."""

    primary_model: str = "gemini-3.1-flash-image"
    fallback_model: str = "gemini-3.1-flash-lite-image"
    image_size: str = "1K"
    aspect_ratio: str = "3:2"
    output_mime_type: str = "image/jpeg"
    output_compression: int = 85
    primary_attempts: int = 2
    fallback_attempts: int = 1
    request_interval_seconds: float = 3.0
    retry_base_seconds: float = 10.0
    retry_max_seconds: float = 30.0
    retry_jitter_seconds: float = 2.0
    fallback_delay_seconds: float = 10.0

    @classmethod
    def from_env(cls) -> ImageSettings:
        primary_attempts = int(os.getenv("IMAGE_PRIMARY_ATTEMPTS", "2"))
        fallback_attempts = int(os.getenv("IMAGE_FALLBACK_ATTEMPTS", "1"))
        if not 1 <= primary_attempts <= 2:
            raise ValueError("IMAGE_PRIMARY_ATTEMPTS must be between 1 and 2")
        if fallback_attempts != 1:
            raise ValueError("IMAGE_FALLBACK_ATTEMPTS must be 1")

        def non_negative_float(name: str, default: str) -> float:
            value = float(os.getenv(name, default))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            return value

        return cls(
            primary_model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image"),
            fallback_model=os.getenv("GEMINI_IMAGE_FALLBACK_MODEL", "gemini-3.1-flash-lite-image"),
            image_size=os.getenv("GEMINI_IMAGE_SIZE", "1K"),
            aspect_ratio=os.getenv("GEMINI_IMAGE_ASPECT_RATIO", "3:2"),
            output_mime_type=os.getenv("GEMINI_IMAGE_MIME_TYPE", "image/jpeg"),
            output_compression=int(os.getenv("GEMINI_IMAGE_COMPRESSION", "85")),
            primary_attempts=primary_attempts,
            fallback_attempts=fallback_attempts,
            request_interval_seconds=non_negative_float("IMAGE_REQUEST_INTERVAL_SECONDS", "3"),
            retry_base_seconds=non_negative_float("IMAGE_RETRY_BASE_SECONDS", "10"),
            retry_max_seconds=non_negative_float("IMAGE_RETRY_MAX_SECONDS", "30"),
            retry_jitter_seconds=non_negative_float("IMAGE_RETRY_JITTER_SECONDS", "2"),
            fallback_delay_seconds=non_negative_float("IMAGE_FALLBACK_DELAY_SECONDS", "10"),
        )


@dataclass(frozen=True, slots=True)
class ImageResult:
    slot_id: str
    image_bytes: bytes
    model: str
    mime_type: str = "image/jpeg"
    provider: str = "google"
    attempts: int = 1


class ImageAdapter(Protocol):
    def generate(
        self,
        *,
        slot_id: str,
        prompt: str,
        reference_paths: Sequence[Path] = (),
    ) -> ImageResult: ...


class ImageGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        category: str = "unknown",
        status_code: int | None = None,
        model: str | None = None,
        provider_message: str | None = None,
        attempt_history: Sequence[dict[str, Any]] = (),
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.category = category
        self.status_code = status_code
        self.model = model
        self.provider_message = provider_message
        self.attempt_history = tuple(dict(item) for item in attempt_history)


def _error_code(exc: Exception) -> int | None:
    for value in (getattr(exc, "status_code", None), getattr(exc, "code", None)):
        value = value() if callable(value) else value
        value = getattr(value, "value", value)
        if value is None:
            continue
        try:
            return int(str(value))
        except (TypeError, ValueError):
            pass
    match = re.search(r"\b(400|401|403|404|408|429|500|502|503|504)\b", str(exc))
    return int(match.group(1)) if match else None


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return _error_code(exc) in {408, 429, 500, 502, 503, 504}


def _terminal(exc: Exception) -> bool:
    message = str(exc).lower()
    return _error_code(exc) in {400, 401, 403} or any(
        marker in message for marker in ("safety", "policy violation", "blocked prompt")
    )


def _error_category(exc: Exception) -> str:
    message = str(exc).lower()
    code = _error_code(exc)
    if any(marker in message for marker in ("safety", "policy violation", "blocked prompt")):
        return "safety"
    if code == 429 or any(marker in message for marker in ("resource_exhausted", "quota")):
        return "rate_limit"
    if code == 408 or isinstance(exc, TimeoutError) or "deadline_exceeded" in message:
        return "timeout"
    if code in {500, 502, 503, 504} or "unavailable" in message:
        return "provider_unavailable"
    if isinstance(exc, ConnectionError):
        return "network"
    if code == 401:
        return "authentication"
    if code == 403:
        return "permission"
    if code == 400:
        return "invalid_request"
    if "did not contain image data" in message:
        return "empty_response"
    return "unknown"


def _safe_error_message(exc: Exception) -> str:
    message = " ".join(str(exc).split()) or type(exc).__name__
    message = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED_API_KEY]", message)
    message = re.sub(
        r"(?i)((?:api[_ -]?key|authorization|token)\s*[:=]\s*)\S+",
        r"\1[REDACTED]",
        message,
    )
    return message[:500]


def _retry_after_seconds(exc: Exception) -> float | None:
    for owner in (exc, getattr(exc, "response", None)):
        if owner is None:
            continue
        headers = getattr(owner, "headers", None)
        if headers is not None:
            try:
                value = headers.get("retry-after") or headers.get("Retry-After")
                if value is not None:
                    return max(0.0, float(value))
            except (AttributeError, TypeError, ValueError):
                pass
        for name in ("retry_after", "retry_delay"):
            value = getattr(owner, name, None)
            if value is None:
                continue
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                pass
    match = re.search(
        r"(?:retry(?:\s+after|\s+in|\s+delay)?)[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?)\s*s",
        str(exc),
        flags=re.IGNORECASE,
    )
    return max(0.0, float(match.group(1))) if match else None


def image_error_metadata(exc: Exception) -> dict[str, Any]:
    """Return persistence-safe diagnostics for one failed image slot."""

    if isinstance(exc, ImageGenerationError):
        return {
            "provider": "google",
            "model": exc.model,
            "attempts": exc.attempts,
            "error_type": type(exc).__name__,
            "error_category": exc.category,
            "error_code": exc.status_code,
            "error_message": exc.provider_message or str(exc),
            "attempt_history": [dict(item) for item in exc.attempt_history],
        }
    return {
        "provider": None,
        "model": None,
        "attempts": max(1, int(getattr(exc, "attempts", 1))),
        "error_type": type(exc).__name__,
        "error_category": _error_category(exc),
        "error_code": _error_code(exc),
        "error_message": _safe_error_message(exc),
        "attempt_history": [],
    }


def _mime_for_path(path: Path) -> str:
    return {
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(path.suffix.lower(), "image/jpeg")


class GeminiImageAdapter:
    """Generate one independent slot with primary retry and one Lite fallback."""

    def __init__(
        self,
        settings: ImageSettings,
        *,
        client: Any,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.settings = settings
        self.client = client
        self.sleep = sleep
        self.monotonic = monotonic
        self.jitter = jitter
        self._last_request_completed_at: float | None = None

    def _wait_for_request_interval(self) -> None:
        if self._last_request_completed_at is None:
            return
        elapsed = self.monotonic() - self._last_request_completed_at
        remaining = self.settings.request_interval_seconds - elapsed
        if remaining > 0:
            self.sleep(remaining)

    def _retry_delay(self, exc: Exception, *, attempts: int, fallback: bool) -> float:
        provider_delay = _retry_after_seconds(exc) or 0.0
        configured = (
            self.settings.fallback_delay_seconds
            if fallback
            else min(
                self.settings.retry_base_seconds * (2 ** max(0, attempts - 1)),
                self.settings.retry_max_seconds,
            )
        )
        jitter = self.settings.retry_jitter_seconds * max(0.0, min(1.0, self.jitter()))
        return min(max(provider_delay, configured) + jitter, self.settings.retry_max_seconds)

    def _spaced_call(self, *, slot_id: str, model: str, contents: Any) -> ImageResult:
        self._wait_for_request_interval()
        try:
            return self._call(slot_id=slot_id, model=model, contents=contents)
        finally:
            self._last_request_completed_at = self.monotonic()

    def generate(
        self,
        *,
        slot_id: str,
        prompt: str,
        reference_paths: Sequence[Path] = (),
    ) -> ImageResult:
        parts: list[Any] = [types.Part.from_text(text=prompt)]
        for path in reference_paths:
            parts.append(
                types.Part.from_bytes(data=path.read_bytes(), mime_type=_mime_for_path(path))
            )
        contents = [types.Content(role="user", parts=parts)]
        attempts = 0
        last_error: Exception | None = None
        last_model: str | None = None
        attempt_history: list[dict[str, Any]] = []
        policies = (
            (self.settings.primary_model, self.settings.primary_attempts),
            (self.settings.fallback_model, self.settings.fallback_attempts),
        )
        for model, limit in policies:
            for attempt in range(1, limit + 1):
                attempts += 1
                last_model = model
                try:
                    return replace(
                        self._spaced_call(slot_id=slot_id, model=model, contents=contents),
                        attempts=attempts,
                    )
                except Exception as exc:
                    last_error = exc
                    retryable = _retryable(exc)
                    terminal = _terminal(exc)
                    delay = 0.0
                    if retryable and not terminal:
                        if attempt < limit:
                            delay = self._retry_delay(exc, attempts=attempts, fallback=False)
                        elif model == self.settings.primary_model:
                            delay = self._retry_delay(exc, attempts=attempts, fallback=True)
                    attempt_history.append(
                        {
                            "model": model,
                            "model_attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error_category": _error_category(exc),
                            "error_code": _error_code(exc),
                            "retryable": retryable,
                            "delay_seconds": round(delay, 3),
                        }
                    )
                    if _terminal(exc):
                        raise ImageGenerationError(
                            f"Image request rejected without retry: {type(exc).__name__}",
                            attempts=attempts,
                            category=_error_category(exc),
                            status_code=_error_code(exc),
                            model=model,
                            provider_message=_safe_error_message(exc),
                            attempt_history=attempt_history,
                        ) from exc
                    if not retryable:
                        break
                    if delay:
                        self.sleep(delay)
        assert last_error is not None
        raise ImageGenerationError(
            "Nano Banana primary and Lite fallback failed; "
            f"last error: {type(last_error).__name__}",
            attempts=attempts,
            category=_error_category(last_error),
            status_code=_error_code(last_error),
            model=last_model,
            provider_message=_safe_error_message(last_error),
            attempt_history=attempt_history,
        ) from last_error

    def _call(self, *, slot_id: str, model: str, contents: Any) -> ImageResult:
        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=[types.Modality.IMAGE],
                image_config=types.ImageConfig(
                    aspect_ratio=self.settings.aspect_ratio,
                    image_size=self.settings.image_size,
                    output_mime_type=self.settings.output_mime_type,
                    output_compression_quality=self.settings.output_compression,
                    person_generation="ALLOW_NONE",
                ),
            ),
        )
        candidates = getattr(response, "candidates", None) or []
        parts = getattr(getattr(candidates[0], "content", None), "parts", []) if candidates else []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return ImageResult(
                    slot_id=slot_id,
                    image_bytes=bytes(inline.data),
                    model=model,
                    mime_type=getattr(inline, "mime_type", None) or self.settings.output_mime_type,
                )
        raise ValueError("Gemini image response did not contain image data")
