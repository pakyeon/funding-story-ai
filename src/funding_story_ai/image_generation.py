from __future__ import annotations

import os
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

    @classmethod
    def from_env(cls) -> ImageSettings:
        primary_attempts = int(os.getenv("IMAGE_PRIMARY_ATTEMPTS", "2"))
        fallback_attempts = int(os.getenv("IMAGE_FALLBACK_ATTEMPTS", "1"))
        if not 1 <= primary_attempts <= 2:
            raise ValueError("IMAGE_PRIMARY_ATTEMPTS must be between 1 and 2")
        if fallback_attempts != 1:
            raise ValueError("IMAGE_FALLBACK_ATTEMPTS must be 1")
        return cls(
            primary_model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image"),
            fallback_model=os.getenv(
                "GEMINI_IMAGE_FALLBACK_MODEL", "gemini-3.1-flash-lite-image"
            ),
            image_size=os.getenv("GEMINI_IMAGE_SIZE", "1K"),
            aspect_ratio=os.getenv("GEMINI_IMAGE_ASPECT_RATIO", "3:2"),
            output_mime_type=os.getenv("GEMINI_IMAGE_MIME_TYPE", "image/jpeg"),
            output_compression=int(os.getenv("GEMINI_IMAGE_COMPRESSION", "85")),
            primary_attempts=primary_attempts,
            fallback_attempts=fallback_attempts,
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
    def __init__(self, message: str, *, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


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
    ) -> None:
        self.settings = settings
        self.client = client
        self.sleep = sleep

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
        policies = (
            (self.settings.primary_model, self.settings.primary_attempts),
            (self.settings.fallback_model, self.settings.fallback_attempts),
        )
        for model, limit in policies:
            for attempt in range(1, limit + 1):
                attempts += 1
                try:
                    return replace(
                        self._call(slot_id=slot_id, model=model, contents=contents),
                        attempts=attempts,
                    )
                except Exception as exc:
                    last_error = exc
                    if _terminal(exc):
                        raise ImageGenerationError(
                            f"Image request rejected without retry: {type(exc).__name__}",
                            attempts=attempts,
                        ) from exc
                    if not _retryable(exc):
                        break
                    if attempt < limit:
                        self.sleep(min(2 ** (attempt - 1), 4))
        assert last_error is not None
        raise ImageGenerationError(
            "Nano Banana primary and Lite fallback failed; "
            f"last error: {type(last_error).__name__}",
            attempts=attempts,
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
        parts = (
            getattr(getattr(candidates[0], "content", None), "parts", [])
            if candidates
            else []
        )
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return ImageResult(
                    slot_id=slot_id,
                    image_bytes=bytes(inline.data),
                    model=model,
                    mime_type=getattr(inline, "mime_type", None)
                    or self.settings.output_mime_type,
                )
        raise ValueError("Gemini image response did not contain image data")
