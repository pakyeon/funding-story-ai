from __future__ import annotations

import base64
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from google.genai import types
from openai import OpenAI


@dataclass(frozen=True, slots=True)
class ImageSettings:
    model: str = "gpt-image-2"
    fallback_model: str = "gemini-2.5-flash-image"
    size: str = "1536x1024"
    quality: str = "low"
    output_format: str = "jpeg"
    output_compression: int = 85
    attempts_per_provider: int = 3

    @classmethod
    def from_env(cls) -> ImageSettings:
        attempts = int(os.getenv("IMAGE_ATTEMPTS_PER_PROVIDER", "3"))
        if attempts < 1 or attempts > 3:
            raise ValueError("IMAGE_ATTEMPTS_PER_PROVIDER must be between 1 and 3")
        return cls(
            model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
            fallback_model=os.getenv(
                "GEMINI_IMAGE_FALLBACK_MODEL", "gemini-2.5-flash-image"
            ),
            size=os.getenv("OPENAI_IMAGE_SIZE", "1536x1024"),
            quality=os.getenv("OPENAI_IMAGE_QUALITY", "low"),
            output_format=os.getenv("OPENAI_IMAGE_OUTPUT_FORMAT", "jpeg"),
            output_compression=int(os.getenv("OPENAI_IMAGE_OUTPUT_COMPRESSION", "85")),
            attempts_per_provider=attempts,
        )


@dataclass(frozen=True, slots=True)
class ImageResult:
    section_id: str
    image_bytes: bytes
    revised_prompt: str | None
    provider: str = "openai"
    model: str = "gpt-image-2"
    mime_type: str = "image/jpeg"
    attempts: int = 1


class ImageAdapter(Protocol):
    def edit_reference(
        self, *, section_id: str, reference_path: Path, prompt: str
    ) -> ImageResult: ...

    def generate_text(self, *, section_id: str, prompt: str) -> ImageResult: ...


class ImageGenerationError(RuntimeError):
    def __init__(self, message: str, *, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class OpenAIImageAdapter:
    def __init__(self, settings: ImageSettings, *, client: Any | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAI()

    @property
    def _mime_type(self) -> str:
        return "image/png" if self.settings.output_format == "png" else "image/jpeg"

    def edit_reference(
        self, *, section_id: str, reference_path: Path, prompt: str
    ) -> ImageResult:
        with reference_path.open("rb") as image_file:
            response = self.client.images.edit(
                image=image_file,
                model=self.settings.model,
                prompt=prompt,
                n=1,
                size=self.settings.size,
                quality=self.settings.quality,
                output_format=self.settings.output_format,
                output_compression=self.settings.output_compression,
                background="opaque",
            )
        return self._result(section_id, response)

    def generate_text(self, *, section_id: str, prompt: str) -> ImageResult:
        response = self.client.images.generate(
            model=self.settings.model,
            prompt=prompt,
            n=1,
            size=self.settings.size,
            quality=self.settings.quality,
            output_format=self.settings.output_format,
            output_compression=self.settings.output_compression,
            background="opaque",
        )
        return self._result(section_id, response)

    def _result(self, section_id: str, response: Any) -> ImageResult:
        if not response.data or not response.data[0].b64_json:
            raise ValueError("OpenAI image response did not contain base64 image data")
        return ImageResult(
            section_id=section_id,
            image_bytes=base64.b64decode(response.data[0].b64_json),
            revised_prompt=response.data[0].revised_prompt,
            provider="openai",
            model=self.settings.model,
            mime_type=self._mime_type,
        )


class GeminiImageAdapter:
    """Vertex AI fallback for text-to-image and reference-conditioned generation."""

    def __init__(self, settings: ImageSettings, *, client: Any) -> None:
        self.settings = settings
        self.client = client

    def edit_reference(
        self, *, section_id: str, reference_path: Path, prompt: str
    ) -> ImageResult:
        mime = {
            ".png": "image/png",
            ".webp": "image/webp",
        }.get(reference_path.suffix.lower(), "image/jpeg")
        contents = [types.Content(role="user", parts=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=reference_path.read_bytes(), mime_type=mime),
        ])]
        return self._generate(section_id=section_id, contents=contents)

    def generate_text(self, *, section_id: str, prompt: str) -> ImageResult:
        return self._generate(section_id=section_id, contents=prompt)

    def _generate(self, *, section_id: str, contents: Any) -> ImageResult:
        response = self.client.models.generate_content(
            model=self.settings.fallback_model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=[types.Modality.IMAGE],
                image_config=types.ImageConfig(
                    aspect_ratio="3:2",
                    output_mime_type="image/jpeg",
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
                    section_id=section_id,
                    image_bytes=bytes(inline.data),
                    revised_prompt=None,
                    provider="google",
                    model=self.settings.fallback_model,
                    mime_type=getattr(inline, "mime_type", None) or "image/jpeg",
                )
        raise ValueError("Gemini image response did not contain image data")


class RetryingFallbackImageAdapter:
    """Try each configured provider up to the bounded retry count."""

    def __init__(
        self,
        adapters: list[ImageAdapter],
        *,
        attempts_per_provider: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not adapters:
            raise ValueError("At least one image adapter is required")
        self.adapters = adapters
        self.attempts_per_provider = attempts_per_provider
        self.sleep = sleep

    def edit_reference(
        self, *, section_id: str, reference_path: Path, prompt: str
    ) -> ImageResult:
        return self._invoke(
            "edit_reference",
            section_id=section_id,
            reference_path=reference_path,
            prompt=prompt,
        )

    def generate_text(self, *, section_id: str, prompt: str) -> ImageResult:
        return self._invoke("generate_text", section_id=section_id, prompt=prompt)

    def _invoke(self, method: str, **kwargs: Any) -> ImageResult:
        last_error: Exception | None = None
        total_attempts = 0
        for adapter in self.adapters:
            for attempt in range(1, self.attempts_per_provider + 1):
                total_attempts += 1
                try:
                    result = getattr(adapter, method)(**kwargs)
                    return replace(result, attempts=total_attempts)
                except Exception as exc:
                    last_error = exc
                    if attempt < self.attempts_per_provider:
                        self.sleep(min(2 ** (attempt - 1), 4))
        assert last_error is not None
        raise ImageGenerationError(
            f"All image providers failed; last error: {type(last_error).__name__}",
            attempts=total_attempts,
        ) from last_error
