from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    project_id: str
    location: str = "global"
    primary_model: str = "gemini-3.7-flash"
    fallback_model: str = "gemini-3.6-flash"
    primary_access_attempts: int = 5
    request_timeout_ms: int = 120000
    max_output_tokens: int = 24576
    thinking_level: str = "MEDIUM"

    @classmethod
    def from_env(cls, *, require_project: bool = True) -> RuntimeSettings:
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
        if require_project and not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT is required")

        thinking_level = os.getenv("GEMINI_THINKING_LEVEL", "MEDIUM").upper()
        if thinking_level not in {"LOW", "MEDIUM", "HIGH"}:
            raise ValueError("GEMINI_THINKING_LEVEL must be LOW, MEDIUM, or HIGH")

        return cls(
            project_id=project_id,
            location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            primary_model=os.getenv("GEMINI_PRIMARY_MODEL", "gemini-3.7-flash"),
            fallback_model=os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.6-flash"),
            primary_access_attempts=_positive_int("GEMINI_PRIMARY_ACCESS_ATTEMPTS", 5),
            request_timeout_ms=_positive_int("GEMINI_REQUEST_TIMEOUT_MS", 120000),
            max_output_tokens=_positive_int("GEMINI_MAX_OUTPUT_TOKENS", 24576),
            thinking_level=thinking_level,
        )
