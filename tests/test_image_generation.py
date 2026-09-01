from types import SimpleNamespace

import pytest

from funding_story_ai.image_generation import (
    GeminiImageAdapter,
    ImageGenerationError,
    ImageSettings,
)


def _image_response(data: bytes = b"image") -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            inline_data=SimpleNamespace(data=data, mime_type="image/jpeg")
                        )
                    ]
                )
            )
        ]
    )


class _Models:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_gemini_image_adapter_uses_primary_and_role_specific_references(tmp_path) -> None:
    product = tmp_path / "product.png"
    dock = tmp_path / "dock.jpg"
    product.write_bytes(b"product")
    dock.write_bytes(b"dock")
    models = _Models([_image_response()])
    adapter = GeminiImageAdapter(
        ImageSettings(), client=SimpleNamespace(models=models), sleep=lambda _: None
    )

    result = adapter.generate(
        slot_id="slot_product_identity_outcome_01",
        prompt="장면",
        reference_paths=[product, dock],
    )

    assert result.model == "gemini-3.1-flash-image"
    assert result.attempts == 1
    assert models.calls[0]["model"] == "gemini-3.1-flash-image"
    assert len(models.calls[0]["contents"][0].parts) == 3
    assert models.calls[0]["config"].image_config.image_size == "1K"
    assert models.calls[0]["config"].image_config.aspect_ratio == "3:2"


def test_transient_primary_failure_retries_then_uses_lite_fallback() -> None:
    models = _Models(
        [
            RuntimeError("503 unavailable"),
            RuntimeError("429 rate limited"),
            _image_response(b"lite"),
        ]
    )
    adapter = GeminiImageAdapter(
        ImageSettings(), client=SimpleNamespace(models=models), sleep=lambda _: None
    )

    result = adapter.generate(slot_id="slot_problem_environment_01", prompt="장면")

    assert result.model == "gemini-3.1-flash-lite-image"
    assert result.image_bytes == b"lite"
    assert result.attempts == 3
    assert [call["model"] for call in models.calls] == [
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
    ]


def test_permission_rejection_is_not_retried_or_fallbacked() -> None:
    models = _Models([RuntimeError("403 permission denied")])
    adapter = GeminiImageAdapter(
        ImageSettings(), client=SimpleNamespace(models=models), sleep=lambda _: None
    )

    with pytest.raises(ImageGenerationError, match="without retry") as error:
        adapter.generate(slot_id="slot_problem_environment_01", prompt="장면")

    assert error.value.attempts == 1
    assert len(models.calls) == 1
