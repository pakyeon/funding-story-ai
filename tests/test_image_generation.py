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


def test_rate_limit_failure_keeps_model_code_and_attempt_history() -> None:
    models = _Models(
        [
            RuntimeError("429 RESOURCE_EXHAUSTED; retry in 12s"),
            RuntimeError("429 RESOURCE_EXHAUSTED; retry in 12s"),
            RuntimeError("429 RESOURCE_EXHAUSTED; retry in 12s"),
        ]
    )
    sleeps: list[float] = []
    adapter = GeminiImageAdapter(
        ImageSettings(
            request_interval_seconds=0,
            retry_base_seconds=2,
            retry_max_seconds=30,
            retry_jitter_seconds=0,
            fallback_delay_seconds=5,
        ),
        client=SimpleNamespace(models=models),
        sleep=sleeps.append,
        jitter=lambda: 0,
    )

    with pytest.raises(ImageGenerationError) as error:
        adapter.generate(slot_id="slot_automation_return_01", prompt="장면")

    assert error.value.attempts == 3
    assert error.value.category == "rate_limit"
    assert error.value.status_code == 429
    assert error.value.model == "gemini-3.1-flash-lite-image"
    assert len(error.value.attempt_history) == 3
    assert [item["model"] for item in error.value.attempt_history] == [
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-image",
        "gemini-3.1-flash-lite-image",
    ]
    assert [item["delay_seconds"] for item in error.value.attempt_history] == [12, 12, 0]
    assert sleeps == [12, 12]


def test_adapter_spaces_successive_slot_requests() -> None:
    models = _Models([_image_response(b"first"), _image_response(b"second")])
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    adapter = GeminiImageAdapter(
        ImageSettings(request_interval_seconds=3),
        client=SimpleNamespace(models=models),
        sleep=sleep,
        monotonic=lambda: now[0],
        jitter=lambda: 0,
    )

    adapter.generate(slot_id="slot_first", prompt="첫 장면")
    adapter.generate(slot_id="slot_second", prompt="둘째 장면")

    assert sleeps == [3]


def test_permission_rejection_is_not_retried_or_fallbacked() -> None:
    models = _Models([RuntimeError("403 permission denied")])
    adapter = GeminiImageAdapter(
        ImageSettings(), client=SimpleNamespace(models=models), sleep=lambda _: None
    )

    with pytest.raises(ImageGenerationError, match="without retry") as error:
        adapter.generate(slot_id="slot_problem_environment_01", prompt="장면")

    assert error.value.attempts == 1
    assert error.value.category == "permission"
    assert error.value.status_code == 403
    assert error.value.attempt_history[0]["retryable"] is False
    assert len(models.calls) == 1
