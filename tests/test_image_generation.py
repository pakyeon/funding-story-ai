import base64
from types import SimpleNamespace

from funding_story_ai.image_generation import (
    ImageResult,
    ImageSettings,
    OpenAIImageAdapter,
    RetryingFallbackImageAdapter,
)


class FakeImages:
    def __init__(self) -> None:
        self.kwargs = None

    def edit(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    b64_json=base64.b64encode(b"fake-image").decode(),
                    revised_prompt=None,
                )
            ],
        )

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    b64_json=base64.b64encode(b"generated-image").decode(),
                    revised_prompt=None,
                )
            ],
        )


def test_image_adapter_edits_reference_and_omits_input_fidelity(tmp_path) -> None:
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(b"reference")
    settings = ImageSettings()
    images = FakeImages()
    adapter = OpenAIImageAdapter(
        settings,
        client=SimpleNamespace(images=images),
    )

    result = adapter.edit_reference(
        section_id="hero", reference_path=reference, prompt="제품 히어로 이미지"
    )

    assert result.image_bytes == b"fake-image"
    assert "input_fidelity" not in images.kwargs


def test_image_adapter_generates_without_reference_image() -> None:
    settings = ImageSettings()
    images = FakeImages()
    adapter = OpenAIImageAdapter(
        settings,
        client=SimpleNamespace(images=images),
    )

    result = adapter.generate_text(section_id="hero", prompt="가상 제품 이미지")

    assert result.image_bytes == b"generated-image"
    assert "image" not in images.kwargs


def test_image_adapter_retries_primary_then_uses_fallback() -> None:
    class Failing:
        def generate_text(self, **kwargs):
            raise RuntimeError("primary unavailable")

    class Fallback:
        def generate_text(self, *, section_id, prompt):
            return ImageResult(
                section_id=section_id,
                image_bytes=b"fallback",
                revised_prompt=None,
                provider="google",
                model="gemini-image",
            )

    adapter = RetryingFallbackImageAdapter(
        [Failing(), Fallback()],  # type: ignore[list-item]
        attempts_per_provider=3,
        sleep=lambda _: None,
    )
    result = adapter.generate_text(section_id="hero", prompt="image")
    assert result.provider == "google"
    assert result.attempts == 4
