from pathlib import Path

import pytest

from funding_story_ai.ui_support import (
    image_failure_summary,
    inline_preview_images,
    load_run_artifacts,
    save_uploaded_image,
)


def test_image_failure_summary_explains_rate_limit_and_attempts() -> None:
    message = image_failure_summary(
        {
            "error_category": "rate_limit",
            "error_code": 429,
            "attempts": 3,
        }
    )

    assert "호출 한도" in message
    assert "HTTP 429" in message
    assert "3회" in message


def test_image_failure_summary_supports_legacy_manifest() -> None:
    message = image_failure_summary({"attempts": 1})

    assert "분류되지 않은" in message


def test_uploaded_image_uses_a_fixed_safe_filename(tmp_path: Path) -> None:
    target = save_uploaded_image(
        root=tmp_path,
        input_id="ui/test",
        filename="../../product.PNG",
        content=b"image",
    )
    assert target == tmp_path / "ui-test" / "reference.png"
    assert target.read_bytes() == b"image"


def test_uploaded_image_rejects_non_image_suffix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="이미지만"):
        save_uploaded_image(
            root=tmp_path,
            input_id="ui-test",
            filename="payload.html",
            content=b"unsafe",
        )


def test_inline_preview_images_embeds_only_run_local_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-example"
    images = run_dir / "images"
    images.mkdir(parents=True)
    (images / "hero.png").write_bytes(b"png")
    rendered = inline_preview_images(
        '<img src="images/hero.png"><img src="https://example.com/image.png">',
        run_dir,
    )
    assert 'src="data:image/png;base64,cG5n"' in rendered
    assert 'src="https://example.com/image.png"' in rendered


def test_load_run_artifacts_inlines_html_and_exposes_downloads(tmp_path: Path) -> None:
    run_id = "run-abcdef"
    run_dir = tmp_path / run_id
    images = run_dir / "images"
    images.mkdir(parents=True)
    (run_dir / "story.json").write_text(
        '{"template_id": "t02", "sections": [], "warnings": []}', encoding="utf-8"
    )
    (run_dir / "media-facts.json").write_text("{}", encoding="utf-8")
    (run_dir / "media-plan.json").write_text("{}", encoding="utf-8")
    (images / "hero.jpeg").write_bytes(b"image")
    (images / "manifest.json").write_text(
        '{"assets": [{"section_id": "hero", "status": "success", '
        '"slot_id": "slot-1", "path": "hero.jpeg"}]}',
        encoding="utf-8",
    )
    (run_dir / "draft.html").write_text('<img src="images/hero.jpeg">', encoding="utf-8")
    record = {
        "run_id": run_id,
        "status": "completed",
        "result": {
            "story": {"path": "story.json"},
            "media_facts": {"path": "media-facts.json"},
            "media_plan": {"path": "media-plan.json"},
            "draft_html": {"path": "draft.html"},
            "publishable_html": None,
            "images": {"manifest": {"path": "images/manifest.json"}},
        },
    }
    payload = load_run_artifacts(tmp_path, record)
    assert payload["story"] == {"template_id": "t02", "sections": [], "warnings": []}
    assert 'src="data:image/jpeg;base64,aW1hZ2U="' in payload["draft_html"]
    assert payload["image_data"]["hero.jpeg"].startswith("data:image/jpeg;base64,")
    assert set(payload["source_files"]) == {
        "story.json",
        "media-facts.json",
        "media-plan.json",
        "images/manifest.json",
        "draft.html",
    }
