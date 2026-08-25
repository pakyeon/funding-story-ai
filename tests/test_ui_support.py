from pathlib import Path

import pytest

from funding_story_ai.ui_support import (
    build_run_resource_payload,
    conversation_payload,
    inline_preview_images,
    save_uploaded_image,
)


def test_conversation_payload_keeps_the_question_with_each_followup() -> None:
    messages = [
        {"role": "user", "content": "얇은 로봇청소기입니다."},
        {"role": "assistant", "content": "누구를 위한 제품인가요?"},
        {"role": "user", "content": "가구 아래 청소가 필요한 사용자입니다."},
    ]
    initial, followups = conversation_payload(messages)
    assert initial == "얇은 로봇청소기입니다."
    assert followups == (
        "질문: 누구를 위한 제품인가요?\n답변: 가구 아래 청소가 필요한 사용자입니다.",
    )


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


def test_run_resource_contains_display_artifacts_without_client_filesystem_access(
    tmp_path: Path,
) -> None:
    run_id = "run-abcdef"
    run_dir = tmp_path / run_id
    images = run_dir / "images"
    images.mkdir(parents=True)
    (run_dir / "story.json").write_text('{"sections": []}', encoding="utf-8")
    (images / "hero.jpeg").write_bytes(b"image")
    (images / "manifest.json").write_text(
        '{"assets": [{"section_id": "hero", "status": "success", '
        '"path": "hero.jpeg"}]}',
        encoding="utf-8",
    )
    (run_dir / "preview.html").write_text(
        '<img src="images/hero.jpeg">', encoding="utf-8"
    )
    record = {
        "run_id": run_id,
        "status": "completed",
        "result": {
            "story": {"path": "story.json"},
            "preview": {"path": "preview.html"},
            "images": {"manifest": {"path": "images/manifest.json"}},
        },
    }
    payload = build_run_resource_payload(tmp_path, record)
    assert payload["artifacts"]["story"] == {"sections": []}
    assert 'src="data:image/jpeg;base64,aW1hZ2U="' in payload["artifacts"]["preview_html"]
    assert payload["artifacts"]["image_data"]["hero.jpeg"].startswith(
        "data:image/jpeg;base64,"
    )
