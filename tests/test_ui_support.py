from pathlib import Path

import pytest

from funding_story_ai.ui_support import (
    conversation_payload,
    inline_preview_images,
    mark_stage_answered,
    resolve_run_directory,
    save_uploaded_image,
)


def test_mark_stage_answered_only_updates_matching_question_group() -> None:
    flags = {
        "primary_answered": False,
        "secondary_answered": False,
        "combined_answered": False,
    }
    updated = mark_stage_answered("primary-details", flags)
    assert updated["primary_answered"] is True
    assert updated["secondary_answered"] is False
    assert flags["primary_answered"] is False


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


def test_run_directory_rejects_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run id"):
        resolve_run_directory(tmp_path, "../../outside")


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
