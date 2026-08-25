from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any

from .data_repository import DataRepository
from .image_generation import ImageAdapter, ImageSettings


def _embed_ai_metadata(data: bytes, *, mime_type: str, model: str) -> tuple[bytes, bool]:
    """Embed a compact AI marker in PNG tEXt or JPEG COM when bytes are valid."""
    marker = json.dumps(
        {"AI-Generated": True, "model": model}, ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    if mime_type == "image/png" and data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 33:
        payload = b"AI-Generated\x00" + marker
        chunk = b"tEXt" + payload
        encoded = struct.pack(">I", len(payload)) + chunk + struct.pack(">I", zlib.crc32(chunk))
        return data[:33] + encoded + data[33:], True
    if mime_type == "image/jpeg" and data.startswith(b"\xff\xd8"):
        payload = marker[:65531]
        segment = b"\xff\xfe" + struct.pack(">H", len(payload) + 2) + payload
        return data[:2] + segment + data[2:], True
    return data, False


def _extension(mime_type: str, fallback: str) -> str:
    return {"image/jpeg": "jpeg", "image/png": "png", "image/webp": "webp"}.get(
        mime_type, fallback
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_section_image_prompt(
    *,
    section: dict[str, Any],
    template: dict[str, Any],
    reference_available: bool = True,
    visual_identity: str | None = None,
) -> str:
    palette = ", ".join(template["style"]["color_palette"])
    section_id = section["template_section_id"]
    template_section = next(
        item for item in template["layout"] if item["id"] == section_id
    )
    visual_strategy = template_section["visual_hint"]
    input_instruction = (
        "참조 이미지에 있는 가상 제품 디자인을 동일한 제품으로 인식할 수 있게 보존하여"
        if reference_available
        else "입력 제품 설명만을 바탕으로 가상의 제품 외관을 일관된 디자인으로 설정하여"
    )
    fidelity_constraint = (
        "- 참조 이미지에서 확인되는 제품 형태·색·구성을 유지"
        if reference_available
        else (
            "- 같은 실행의 모든 섹션에서 동일 제품으로 보이도록 "
            "본체·센서·도크의 형태와 색을 일관되게 표현"
        )
    )
    return f"""{input_instruction}, 한국 크라우드펀딩 상세페이지의
`{section_id}` 섹션용 가로 이미지를 만드세요.

섹션 목적: {section['image_intent']['purpose']}
안전한 시각 전략: {visual_strategy}
템플릿 무드: {template['style']['visual_mood']}
색 팔레트 참고: {palette}
제품 시각 정체성(사용자 입력·brief 범위): {visual_identity or '구체 외형 미제공'}

필수 제약:
{fidelity_constraint}
- 입력이 합성 예제이면 실제 판매 제품이나 브랜드가 아닌 가상 디자인으로 표현
- 읽을 수 있는 텍스트, 로고, 숫자, 가격, 인증, 수상, UI 글자, 워터마크를 넣지 않음
- 사람, 반려동물, 입력에 없는 추가 구성품을 넣지 않음
- 입력에 없는 성능을 암시하는 과장 효과나 시험 장면을 넣지 않음
- 분할 패널, 연속 동작 프레임, 화살표, 이동 경로선, 센서 광선, 투시 효과를 넣지 않음
- 같은 제품을 불필요하게 복제하지 않음
- 충분한 여백과 사실적인 제품 사진 또는 단순한 비문자 시각 흐름으로 구성
- 모든 제목, 설명, 성능 수치와 시험 조건은 이미지 밖의 편집 가능한 HTML로 표현
- 이미지 안에는 타이포그래피를 만들지 않음
"""


def planned_image_sections(
    story: dict[str, Any],
    template: dict[str, Any],
    section_ids: set[str] | None = None,
    *,
    reference_available: bool = True,
    visual_identity: str | None = None,
) -> list[dict[str, str]]:
    available = {
        section["template_section_id"]
        for section in story["sections"]
        if section["image_intent"]["required"]
    }
    unknown = (section_ids or set()) - available
    if unknown:
        raise ValueError(f"Unknown or image-optional section ids: {sorted(unknown)}")
    plans = []
    for section in story["sections"]:
        if not section["image_intent"]["required"]:
            continue
        if section_ids is not None and section["template_section_id"] not in section_ids:
            continue
        prompt = build_section_image_prompt(
            section=section,
            template=template,
            reference_available=reference_available,
            visual_identity=visual_identity,
        )
        plans.append(
            {
                "section_id": section["template_section_id"],
                "prompt": prompt,
            }
        )
    return plans


def generate_section_images(
    *,
    story_path: Path,
    reference_path: Path | None,
    output_dir: Path,
    repository: DataRepository,
    adapter: ImageAdapter,
    settings: ImageSettings,
    section_ids: set[str] | None = None,
    run_id: str | None = None,
    visual_identity: str | None = None,
) -> dict[str, Any]:
    story = repository.load_story_result(story_path)
    template = repository.get_template(story["template_id"])
    reference_available = reference_path is not None
    plans = planned_image_sections(
        story,
        template,
        section_ids,
        reference_available=reference_available,
        visual_identity=visual_identity,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    assets: list[dict[str, Any]] = []
    generated_seed_path: Path | None = None
    generated_seed_sha256: str | None = None
    sections_by_id = {
        section["template_section_id"]: section for section in story["sections"]
    }
    for plan in plans:
        try:
            active_reference = reference_path or generated_seed_path
            if active_reference is None:
                result = adapter.generate_text(
                    section_id=plan["section_id"],
                    prompt=plan["prompt"],
                )
            else:
                if reference_path is None:
                    prompt = build_section_image_prompt(
                        section=sections_by_id[plan["section_id"]],
                        template=template,
                        reference_available=True,
                        visual_identity=visual_identity,
                    )
                    plan["prompt"] = prompt
                result = adapter.edit_reference(
                    section_id=plan["section_id"],
                    reference_path=active_reference,
                    prompt=plan["prompt"],
                )
            image_bytes, metadata_embedded = _embed_ai_metadata(
                result.image_bytes,
                mime_type=result.mime_type,
                model=result.model,
            )
            filename = (
                f"{plan['section_id']}.{_extension(result.mime_type, settings.output_format)}"
            )
            target = output_dir / filename
            target.write_bytes(image_bytes)
            if reference_path is None and generated_seed_path is None:
                generated_seed_path = target
                generated_seed_sha256 = hashlib.sha256(image_bytes).hexdigest()
            assets.append(
                {
                    "section_id": plan["section_id"],
                    "status": "success",
                    "path": filename,
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "error_type": None,
                    "qa_status": "pending",
                    "qa_notes": ["게시 전 사람의 사실·품질 검토가 필요합니다."],
                    "provider": result.provider,
                    "model": result.model,
                    "mime_type": result.mime_type,
                    "attempts": result.attempts,
                    "ai_metadata_embedded": metadata_embedded,
                }
            )
        except Exception as exc:
            assets.append(
                {
                    "section_id": plan["section_id"],
                    "status": "error",
                    "path": None,
                    "sha256": None,
                    "error_type": type(exc).__name__,
                    "qa_status": "fail",
                    "qa_notes": ["생성 오류로 미리보기 사용 불가"],
                    "provider": None,
                    "model": None,
                    "mime_type": None,
                    "attempts": max(1, int(getattr(exc, "attempts", 1))),
                    "ai_metadata_embedded": False,
                }
            )

    manifest = {
        "schema_version": "story-image-manifest-v1",
        "run_id": run_id or output_dir.name,
        "story_sha256": file_sha256(story_path),
        "reference_sha256": file_sha256(reference_path) if reference_path else None,
        "input_mode": "reference-edit" if reference_path else "text-seeded-edit",
        "generated_seed_sha256": generated_seed_sha256,
        "model": settings.model,
        "size": settings.size,
        "quality": settings.quality,
        "requested": len(plans),
        "succeeded": sum(asset["status"] == "success" for asset in assets),
        "failed": sum(asset["status"] == "error" for asset in assets),
        "assets": assets,
    }
    repository.validate_story_image_manifest(manifest)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
