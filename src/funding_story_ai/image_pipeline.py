from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from .data_repository import DataRepository
from .image_generation import ImageSettings, OpenAIImageAdapter


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
    visual_strategy = {
        "hero": (
            "제품 본체와 도킹 스테이션을 한 장면의 정면 3/4 제품 사진으로 배치하고, "
            "본문 제목을 HTML로 얹을 수 있도록 넓은 여백을 남김. "
            "분할 패널이나 단계 도식은 사용하지 않음"
        ),
        "automation_journey": (
            "제품 본체와 도킹 스테이션의 확인 가능한 외관만 사용한 3~4단계 비문자 흐름. "
            "내부 탱크, 배관, 물의 내부 이동 경로, 투시 구조, 센서 광선은 만들지 않음"
        ),
        "performance_proof": (
            "제품 외관 클로즈업과 바닥 먼지 관리 장면을 단순하게 표현하고, "
            "성능 수치와 시험 조건을 HTML 카드로 배치할 넓은 빈 공간을 남김. "
            "차트, 지표 카드, 아이콘, 수치 시각화는 만들지 않음"
        ),
        "solution": (
            "참조 제품 본체와 도킹 스테이션을 실제 거실 바닥에 배치한 단일 사용 장면. "
            "기능 효과, 센서 광선, 내부 구조, 젖음 방지나 살균 효과를 시각적으로 단정하지 않음"
        ),
        "features": (
            "참조 제품 본체와 도킹 스테이션만 단정한 제품 사진으로 배치. "
            "물걸레 패드, 브러시, 필터, 먼지봉투, 전원선 등 입력 이미지에 없는 "
            "구성품을 추가하지 않음"
        ),
    }.get(
        section_id,
        "확인 가능한 제품 외관을 중심으로 한 단일 장면과 HTML 본문을 위한 충분한 여백",
    )
    input_instruction = (
        "참조 이미지에 있는 가상 제품 디자인을 동일한 제품으로 인식할 수 있게 보존하여"
        if reference_available
        else "입력 제품 설명만을 바탕으로 가상의 제품 외관을 일관된 디자인으로 설정하여"
    )
    fidelity_constraint = (
        "- 참조 제품의 본체, 센서, 도크 형태와 색을 유지"
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
- 사람, 반려동물, 추가 구성품을 넣지 않음
- 입력에 없는 성능을 암시하는 과장 효과나 시험 장면을 넣지 않음
- 분할 패널, 연속 동작 프레임, 화살표, 이동 경로선, 센서 광선, 투시 효과를 넣지 않음
- 하나의 제품 본체와 하나의 도크만 한 장면에 배치하고 같은 제품을 복제하지 않음
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
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    return plans


def generate_section_images(
    *,
    story_path: Path,
    reference_path: Path | None,
    output_dir: Path,
    repository: DataRepository,
    adapter: OpenAIImageAdapter,
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
    total_cost = Decimal("0")
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
                    plan["prompt_sha256"] = hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest()
                result = adapter.edit_reference(
                    section_id=plan["section_id"],
                    reference_path=active_reference,
                    prompt=plan["prompt"],
                )
            filename = f"{plan['section_id']}.{settings.output_format}"
            target = output_dir / filename
            target.write_bytes(result.image_bytes)
            if reference_path is None and generated_seed_path is None:
                generated_seed_path = target
                generated_seed_sha256 = hashlib.sha256(result.image_bytes).hexdigest()
            total_cost += result.estimated_cost_usd
            assets.append(
                {
                    "section_id": plan["section_id"],
                    "status": "success",
                    "prompt_sha256": plan["prompt_sha256"],
                    "path": filename,
                    "sha256": hashlib.sha256(result.image_bytes).hexdigest(),
                    "duration_ms": result.duration_ms,
                    "estimated_cost_usd": str(result.estimated_cost_usd),
                    "usage": asdict(result.usage),
                    "error_type": None,
                    "qa_status": "pending",
                    "qa_notes": [],
                }
            )
        except Exception as exc:
            assets.append(
                {
                    "section_id": plan["section_id"],
                    "status": "error",
                    "prompt_sha256": plan["prompt_sha256"],
                    "path": None,
                    "sha256": None,
                    "duration_ms": 0,
                    "estimated_cost_usd": "0",
                    "usage": {
                        "text_input_tokens": 0,
                        "image_input_tokens": 0,
                        "image_output_tokens": 0,
                    },
                    "error_type": type(exc).__name__,
                    "qa_status": "fail",
                    "qa_notes": ["생성 오류로 미리보기 사용 불가"],
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
        "estimated_cost_usd": str(total_cost),
        "assets": assets,
    }
    repository.validate_story_image_manifest(manifest)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
