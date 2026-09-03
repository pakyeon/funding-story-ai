from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .data_repository import DataRepository
from .image_generation import ImageAdapter, ImageSettings
from .media_projection import canonical_digest


def _embed_ai_metadata(data: bytes, *, mime_type: str, model: str) -> tuple[bytes, bool]:
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


def _extension(mime_type: str) -> str:
    return {"image/jpeg": "jpeg", "image/png": "png", "image/webp": "webp"}.get(mime_type, "bin")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_slot_image_prompt(
    *,
    slot: dict[str, Any],
    media_facts: dict[str, Any],
    reference_available: bool,
) -> str:
    proposition_by_id = {item["proposition_id"]: item for item in media_facts["propositions"]}
    approved = [proposition_by_id[item]["text"] for item in slot["proposition_ids"]]
    reference_rule = (
        "첨부된 제품 참조 자산의 본체·도크·부속품 외형을 동일하게 보존하세요."
        if reference_available
        else "실제 제품 외형을 표방하지 않는 설명용 비제품 그래픽으로 만드세요."
    )
    text_rule = (
        "이미지 내 문자는 허용하되 아래 승인 문구만 그대로 사용하세요."
        if slot["scene"]["text_policy"] == "allowed_grounded_only"
        else "이미지 안에 문자·숫자·로고를 넣지 마세요."
    )
    return f"""한국 크라우드펀딩 상세 페이지용 독립 장면 한 장을 생성하세요.

슬롯: {slot["slot_id"]}
설득 목적: {slot["persuasion_goal"]}
장면 요약: {slot["scene"]["summary"]}
시각 방향: {slot["scene"]["visual_direction"]}
승인 사실: {json.dumps(approved, ensure_ascii=False)}

제약:
- {reference_rule}
- {text_rule}
- 승인 사실에 없는 성능·수치·인증·가격·구성품·UI를 만들지 마세요.
- 다른 슬롯의 장면을 이어받거나 동일한 구도와 배경을 반복하지 마세요.
- 사람, 반려동물은 승인 사실이나 장면 명세가 요구할 때만 포함하세요.
- 충분한 여백과 자연스러운 3:2 가로 구도로 구성하세요.
"""


def build_generated_reference_prompt(*, brief: dict[str, Any], roles: Sequence[str]) -> str:
    facts = [
        f"{item['name']}: {item['value']}{item['unit'] or ''}" for item in brief["product"]["facts"]
    ]
    features = [item["description"] for item in brief["features"]]
    return f"""크라우드펀딩 이미지 제작에 사용할 가상의 제품 기준 이미지를 생성하세요.

제품명: {brief["product"]["name"]}
제품 유형: {brief["product"]["product_type"]}
제품 요약: {brief["product"]["summary"]}
확인된 제품 정보: {json.dumps([*facts, *features], ensure_ascii=False)}
필요한 참조 역할: {json.dumps(list(roles), ensure_ascii=False)}

제약:
- 실제 출시 제품이나 특정 브랜드 제품을 복제하지 말고 새로운 가상 제품 디자인으로 만드세요.
- 이후 여러 장면에서 외형을 일관되게 재사용할 수 있도록 본체 전체가 잘 보이는
  중립적인 스튜디오 제품 사진으로 만드세요.
- 확인된 정보에 없는 성능, 수치, 인증, 로고, 구성품, UI를 추가하지 마세요.
- 제품명과 설명 문구를 이미지 안에 넣지 마세요.
- 사람이나 반려동물을 포함하지 말고 자연스러운 3:2 가로 구도로 구성하세요.
"""


def generate_product_reference(
    *,
    brief: dict[str, Any],
    roles: Sequence[str],
    output_dir: Path,
    adapter: ImageAdapter,
) -> tuple[dict[str, Any], Path]:
    """Create one clearly synthetic identity reference for otherwise ungrounded scenes."""
    asset_id = "asset_generated_product_reference"
    prompt = build_generated_reference_prompt(brief=brief, roles=roles)
    result = adapter.generate(
        slot_id="reference_product_body",
        prompt=prompt,
        reference_paths=(),
    )
    image_bytes, metadata_embedded = _embed_ai_metadata(
        result.image_bytes,
        mime_type=result.mime_type,
        model=result.model,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    filename = f"{asset_id}.{_extension(result.mime_type)}"
    target = output_dir / filename
    target.write_bytes(image_bytes)
    asset = {
        "asset_id": asset_id,
        "roles": list(dict.fromkeys(roles)),
        "description": "AI가 생성한 가상 제품 기준 이미지이며 실제 제품 사진이 아닙니다.",
        "source_refs": [brief["source"]["refs"][0]["source_id"]],
        "path": filename,
        "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "provider": result.provider,
        "model": result.model,
        "mime_type": result.mime_type,
        "attempts": result.attempts,
        "ai_metadata_embedded": metadata_embedded,
    }
    return asset, target


def attach_generated_reference(
    *, media_facts: dict[str, Any], asset: dict[str, Any]
) -> dict[str, Any]:
    """Attach a runtime-only visual reference without mutating the approved maker input."""
    enriched = deepcopy(media_facts)
    projection = {key: asset[key] for key in ("asset_id", "roles", "description", "source_refs")}
    enriched["assets"].append(projection)
    roles = set(asset["roles"])
    for fact in enriched["facts"]:
        if roles.intersection(fact["reference_roles"]):
            fact["asset_refs"] = list(dict.fromkeys([*fact["asset_refs"], asset["asset_id"]]))
    return enriched


def planned_image_slots(
    media_plan: dict[str, Any], *, slot_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    available = {slot["slot_id"] for slot in media_plan["slots"]}
    unknown = (slot_ids or set()) - available
    if unknown:
        raise ValueError(f"Unknown media slot ids: {sorted(unknown)}")
    return [slot for slot in media_plan["slots"] if slot_ids is None or slot["slot_id"] in slot_ids]


def empty_image_manifest(
    *,
    media_plan: dict[str, Any],
    generation_package: dict[str, Any],
    settings: ImageSettings,
    run_id: str,
    generated_references: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "story-image-manifest-v2",
        "run_id": run_id,
        "media_plan_digest": canonical_digest(media_plan),
        "generation_package_digest": canonical_digest(generation_package),
        "primary_model": settings.primary_model,
        "fallback_model": settings.fallback_model,
        "image_size": settings.image_size,
        "aspect_ratio": settings.aspect_ratio,
        "requested": 0,
        "succeeded": 0,
        "failed": 0,
        "assets": [],
        "generated_references": generated_references or [],
    }


def generate_planned_images(
    *,
    media_plan: dict[str, Any],
    media_facts: dict[str, Any],
    generation_package: dict[str, Any],
    output_dir: Path,
    repository: DataRepository,
    adapter: ImageAdapter,
    settings: ImageSettings,
    slot_ids: set[str] | None = None,
    run_id: str | None = None,
    runtime_reference_paths: dict[str, Path] | None = None,
    generated_references: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    repository.validate_media_plan(media_plan)
    repository.validate_media_facts(media_facts)
    repository.validate_approved_generation_package(generation_package)
    if media_plan["brief_id"] != media_facts["brief_id"]:
        raise ValueError("Media plan and MediaFacts brief ids do not match")
    if not media_plan["generation_allowed"]:
        raise ValueError("Media plan does not allow image generation")
    slots = planned_image_slots(media_plan, slot_ids=slot_ids)
    local_paths = {
        asset_id: Path(path) for asset_id, path in generation_package["local_asset_paths"].items()
    }
    local_paths.update(runtime_reference_paths or {})
    for path in local_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    fact_ids = {item["fact_id"] for item in media_facts["facts"]}
    proposition_ids = {item["proposition_id"] for item in media_facts["propositions"]}
    asset_ids = {item["asset_id"] for item in media_facts["assets"]}
    for slot in slots:
        if not set(slot["fact_ids"]).issubset(fact_ids):
            raise ValueError(f"{slot['slot_id']} references an unknown fact")
        if not set(slot["proposition_ids"]).issubset(proposition_ids):
            raise ValueError(f"{slot['slot_id']} references an unknown proposition")
        if not set(slot["reference_asset_ids"]).issubset(asset_ids):
            raise ValueError(f"{slot['slot_id']} references an unknown asset")
        if slot["reference_policy"] == "required" and not any(
            asset_id in local_paths for asset_id in slot["reference_asset_ids"]
        ):
            raise ValueError(f"{slot['slot_id']} is missing a local reference asset")

    output_dir.mkdir(parents=True, exist_ok=False)
    assets: list[dict[str, Any]] = []
    for slot in slots:
        reference_paths = [
            local_paths[asset_id]
            for asset_id in slot["reference_asset_ids"]
            if asset_id in local_paths
        ]
        prompt = build_slot_image_prompt(
            slot=slot,
            media_facts=media_facts,
            reference_available=bool(reference_paths),
        )
        base = {
            "slot_id": slot["slot_id"],
            "section_id": slot["section_id"],
            "capability_group": slot["capability_group"],
            "reference_asset_ids": slot["reference_asset_ids"],
            "prompt_digest": canonical_digest(prompt),
        }
        try:
            result = adapter.generate(
                slot_id=slot["slot_id"],
                prompt=prompt,
                reference_paths=reference_paths,
            )
            image_bytes, metadata_embedded = _embed_ai_metadata(
                result.image_bytes,
                mime_type=result.mime_type,
                model=result.model,
            )
            filename = f"{slot['slot_id']}.{_extension(result.mime_type)}"
            target = output_dir / filename
            target.write_bytes(image_bytes)
            assets.append(
                {
                    **base,
                    "status": "success",
                    "path": filename,
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "error_type": None,
                    "qa_status": "pending",
                    "qa_notes": ["게시 전 사람의 사실·외형·문자 품질 검토가 필요합니다."],
                    "review_checks": {
                        "scene_distinctness": "pending",
                        "product_fidelity": "pending",
                        "text_legibility": "pending",
                        "claim_grounding": "pending",
                    },
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
                    **base,
                    "status": "error",
                    "path": None,
                    "sha256": None,
                    "error_type": type(exc).__name__,
                    "qa_status": "fail",
                    "qa_notes": ["생성 오류로 초안 이미지를 만들지 못했습니다."],
                    "review_checks": {
                        "scene_distinctness": "not_applicable",
                        "product_fidelity": "not_applicable",
                        "text_legibility": "not_applicable",
                        "claim_grounding": "not_applicable",
                    },
                    "provider": None,
                    "model": None,
                    "mime_type": None,
                    "attempts": max(1, int(getattr(exc, "attempts", 1))),
                    "ai_metadata_embedded": False,
                }
            )

    manifest = {
        "schema_version": "story-image-manifest-v2",
        "run_id": run_id or output_dir.name,
        "media_plan_digest": canonical_digest(media_plan),
        "generation_package_digest": canonical_digest(generation_package),
        "primary_model": settings.primary_model,
        "fallback_model": settings.fallback_model,
        "image_size": settings.image_size,
        "aspect_ratio": settings.aspect_ratio,
        "requested": len(slots),
        "succeeded": sum(item["status"] == "success" for item in assets),
        "failed": sum(item["status"] == "error" for item in assets),
        "assets": assets,
        "generated_references": generated_references or [],
    }
    repository.validate_story_image_manifest(manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
