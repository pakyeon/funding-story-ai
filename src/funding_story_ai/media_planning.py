from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any, Protocol

from .data_repository import DataRepository

PRIORITY_RANK = {"required": 0, "recommended": 1, "optional": 2}
class SceneAdapter(Protocol):
    def generate_json(
        self, *, prompt: str, response_schema: dict[str, Any]
    ) -> Any: ...


class MediaPlanningError(ValueError):
    """Raised when media facts cannot produce a safe plan."""


def _planner_state(fact: dict[str, Any]) -> str:
    if fact["availability"] == "unknown":
        return "unknown"
    if fact["availability"] == "explicitly_absent":
        return "inactive"
    if fact["support_level"] in {"none", "rejected"}:
        return "inactive"
    if fact["support_level"] == "supported":
        return "verified"
    return "unverified"


def _section_for_group(
    *, template: dict[str, Any], group: dict[str, Any]
) -> dict[str, Any]:
    bound_id = template.get("media_section_bindings", {}).get(group["id"])
    if bound_id is not None:
        for section in template["layout"]:
            if section["id"] == bound_id:
                return section
    allowed = set(group["allowed_section_types"])
    for section in template["layout"]:
        if section["type"] in allowed:
            return section
    raise MediaPlanningError(
        f"Template {template['id']} has no section for {group['id']}"
    )


def _required_reference_roles(
    *,
    group_id: str,
    group: dict[str, Any],
    proposition_texts: list[str],
    fact_reference_roles: list[str],
) -> list[str]:
    allowed = set(group["reference_asset_roles"])
    declared = [role for role in fact_reference_roles if role in allowed]
    if declared:
        return _unique(declared)
    if group["reference_policy"] != "required":
        return []
    if group_id == "cleaning_mechanism":
        return ["product_body"]
    if group_id == "automation_return":
        joined = " ".join(proposition_texts).lower()
        if any(word in joined for word in ("도크", "스테이션", "dock")):
            return ["dock"]
        return ["product_body"]
    return list(group["reference_asset_roles"][:1])


def _numbers(value: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", value))


def _scene_response_schema(slot_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["scenes"],
        "properties": {
            "scenes": {
                "type": "array",
                "minItems": len(slot_ids),
                "maxItems": len(slot_ids),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "slot_id",
                        "grounding_fact_ids",
                        "reference_asset_ids",
                        "visual_direction",
                        "text_policy",
                    ],
                    "properties": {
                        "slot_id": {"enum": slot_ids},
                        "grounding_fact_ids": {
                            "type": "array",
                            "minItems": 1,
                            "uniqueItems": True,
                            "items": {"type": "string"},
                        },
                        "reference_asset_ids": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": {"type": "string"},
                        },
                        "visual_direction": {"type": "string", "minLength": 1},
                        "text_policy": {
                            "enum": ["allowed_grounded_only", "no_text"]
                        },
                    },
                },
            }
        },
    }


def _scene_prompt(slots: list[dict[str, Any]], fact_texts: dict[str, str]) -> str:
    view = []
    for slot in slots:
        view.append(
            {
                "slot_id": slot["slot_id"],
                "capability_group": slot["capability_group"],
                "persuasion_goal": slot["persuasion_goal"],
                "grounding_fact_ids": slot["fact_ids"],
                "approved_facts": [fact_texts[fact_id] for fact_id in slot["fact_ids"]],
                "reference_policy": slot["reference_policy"],
                "reference_asset_ids": slot["reference_asset_ids"],
                "base_visual_direction": slot["scene"]["visual_direction"],
            }
        )
    return f"""승인된 로봇청소기 사실과 미디어 슬롯으로 서로 구별되는 장면 명세를 만드세요.

규칙:
- 모든 slot_id를 정확히 한 번 반환합니다.
- 각 슬롯의 grounding_fact_ids와 reference_asset_ids를 입력과 똑같이 반환합니다.
- 승인된 사실에 없는 기능·수치·인증·구성품·UI를 추가하지 않습니다.
- 제품 외형 장면은 제공된 reference_asset_ids의 제품만 사용합니다.
- 서로 다른 슬롯에 같은 구도·배경·동작을 반복하지 않습니다.
- 이미지 안 문자는 허용하지만 승인된 짧은 문구와 수치만 사용할 수 있습니다.
- 실제 동작 영상은 만들지 않으므로 text_policy만 결정하고 media_kind를 변경하지 않습니다.

입력:
{json.dumps(view, ensure_ascii=False)}
"""


def _apply_scene_output(
    *,
    slots: list[dict[str, Any]],
    response: dict[str, Any],
    fact_texts: dict[str, str],
) -> list[dict[str, Any]]:
    rows = response.get("scenes")
    if not isinstance(rows, list):
        raise MediaPlanningError("Scene response must contain scenes")
    by_id = {row.get("slot_id"): row for row in rows if isinstance(row, dict)}
    expected = {slot["slot_id"] for slot in slots}
    if len(by_id) != len(rows) or set(by_id) != expected:
        raise MediaPlanningError("Scene response must cover every slot exactly once")
    result = []
    for slot in slots:
        row = by_id[slot["slot_id"]]
        if set(row.get("grounding_fact_ids", [])) != set(slot["fact_ids"]):
            raise MediaPlanningError(
                f"{slot['slot_id']} scene changed its grounding fact ids"
            )
        if set(row.get("reference_asset_ids", [])) != set(
            slot["reference_asset_ids"]
        ):
            raise MediaPlanningError(
                f"{slot['slot_id']} scene changed its reference asset ids"
            )
        source_numbers = set().union(
            *(_numbers(fact_texts[fact_id]) for fact_id in slot["fact_ids"])
        )
        output_numbers = _numbers(str(row.get("visual_direction", "")))
        if not output_numbers.issubset(source_numbers):
            raise MediaPlanningError(f"{slot['slot_id']} scene invented a numeric claim")
        if row.get("text_policy") not in {"allowed_grounded_only", "no_text"}:
            raise MediaPlanningError(f"{slot['slot_id']} scene has invalid text policy")
        result.append(
            {
                **slot,
                "scene": {
                    "summary": slot["scene"]["summary"],
                    "visual_direction": str(row["visual_direction"]).strip(),
                    "text_policy": row["text_policy"],
                },
            }
        )
    return result


class MediaPlanner:
    def __init__(
        self,
        *,
        repository: DataRepository,
        scene_adapter: SceneAdapter | None = None,
        max_images: int = 8,
    ) -> None:
        if not 1 <= max_images <= 8:
            raise ValueError("max_images must be between 1 and 8")
        self.repository = repository
        self.scene_adapter = scene_adapter
        self.max_images = max_images

    def plan(
        self,
        *,
        media_facts: dict[str, Any],
        template: dict[str, Any],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        self.repository.validate_media_facts(media_facts)
        self.repository.validate_media_profile(profile)
        if template["media_profile_ref"] != profile["id"]:
            raise MediaPlanningError("Template and media profile do not match")

        proposition_by_id = {
            item["proposition_id"]: item for item in media_facts["propositions"]
        }
        fact_by_id = {item["fact_id"]: item for item in media_facts["facts"]}
        asset_by_id = {item["asset_id"]: item for item in media_facts["assets"]}
        group_facts: dict[str, list[dict[str, Any]]] = defaultdict(list)
        group_propositions: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for fact in media_facts["facts"]:
            for proposition_id in fact["proposition_ids"]:
                proposition = proposition_by_id[proposition_id]
                group = proposition["capability_group"]
                group_facts[group].append(fact)
                group_propositions[group].append(proposition)

        active_groups: list[str] = []
        inactive_groups: list[str] = []
        placeholder_groups: list[str] = []
        missing_reference_roles: list[str] = []
        slots: list[dict[str, Any]] = []
        placeholders: list[dict[str, Any]] = []
        missing_required_groups: list[str] = []
        section_order = {
            section["id"]: index for index, section in enumerate(template["layout"])
        }

        for group in profile["capability_groups"]:
            group_id = group["id"]
            is_required = group["cardinality"]["min"] > 0
            facts = list({item["fact_id"]: item for item in group_facts[group_id]}.values())
            states = {_planner_state(fact) for fact in facts}
            usable = [
                fact
                for fact in facts
                if _planner_state(fact) in {"verified", "unverified"}
                and not (
                    group_id == "evidence_performance"
                    and _planner_state(fact) != "verified"
                )
            ]
            if not usable:
                if "unknown" in states:
                    section = _section_for_group(template=template, group=group)
                    if is_required:
                        inactive_groups.append(group_id)
                        missing_required_groups.append(group_id)
                    else:
                        placeholder_groups.append(group_id)
                        placeholder = group["placeholder"]
                        placeholders.append(
                            {
                                "capability_group": group_id,
                                "section_id": section["id"],
                                "title": placeholder["title"],
                                "description": placeholder["description"],
                                "input_format": placeholder["input_format"],
                            }
                        )
                else:
                    inactive_groups.append(group_id)
                    if is_required:
                        missing_required_groups.append(group_id)
                continue

            section = _section_for_group(template=template, group=group)
            active_groups.append(group_id)
            proposition_ids = _unique(
                proposition["proposition_id"]
                for proposition in group_propositions[group_id]
                if proposition["fact_id"] in {fact["fact_id"] for fact in usable}
            )
            fact_ids = _unique(fact["fact_id"] for fact in usable)
            approved_asset_ids = _unique(
                asset_id
                for fact in usable
                for asset_id in fact["asset_refs"]
                if asset_id in asset_by_id
            )
            reference_roles = set(group["reference_asset_roles"])
            linked_reference_asset_ids = [
                asset_id
                for asset_id in approved_asset_ids
                if asset_by_id[asset_id].get("generation_available", True)
                and reference_roles.intersection(asset_by_id[asset_id].get("roles", []))
            ]
            catalog_reference_asset_ids = [
                asset_id
                for asset_id, asset in asset_by_id.items()
                if asset.get("generation_available", True)
                and reference_roles.intersection(asset.get("roles", []))
            ]
            reference_asset_ids = _unique(
                [*linked_reference_asset_ids, *catalog_reference_asset_ids]
            )
            proposition_texts = [proposition_by_id[item]["text"] for item in proposition_ids]
            fact_reference_roles = _unique(
                role for fact in usable for role in fact.get("reference_roles", [])
            )
            required_roles = _required_reference_roles(
                group_id=group_id,
                group=group,
                proposition_texts=proposition_texts,
                fact_reference_roles=fact_reference_roles,
            )
            available_roles = {
                role
                for asset_id in reference_asset_ids
                for role in asset_by_id[asset_id].get("roles", [])
            }
            missing_reference_roles.extend(
                role for role in required_roles if role not in available_roles
            )
            if "static" not in group["supported_media_kinds"]:
                raise MediaPlanningError(
                    f"MVP supports static media only: {group_id}"
                )
            slots.append(
                {
                    "slot_id": f"slot_{group_id}_01",
                    "capability_group": group_id,
                    "grouping_key": group["slot_grouping_key"],
                    "section_id": section["id"],
                    "persuasion_goal": group["persuasion_goal"],
                    "priority": group["priority"],
                    "placement": group["placement_candidates"][0],
                    "media_kind": "static",
                    "reference_policy": (
                        "required" if required_roles else group["reference_policy"]
                    ),
                    "reference_asset_ids": reference_asset_ids,
                    "fact_ids": fact_ids,
                    "proposition_ids": proposition_ids,
                    "scene": {
                        "summary": " ".join(proposition_texts),
                        "visual_direction": group["visual_hint"],
                        "text_policy": "allowed_grounded_only",
                    },
                }
            )

        if len(slots) > self.max_images:
            selected = sorted(
                slots,
                key=lambda slot: (
                    PRIORITY_RANK[slot["priority"]],
                    section_order[slot["section_id"]],
                    slot["slot_id"],
                ),
            )[: self.max_images]
            selected_ids = {slot["slot_id"] for slot in selected}
            for slot in slots:
                if slot["slot_id"] not in selected_ids:
                    group = next(
                        item
                        for item in profile["capability_groups"]
                        if item["id"] == slot["capability_group"]
                    )
                    placeholder = group["placeholder"]
                    placeholder_groups.append(slot["capability_group"])
                    placeholders.append(
                        {
                            "capability_group": slot["capability_group"],
                            "section_id": slot["section_id"],
                            "title": placeholder["title"],
                            "description": placeholder["description"],
                            "input_format": placeholder["input_format"],
                        }
                    )
            slots = selected

        slots.sort(
            key=lambda slot: (section_order[slot["section_id"]], slot["slot_id"])
        )
        fact_texts = {
            fact_id: " ".join(
                proposition_by_id[prop_id]["text"]
                for prop_id in fact_by_id[fact_id]["proposition_ids"]
            )
            for fact_id in fact_by_id
        }
        if slots and self.scene_adapter is not None:
            result = self.scene_adapter.generate_json(
                prompt=_scene_prompt(slots, fact_texts),
                response_schema=_scene_response_schema(
                    [slot["slot_id"] for slot in slots]
                ),
            )
            slots = _apply_scene_output(
                slots=slots,
                response=result.data,
                fact_texts=fact_texts,
            )

        missing_reference_roles = _unique(missing_reference_roles)
        placeholder_groups = _unique(placeholder_groups)
        if missing_required_groups:
            decision = "needs_required_information"
        elif missing_reference_roles:
            decision = "needs_reference_assets"
        elif placeholders:
            decision = "draft_with_placeholders"
        else:
            decision = "ready"
        generation_allowed = decision in {"ready", "draft_with_placeholders"}
        publishable = decision == "ready"
        reasons = []
        if missing_required_groups:
            reasons.append(
                "필수 능력군 정보가 부족합니다: " + ", ".join(missing_required_groups)
            )
        if missing_reference_roles:
            reasons.append(
                "필수 참조 자산이 부족합니다: " + ", ".join(missing_reference_roles)
            )
        if placeholders:
            reasons.append("선택 정보가 없어 편집용 placeholder를 포함합니다.")
        if not reasons:
            reasons.append("승인된 사실과 참조 자산으로 미디어 계획을 생성할 수 있습니다.")
        plan = {
            "schema_version": "media-plan-v1",
            "brief_id": media_facts["brief_id"],
            "template_id": template["id"],
            "media_profile_id": profile["id"],
            "decision": decision,
            "publishable": publishable,
            "generation_allowed": generation_allowed,
            "active_groups": _unique(active_groups),
            "inactive_groups": _unique(inactive_groups),
            "placeholder_groups": placeholder_groups,
            "missing_reference_roles": missing_reference_roles,
            "slots": slots,
            "placeholders": placeholders,
            "reasons": reasons,
        }
        self.repository.validate_media_plan(plan)
        return plan


def _unique(values: Any) -> list[Any]:
    return list(dict.fromkeys(values))
