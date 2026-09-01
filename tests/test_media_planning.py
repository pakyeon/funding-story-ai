from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from funding_story_ai.data_repository import DataRepository, DataValidationError
from funding_story_ai.media_planning import MediaPlanner, MediaPlanningError

DATA_ROOT = (
    Path(__file__).parents[1]
    / "evals"
    / "datasets"
    / "robot-vacuum-media-planning-v1"
)


def _id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode()).hexdigest()[:12]}"


def _media_facts_from_case(case: dict[str, Any]) -> dict[str, Any]:
    group_by_fact = {
        item["fact_id"]: item["capability_group"]
        for item in case["expected"]["fact_mappings"]
    }
    asset_ids_by_role: dict[str, list[str]] = {}
    assets = []
    for asset in case["input"]["available_assets"]:
        asset_ids_by_role.setdefault(asset["role"], []).append(asset["asset_id"])
        assets.append(
            {
                "asset_id": asset["asset_id"],
                "roles": [asset["role"]],
                "description": f"{asset['role']} 평가 자산",
                "source_refs": asset["source_refs"],
            }
        )
    propositions = []
    facts = []
    for source in case["input"]["facts"]:
        fact_id = _id("f", f"{case['case_id']}:{source['fact_id']}")
        proposition_id = _id("p", f"{case['case_id']}:{source['fact_id']}")
        propositions.append(
            {
                "proposition_id": proposition_id,
                "fact_id": fact_id,
                "text": source["text"],
                "capability_group": group_by_fact[source["fact_id"]],
            }
        )
        facts.append(
            {
                "fact_id": fact_id,
                "proposition_ids": [proposition_id],
                "source_refs": source["source_refs"],
                "evidence_refs": [],
                "asset_refs": [
                    asset_id
                    for role in source["asset_roles"]
                    for asset_id in asset_ids_by_role.get(role, [])
                ],
                "reference_roles": source["asset_roles"],
                "availability": source["availability"],
                "support_level": source["support_level"],
                "collection_state": source["collection_state"],
            }
        )
    return {
        "schema_version": "media-facts-v1",
        "brief_id": case["case_id"],
        "approved_revision": case["input"]["worker_revision"],
        "brief_digest": "sha256:" + "1" * 64,
        "worker_projection_digest": "sha256:" + "2" * 64,
        "propositions": propositions,
        "facts": facts,
        "sources": [],
        "evidence": [],
        "assets": assets,
        "ignored_fact_ids": [],
    }


def _cases(name: str) -> list[dict[str, Any]]:
    return json.loads((DATA_ROOT / name).read_text(encoding="utf-8"))


def test_product_variant_plans_match_activation_and_safety_contracts() -> None:
    repository = DataRepository()
    profile = repository.get_media_profile("robotic-floor-cleaner-v1")
    planner = MediaPlanner(repository=repository)

    for case in _cases("product-variant-cases.json"):
        plan = planner.plan(
            media_facts=_media_facts_from_case(case),
            template=repository.get_template(case["template_id"]),
            profile=profile,
        )
        expected = case["expected"]
        assert plan["decision"] == expected["decision"], case["case_id"]
        assert set(plan["active_groups"]) == set(expected["active_groups"]), case[
            "case_id"
        ]
        assert set(plan["inactive_groups"]) == set(expected["inactive_groups"]), case[
            "case_id"
        ]
        assert plan["missing_reference_roles"] == [], case["case_id"]
        assert 4 <= len(plan["slots"]) <= 8, case["case_id"]


def test_defensive_plans_match_non_conflict_decisions_and_group_states() -> None:
    repository = DataRepository()
    profile = repository.get_media_profile("robotic-floor-cleaner-v1")
    planner = MediaPlanner(repository=repository)

    for case in _cases("defensive-cases.json"):
        if case["expected"]["decision"] == "reject_conflict":
            with pytest.raises(DataValidationError, match="conflicting"):
                planner.plan(
                    media_facts=_media_facts_from_case(case),
                    template=repository.get_template(case["template_id"]),
                    profile=profile,
                )
            continue
        plan = planner.plan(
            media_facts=_media_facts_from_case(case),
            template=repository.get_template(case["template_id"]),
            profile=profile,
        )
        expected = case["expected"]
        assert plan["decision"] == expected["decision"], case["case_id"]
        assert set(plan["active_groups"]) == set(expected["active_groups"]), case[
            "case_id"
        ]
        assert set(plan["placeholder_groups"]) == set(
            expected["placeholder_groups"]
        ), case["case_id"]
        assert set(plan["inactive_groups"]) == set(expected["inactive_groups"]), case[
            "case_id"
        ]
        assert set(plan["missing_reference_roles"]) == set(
            expected["missing_reference_roles"]
        ), case["case_id"]


@dataclass
class _Result:
    data: dict[str, Any]


class _SceneAdapter:
    def __init__(
        self, *, slots: list[dict[str, Any]] | None = None, invented_number: bool = False
    ) -> None:
        self.slots = slots or []
        self.invented_number = invented_number
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, *, prompt: str, response_schema: dict[str, Any]) -> _Result:
        self.calls.append({"prompt": prompt, "schema": response_schema})
        slot_ids = response_schema["properties"]["scenes"]["items"]["properties"][
            "slot_id"
        ]["enum"]
        return _Result(
            {
                "scenes": [
                    {
                        "slot_id": slot_id,
                        "grounding_fact_ids": next(
                            slot["fact_ids"]
                            for slot in self.slots
                            if slot["slot_id"] == slot_id
                        ),
                        "reference_asset_ids": next(
                            slot["reference_asset_ids"]
                            for slot in self.slots
                            if slot["slot_id"] == slot_id
                        ),
                        "visual_direction": (
                            "9999단계의 서로 다른 구도"
                            if self.invented_number
                            else "서로 다른 구도"
                        ),
                        "text_policy": "allowed_grounded_only",
                    }
                    for slot_id in slot_ids
                ]
            }
        )


def test_scene_adapter_must_cover_slots_without_new_numbers() -> None:
    repository = DataRepository()
    case = _cases("product-variant-cases.json")[0]
    profile = repository.get_media_profile("robotic-floor-cleaner-v1")
    base_slots = MediaPlanner(repository=repository).plan(
        media_facts=_media_facts_from_case(case),
        template=repository.get_template(case["template_id"]),
        profile=profile,
    )["slots"]
    adapter = _SceneAdapter(slots=base_slots)

    plan = MediaPlanner(repository=repository, scene_adapter=adapter).plan(
        media_facts=_media_facts_from_case(case),
        template=repository.get_template(case["template_id"]),
        profile=profile,
    )

    assert adapter.calls
    assert {slot["scene"]["visual_direction"] for slot in plan["slots"]} == {
        "서로 다른 구도"
    }

    with pytest.raises(MediaPlanningError, match="numeric claim"):
        MediaPlanner(
            repository=repository,
            scene_adapter=_SceneAdapter(slots=base_slots, invented_number=True),
        ).plan(
            media_facts=_media_facts_from_case(case),
            template=repository.get_template(case["template_id"]),
            profile=profile,
        )
