from copy import deepcopy

import pytest

from funding_story_ai.data_repository import DataRepository, DataValidationError


def _approved_package(repository: DataRepository) -> dict:
    brief = repository.load_brief()
    digest = "sha256:" + "1" * 64
    projection_digest = "sha256:" + "2" * 64
    fact_id = "f_" + "a" * 12
    return {
        "schema_version": "approved-generation-package-v1",
        "input_id": "input-one",
        "thread_id": "thread-one",
        "approval": {
            "status": "approved",
            "summary_version": 1,
            "facts_revision": 2,
            "collection_revision": 1,
        },
        "brief": brief,
        "brief_digest": digest,
        "worker_projection": {
            "approval_status": "approved",
            "summary_version": 1,
            "facts_revision": 2,
            "collection_revision": 1,
            "brief_digest": digest,
            "fact_states": [
                {
                    "fact_id": fact_id,
                    "availability": "provided",
                    "support_level": "supported",
                    "collection_state": "resolved",
                    "revision": 2,
                }
            ],
        },
        "worker_projection_digest": projection_digest,
        "entity_projection": {
            "schema_version": "approved-entity-projection-v1",
            "brief_id": brief["brief_id"],
            "facts": [
                {
                    "fact_id": fact_id,
                    "entity_id": "feature-one",
                    "entity_kind": "feature",
                    "statement": "흡입과 물걸레 청소를 한 번의 경로에서 수행한다.",
                    "source_refs": ["source_maker_input"],
                    "evidence_refs": [],
                    "asset_refs": ["asset_product_hero"],
                    "reference_roles": ["product_body"],
                }
            ],
            "sources": [
                {
                    "source_id": "source_maker_input",
                    "source_type": "maker-input",
                    "description": "메이커가 승인한 입력",
                }
            ],
            "evidence": [],
            "assets": [
                {
                    "asset_id": "asset_product_hero",
                    "roles": ["product_body"],
                    "description": "제품 본체 참조 이미지",
                    "source_refs": ["source_product_image"],
                }
            ],
        },
        "entity_projection_digest": "sha256:" + "3" * 64,
        "local_asset_paths": {
            "asset_product_hero": "examples/robot-vacuum/product-reference.png"
        },
    }


def test_media_profiles_and_template_links_are_valid() -> None:
    repository = DataRepository()

    profiles = repository.load_media_profiles()
    repository.validate_template_media_profile_links()

    assert [profile["id"] for profile in profiles] == ["robotic-floor-cleaner-v1"]
    assert len(profiles[0]["capability_groups"]) == 8
    assert {
        template["media_profile_ref"] for template in repository.load_templates()
    } == {"robotic-floor-cleaner-v1"}


def test_media_profile_does_not_embed_example_product_identity() -> None:
    repository = DataRepository()
    serialized = str(repository.get_media_profile("robotic-floor-cleaner-v1"))

    assert "클린포지" not in serialized
    assert "Roomba" not in serialized
    assert "8,000Pa" not in serialized
    assert "LDS" not in serialized


def test_media_profile_rejects_inverted_cardinality() -> None:
    repository = DataRepository()
    profile = deepcopy(repository.get_media_profile("robotic-floor-cleaner-v1"))
    profile["capability_groups"][0]["cardinality"] = {"min": 2, "max": 1}

    with pytest.raises(DataValidationError, match="cardinality min exceeds max"):
        repository.validate_media_profile(profile)


def test_approved_generation_package_contract_validates_nested_brief() -> None:
    repository = DataRepository()
    value = _approved_package(repository)

    repository.validate_approved_generation_package(value)

    invalid = deepcopy(value)
    invalid["approval"]["status"] = "pending"
    with pytest.raises(DataValidationError, match="approval.status"):
        repository.validate_approved_generation_package(invalid)


def test_media_facts_and_plan_contracts_validate() -> None:
    repository = DataRepository()
    proposition_id = "p_" + "b" * 12
    fact_id = "f_" + "a" * 12
    media_facts = {
        "schema_version": "media-facts-v1",
        "brief_id": "brief-one",
        "approved_revision": 2,
        "brief_digest": "sha256:" + "1" * 64,
        "worker_projection_digest": "sha256:" + "2" * 64,
        "propositions": [
            {
                "proposition_id": proposition_id,
                "fact_id": fact_id,
                "text": "흡입과 물걸레 청소를 한 번의 경로에서 수행한다",
                "capability_group": "cleaning_mechanism",
            }
        ],
        "facts": [
            {
                "fact_id": fact_id,
                "proposition_ids": [proposition_id],
                "source_refs": ["source-one"],
                "evidence_refs": [],
                "asset_refs": ["asset-one"],
                "reference_roles": ["product_body"],
                "availability": "provided",
                "support_level": "supported",
                "collection_state": "resolved",
            }
        ],
        "sources": [],
        "evidence": [],
        "assets": [],
        "ignored_fact_ids": [],
    }
    repository.validate_media_facts(media_facts)

    plan = {
        "schema_version": "media-plan-v1",
        "brief_id": "brief-one",
        "template_id": "t02_problem_solution_automation",
        "media_profile_id": "robotic-floor-cleaner-v1",
        "decision": "ready",
        "publishable": True,
        "generation_allowed": True,
        "active_groups": ["cleaning_mechanism"],
        "inactive_groups": [],
        "placeholder_groups": [],
        "missing_reference_roles": [],
        "slots": [
                {
                    "slot_id": "slot_cleaning_mechanism_01",
                    "capability_group": "cleaning_mechanism",
                    "grouping_key": "cleaning_scene",
                "section_id": "solution",
                "persuasion_goal": "청소 방식을 설명한다.",
                "priority": "required",
                "placement": "inline",
                "media_kind": "static",
                "reference_policy": "required",
                "reference_asset_ids": ["asset-one"],
                "fact_ids": [fact_id],
                "proposition_ids": [proposition_id],
                "scene": {
                    "summary": "제품이 바닥을 청소하는 장면",
                    "visual_direction": "제품 외형을 유지한 생활 공간",
                    "text_policy": "allowed_grounded_only",
                },
            }
        ],
        "placeholders": [],
        "reasons": ["필수 정보와 참조 자산이 제공됨"],
    }
    repository.validate_media_plan(plan)

    too_many = deepcopy(plan)
    too_many["slots"] = too_many["slots"] * 9
    with pytest.raises(DataValidationError, match="too long"):
        repository.validate_media_plan(too_many)
