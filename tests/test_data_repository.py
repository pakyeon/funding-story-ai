import json
import re

from funding_story_ai.data_repository import DataRepository


def test_all_schemas_are_valid() -> None:
    DataRepository().check_schemas()


def test_all_templates_validate_and_have_unique_sections() -> None:
    templates = DataRepository().load_templates()
    assert {template["id"] for template in templates} == {
        "t01_performance_value_evidence",
        "t02_problem_solution_automation",
        "t03_lifestyle_social_proof",
    }
    assert {template["id"]: len(template["layout"]) for template in templates} == {
        "t01_performance_value_evidence": 4,
        "t02_problem_solution_automation": 4,
        "t03_lifestyle_social_proof": 4,
    }
    for template in templates:
        assert "platform_choice" not in {section["id"] for section in template["layout"]}
        assert template["content_placeholders"]


def test_template_html_contains_only_placeholders_for_values() -> None:
    for template in DataRepository().load_templates():
        for section in template["layout"]:
            html = section["html_content"]
            assert "{{" in html and "}}" in html
            assert re.search(r"\d{3,}", html) is None


def test_catalog_matches_template_files() -> None:
    DataRepository().validate_catalog_links()


def test_retrieval_index_links_three_persuasion_templates() -> None:
    index = DataRepository().load_template_retrieval_index()
    executable = {
        item["executable_template_id"]
        for item in index["candidates"]
        if item["executable_template_id"] is not None
    }
    assert len(index["candidates"]) == 16
    assert len(executable) == 3


def test_robotic_floor_cleaner_module_composes_conditional_sections() -> None:
    repository = DataRepository()
    brief = repository.load_brief()
    composed = repository.compose_template(
        template_id="t02_problem_solution_automation",
        brief=brief,
    )

    assert composed["category_module_id"] == "robotic-floor-cleaner-v1"
    assert composed["media_profile_ref"] == "robotic-floor-cleaner-v1"
    assert [section["id"] for section in composed["layout"]] == [
        "introduction",
        "problem_context",
        "benefits_differentiation",
        "cleaning_performance",
        "space_response",
        "automation",
        "control",
        "trust",
        "maintenance",
        "participation",
    ]
    assert composed["media_section_bindings"]["automation_return"] == "automation"


def test_category_module_links_are_schema_and_section_compatible() -> None:
    repository = DataRepository()
    modules = repository.load_category_modules()
    assert [module["id"] for module in modules] == ["robotic-floor-cleaner-v1"]
    repository.validate_template_media_profile_links()


def test_synthetic_example_is_schema_valid() -> None:
    brief = DataRepository().load_brief()
    assert brief["source"]["purpose"] == "synthetic-fixture"
    assert brief["product"]["name"] == "클린포지 R1"
    assert all(source["source_type"] != "public-page" for source in brief["source"]["refs"])


def test_external_brief_path_uses_the_same_validation(tmp_path) -> None:
    repository = DataRepository()
    brief = repository.load_brief()
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(brief, ensure_ascii=False), encoding="utf-8")
    assert repository.load_brief_path(path)["brief_id"] == brief["brief_id"]


def test_product_agnostic_semantic_state_accepts_explicit_absence() -> None:
    slot_names = (
        "product_name",
        "product_type",
        "category",
        "key_strengths",
        "target_supporters",
        "problem_context",
        "trust_elements",
        "maker_team_intro",
        "rewards",
        "funding_end",
        "shipping_start",
        "refund_policy",
        "as_policy",
        "funding_plan",
        "risk_response",
    )
    slots = {
        name: {"status": "unknown", "values": [], "source_turn": "none"} for name in slot_names
    }
    slots.update(
        {
            "product_name": {
                "status": "provided",
                "values": ["클린포지 R1"],
                "source_turn": "initial",
            },
            "product_type": {
                "status": "provided",
                "values": ["로봇청소기"],
                "source_turn": "initial",
            },
            "category": {"status": "provided", "values": ["테크·가전"], "source_turn": "initial"},
            "key_strengths": {
                "status": "provided",
                "values": ["8,000Pa", "올인원 도크"],
                "source_turn": "initial",
            },
            "target_supporters": {
                "status": "provided",
                "values": ["반복 청소 부담을 줄이려는 가구"],
                "source_turn": "initial",
            },
            "trust_elements": {
                "status": "explicitly-absent",
                "values": [],
                "source_turn": "initial",
            },
        }
    )
    value = {
        "schema_version": "story-intake-semantic-state-v2",
        "input_id": "robot-vacuum-example",
        "language": "ko",
        "image_attached": True,
        "slots": slots,
        "fact_conflict": {
            "status": "none",
            "authoritative_values": [],
            "superseded_values": [],
        },
        "decision": {
            "ready_to_confirm": True,
            "requested_fields": [],
            "follow_up_question": None,
        },
    }
    DataRepository().validate_intake_semantic_state(value)
