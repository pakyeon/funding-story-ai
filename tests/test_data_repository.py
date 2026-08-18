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
        "t04_full_campaign",
        "t05_value_practical_full_campaign",
        "t06_trust_maintenance_full_campaign",
    }
    assert {template["id"]: len(template["layout"]) for template in templates} == {
        "t01_performance_value_evidence": 12,
        "t02_problem_solution_automation": 12,
        "t03_lifestyle_social_proof": 12,
        "t04_full_campaign": 12,
        "t05_value_practical_full_campaign": 12,
        "t06_trust_maintenance_full_campaign": 12,
    }


def test_template_html_contains_only_placeholders_for_values() -> None:
    for template in DataRepository().load_templates():
        for section in template["layout"]:
            html = section["html_content"]
            assert "{{" in html and "}}" in html
            assert re.search(r"\d{3,}", html) is None


def test_catalog_matches_template_files() -> None:
    DataRepository().validate_catalog_links()


def test_retrieval_index_links_six_executable_templates() -> None:
    index = DataRepository().load_template_retrieval_index()
    executable = {
        item["executable_template_id"]
        for item in index["candidates"]
        if item["executable_template_id"] is not None
    }
    assert len(index["candidates"]) == 16
    assert len(executable) == 6


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


def test_robot_vacuum_profile_is_valid_and_domain_specific() -> None:
    repository = DataRepository()
    profile = repository.get_category_profile("robot-vacuum-ko-v1")
    assert profile["category"] == "테크·가전"
    assert "로봇청소기" in profile["match_terms"]
    assert set(profile["semantic_slot_guidance"]) == {
        "product_identity",
        "key_strengths",
        "target_supporters",
        "problem_context",
        "trust_elements",
        "maker_team_intro",
    }


def test_product_agnostic_semantic_state_accepts_explicit_absence() -> None:
    value = {
        "schema_version": "story-intake-semantic-state-v1",
        "input_id": "robot-vacuum-example",
        "profile_id": "robot-vacuum-ko-v1",
        "image_attached": True,
        "slots": {
            "product_identity": {
                "status": "provided",
                "values": ["클린포지 R1", "로봇청소기"],
                "source_turn": "initial",
            },
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
            "problem_context": {
                "status": "provided",
                "values": ["청소 후 관리 부담"],
                "source_turn": "initial",
            },
            "trust_elements": {
                "status": "explicitly-absent",
                "values": [],
                "source_turn": "initial",
            },
            "maker_team_intro": {
                "status": "explicitly-absent",
                "values": [],
                "source_turn": "initial",
            },
        },
        "fact_conflict": {
            "status": "none",
            "authoritative_values": [],
            "superseded_values": [],
        },
        "turn_state": {
            "primary_answered": False,
            "combined_answered": False,
            "secondary_answered": False,
            "skip_requested": False,
            "confirmed": False,
        },
    }
    DataRepository().validate_intake_semantic_state(value)
