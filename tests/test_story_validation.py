from copy import deepcopy

import pytest

from funding_story_ai.data_repository import DataRepository
from funding_story_ai.validation import StoryValidator


def _valid_content(repository: DataRepository, template_id: str) -> dict:
    template = repository.get_template(template_id)
    return {
        "title_candidates": ["클린포지 R1, 청소 이후의 관리까지 한 흐름으로"],
        "sections": [
            {
                "template_section_id": section["id"],
                "type": section["type"],
                "heading": section["label"],
                "body": "클린포지 R1에 입력된 제품 정보와 미확인 항목을 구분합니다.",
                "source_fields": ["product.name"],
                "image_intent": {
                    "required": section["image_required"],
                    "purpose": "제품 외형과 사용 맥락 제시" if section["image_required"] else "",
                    "visual_hint": section["visual_hint"] if section["image_required"] else "",
                    "source_fields": ["asset_product_hero"]
                    if section["image_required"]
                    else [],
                },
            }
            for section in template["layout"]
        ],
    }


def test_valid_content_matches_t02_contract() -> None:
    repository = DataRepository()
    brief = repository.load_brief("robot-vacuum/brief.json")
    template = repository.get_template("t02_problem_solution_automation")
    content = _valid_content(repository, template["id"])

    repository.validate_story_generation_content(content)
    assert StoryValidator().validate(
        content=content, brief=brief, template=template
    ) == []


def test_validator_flags_unlisted_number_and_template_drift() -> None:
    repository = DataRepository()
    brief = repository.load_brief("robot-vacuum/brief.json")
    template = repository.get_template("t02_problem_solution_automation")
    content = deepcopy(_valid_content(repository, template["id"]))
    content["sections"][0]["body"] = "입력에 없는 99% 만족도를 주장합니다."
    content["sections"][0]["type"] = "offer"
    content["sections"][0]["image_intent"]["required"] = False

    codes = {
        warning.code
        for warning in StoryValidator().validate(
            content=content, brief=brief, template=template
        )
    }
    assert {"unlisted-number", "section-type-mismatch", "image-contract-mismatch"} <= codes


def test_validator_ignores_ordered_list_labels_but_keeps_claim_numbers() -> None:
    repository = DataRepository()
    brief = repository.load_brief("robot-vacuum/brief.json")
    template = repository.get_template("t02_problem_solution_automation")
    content = _valid_content(repository, template["id"])
    content["sections"][0]["body"] = (
        "1. 입력된 제품 정보\n"
        "2) 확인된 기능\n"
        "3. 근거 없이 99% 만족도를 주장"
    )

    warnings = StoryValidator().validate(
        content=content,
        brief=brief,
        template=template,
    )
    number_warnings = [warning for warning in warnings if warning.code == "unlisted-number"]

    assert len(number_warnings) == 1
    assert "99" in number_warnings[0].message
    assert all(label not in number_warnings[0].message for label in ("1", "2", "3"))


def test_validator_flags_promises_created_from_unknown_input() -> None:
    repository = DataRepository()
    brief = repository.load_brief("robot-vacuum/brief.json")
    template = repository.get_template("t02_problem_solution_automation")
    content = _valid_content(repository, template["id"])
    service = next(
        section for section in content["sections"] if section["template_section_id"] == "timeline"
    )
    service["body"] = "AS 정책은 현재 확인 중이며 추후 공지될 예정입니다."
    service["source_fields"] = ["unknown.as_and_refund_policy"]

    codes = {
        warning.code
        for warning in StoryValidator().validate(
            content=content, brief=brief, template=template
        )
    }
    assert "unsupported-future-commitment" in codes


@pytest.mark.parametrize(
    "future_copy",
    [
        "가격과 일정은 확정 후 추가 안내될 예정입니다.",
        "향후 업데이트될 리워드와 일정을 확인해 주세요.",
        "일정 확정 시 프로젝트 페이지에서 확인할 수 있습니다.",
    ],
)
def test_validator_flags_future_copy_variants(future_copy: str) -> None:
    repository = DataRepository()
    brief = repository.load_brief()
    template = repository.get_template("t02_problem_solution_automation")
    content = _valid_content(repository, template["id"])
    content["sections"][-1]["body"] = future_copy
    content["sections"][-1]["source_fields"] = ["unknown.reward_and_schedule"]
    warnings = StoryValidator().validate(
        content=content, brief=brief, template=template
    )
    assert any(warning.code == "unsupported-future-commitment" for warning in warnings)


def test_validator_flags_carpet_inference_without_carpet_input() -> None:
    repository = DataRepository()
    brief = repository.load_brief()
    brief["audiences"] = []
    brief["features"] = [
        feature
        for feature in brief["features"]
        if feature["id"] != "feature_carpet_lift"
    ]
    template = repository.get_template("t02_problem_solution_automation")
    content = _valid_content(repository, template["id"])
    content["sections"][2]["body"] = "12mm 리프팅으로 카펫 환경에 대응합니다."
    warnings = StoryValidator().validate(
        content=content, brief=brief, template=template
    )
    assert any(warning.code == "unsupported-generated-text" for warning in warnings)


def test_validator_flags_product_common_sense_and_concept_expansion() -> None:
    repository = DataRepository()
    brief = repository.load_brief("robot-vacuum/brief.json")
    template = repository.get_template("t02_problem_solution_automation")
    content = _valid_content(repository, template["id"])
    content["sections"][3]["body"] = (
        "전면 센서로 경로를 감지하고 도크로 자동 복귀합니다. "
        "정수통 채움과 오수통 비움, 먼지봉투 교체는 수동 관리가 필요합니다."
    )
    content["sections"][4]["body"] = "전용 모바일 앱을 지원합니다."
    content["sections"][7]["body"] = "개발팀은 앱 연동 경험을 갖췄습니다."

    warnings = StoryValidator().validate(
        content=content,
        brief=brief,
        template=template,
    )

    codes = [warning.code for warning in warnings]
    messages = [warning.message for warning in warnings]
    assert codes.count("unsupported-generated-text") >= 5
    assert "source-role-imprecision" in codes
    assert any("도크 자동 복귀" in message for message in messages)
    assert any("정수통 관리 방식" in message for message in messages)
    assert any("오수통 관리 방식" in message for message in messages)
    assert any("먼지봉투 관리 방식" in message for message in messages)
    assert any("앱 연동 경험" in message for message in messages)


def test_validator_does_not_treat_explicit_app_absence_as_new_app_claim() -> None:
    repository = DataRepository()
    brief = repository.load_brief()
    brief["product"]["summary"] += " 전용 앱 미지원"
    template = repository.get_template("t02_problem_solution_automation")
    content = _valid_content(repository, template["id"])
    content["sections"][0]["body"] = "전용 앱은 지원하지 않습니다."
    warnings = StoryValidator().validate(
        content=content, brief=brief, template=template
    )
    assert not any(
        warning.code == "unsupported-generated-text" and "전용 앱" in warning.message
        for warning in warnings
    )


def test_validator_requires_explicit_support_for_automatic_dust_emptying() -> None:
    repository = DataRepository()
    brief = repository.load_brief("robot-vacuum/brief.json")
    for feature in brief["features"]:
        if feature["id"] == "feature_all_in_one_dock":
            feature["description"] = feature["description"].replace("자동 먼지 비움", "먼지 비움")
    template = repository.get_template("t02_problem_solution_automation")
    content = _valid_content(repository, template["id"])
    content["sections"][0]["body"] = "도킹 스테이션에서 자동 먼지 비움을 지원합니다."

    warnings = StoryValidator().validate(
        content=content,
        brief=brief,
        template=template,
    )

    assert any("먼지 비움의 자동 동작" in warning.message for warning in warnings)


def test_validator_requires_explicit_support_for_post_cleaning_dock_timing() -> None:
    repository = DataRepository()
    brief = repository.load_brief()
    template = repository.get_template("t02_problem_solution_automation")
    content = _valid_content(repository, template["id"])
    content["sections"][2]["body"] = "청소 후 도크에서 먼지 비움과 충전을 진행합니다."
    warnings = StoryValidator().validate(
        content=content, brief=brief, template=template
    )
    assert any("청소 후 도크 동작 시점" in warning.message for warning in warnings)


def test_validator_flags_internal_unknown_identifier_in_prose() -> None:
    repository = DataRepository()
    brief = repository.load_brief("robot-vacuum/brief.json")
    template = repository.get_template("t02_problem_solution_automation")
    content = _valid_content(repository, template["id"])
    content["sections"][6]["body"] = (
        "AS 정책은 입력되지 않았습니다(unknown.as_and_refund_policy)."
    )
    content["sections"][6]["source_fields"] = ["unknown.as_and_refund_policy"]

    codes = {
        warning.code
        for warning in StoryValidator().validate(
            content=content,
            brief=brief,
            template=template,
        )
    }
    assert "internal-identifier-leak" in codes
