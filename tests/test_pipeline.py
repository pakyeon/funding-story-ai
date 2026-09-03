from collections.abc import Iterable

from funding_story_ai.adapter import GenerationResult
from funding_story_ai.data_repository import DataRepository
from funding_story_ai.pipeline import StoryPipeline


def _content(repository: DataRepository, *, unsupported_number: bool = False) -> dict:
    template = repository.compose_template(
        template_id="t02_problem_solution_automation",
        brief=repository.load_brief("robot-vacuum/brief.json"),
    )
    sections = []
    for index, section in enumerate(template["layout"]):
        body = "클린포지 R1의 입력 사실과 미확인 항목을 구분합니다."
        if unsupported_number and index == 0:
            body = "근거 없이 99% 만족도를 제시합니다."
        sections.append(
            {
                "template_section_id": section["id"],
                "type": section["type"],
                "heading": section["label"],
                "body": body,
                "source_fields": ["product.name"],
                "image_intent": {
                    "required": section["image_required"],
                    "purpose": "제품 외형과 사용 맥락 제시"
                    if section["image_required"]
                    else "",
                    "visual_hint": section["visual_hint"]
                    if section["image_required"]
                    else "",
                    "source_fields": ["asset_product_hero"]
                    if section["image_required"]
                    else [],
                },
            }
        )
    return {
        "title_candidates": ["클린포지 R1, 청소 후 관리까지 한 흐름으로"],
        "sections": sections,
    }


class FakeAdapter:
    def __init__(self, outcomes: Iterable[dict]) -> None:
        self.outcomes = iter(outcomes)
        self.prompts: list[str] = []

    def generate_json(self, *, prompt: str, response_schema: dict) -> GenerationResult:
        self.prompts.append(prompt)
        return GenerationResult(
            model="gemini-3.8-flash",
            data=next(self.outcomes),
        )


def test_pipeline_selects_t02_and_builds_valid_result() -> None:
    repository = DataRepository()
    adapter = FakeAdapter([_content(repository)])
    result = StoryPipeline(repository=repository, adapter=adapter).invoke(
        repository.load_brief("robot-vacuum/brief.json")
    )

    assert result["template_id"] == "t02_problem_solution_automation"
    assert result["automated_validation_passed"] is True
    assert result["review_required"] is True
    assert result["warnings"] == []
    assert len(result["sections"]) == 10


def test_pipeline_records_validation_warning_without_full_regeneration() -> None:
    repository = DataRepository()
    adapter = FakeAdapter(
        [_content(repository, unsupported_number=True), _content(repository)]
    )
    result = StoryPipeline(repository=repository, adapter=adapter).invoke(
        repository.load_brief("robot-vacuum/brief.json")
    )

    assert len(adapter.prompts) == 1
    assert result["automated_validation_passed"] is False
    assert any(item["code"] == "unlisted-number" for item in result["warnings"])
    assert result["review_required"] is True
