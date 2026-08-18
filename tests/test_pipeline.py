from collections.abc import Iterable

from funding_story_ai.adapter import GenerationResult
from funding_story_ai.data_repository import DataRepository
from funding_story_ai.pipeline import StoryPipeline
from funding_story_ai.pricing import TokenUsage


def _content(repository: DataRepository, *, unsupported_number: bool = False) -> dict:
    template = repository.get_template("t02_problem_solution_automation")
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
            request_id=f"request-{len(self.prompts)}",
            model="gemini-3.7-flash",
            data=next(self.outcomes),
            usage=TokenUsage(prompt_tokens=100, output_tokens=50, thinking_tokens=10),
            duration_ms=200,
            attempts=1,
            finish_reason="STOP",
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
    assert result["usage"]["model_calls"] == 1
    assert len(result["sections"]) == 12


def test_pipeline_corrects_once_and_aggregates_usage() -> None:
    repository = DataRepository()
    adapter = FakeAdapter(
        [_content(repository, unsupported_number=True), _content(repository)]
    )
    result = StoryPipeline(repository=repository, adapter=adapter).invoke(
        repository.load_brief("robot-vacuum/brief.json")
    )

    assert len(adapter.prompts) == 2
    assert "브리프에서 찾지 못한 수치" in adapter.prompts[1]
    assert result["automated_validation_passed"] is True
    assert result["review_required"] is True
    assert result["usage"] == {
        "prompt_tokens": 200,
        "output_tokens": 100,
        "thinking_tokens": 20,
        "duration_ms": 400,
        "attempts": 2,
        "model_calls": 2,
    }
