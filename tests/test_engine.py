from decimal import Decimal
from typing import Any

from funding_story_ai.data_repository import DataRepository
from funding_story_ai.engine import (
    IntegratedStoryMakerExecutor,
    StoryExecutionInput,
    StoryMakerExecutor,
)
from funding_story_ai.image_generation import ImageResult, ImageSettings, ImageUsage


class _Pipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(
        self,
        brief: dict[str, Any],
        *,
        template_id: str | None = None,
        category_profile_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "brief": brief,
                "template_id": template_id,
                "category_profile_id": category_profile_id,
            }
        )
        return {"status": "complete"}


def test_story_maker_executor_has_no_mcp_dependency() -> None:
    repository = DataRepository()
    pipeline = _Pipeline()
    executor = StoryMakerExecutor(repository=repository, pipeline=pipeline)  # type: ignore[arg-type]
    brief = repository.load_brief()

    result = executor.execute(
        StoryExecutionInput(
            brief=brief,
            template_id="t02_problem_solution_automation",
            category_profile_id="robot-vacuum-ko-v1",
        )
    )

    assert result == {"status": "complete"}
    assert pipeline.calls == [
        {
            "brief": brief,
            "template_id": "t02_problem_solution_automation",
            "category_profile_id": "robot-vacuum-ko-v1",
        }
    ]


class _IntegratedPipeline:
    def __init__(
        self, repository: DataRepository, warnings: list[dict[str, Any]] | None = None
    ) -> None:
        self.repository = repository
        self.warnings = warnings or []

    def invoke(self, brief, *, template_id=None, category_profile_id=None):
        template = self.repository.get_template(
            template_id or "t04_full_campaign"
        )
        return {
            "schema_version": "story-result-v1",
            "request_id": "integrated-test",
            "language": "ko",
            "template_id": template["id"],
            "template_version": "0.1.0",
            "model": "gemini-test",
            "prompt_version": "story-generation-v5",
            "title_candidates": ["통합 실행 테스트"],
            "sections": [
                {
                    "template_section_id": section["id"],
                    "type": section["type"],
                    "heading": section["label"],
                    "body": "입력 사실만 사용하는 테스트 본문입니다.",
                    "source_fields": ["product.name"],
                    "image_intent": {
                        "required": section["image_required"],
                        "purpose": "제품 외형" if section["image_required"] else "",
                        "visual_hint": (
                            section["visual_hint"] if section["image_required"] else ""
                        ),
                        "source_fields": (
                            ["asset_product_hero"] if section["image_required"] else []
                        ),
                    },
                }
                for section in template["layout"]
            ],
            "warnings": self.warnings,
            "automated_validation_passed": not self.warnings,
            "review_required": True,
            "usage": {
                "prompt_tokens": 1,
                "output_tokens": 1,
                "thinking_tokens": 0,
                "duration_ms": 1,
                "attempts": 1,
                "model_calls": 1,
            },
        }


class _ImageLedger:
    def __init__(self) -> None:
        self.reserves = []

    def assert_can_call(self, reserve):
        self.reserves.append(reserve)


class _IntegratedImages:
    def __init__(self, fail_section: str | None = None) -> None:
        self.ledger = _ImageLedger()
        self.fail_section = fail_section

    def edit_reference(self, *, section_id, reference_path, prompt):
        if section_id == self.fail_section:
            raise RuntimeError("image failure")
        return ImageResult(
            section_id=section_id,
            image_bytes=f"image-{section_id}".encode(),
            revised_prompt=None,
            duration_ms=2,
            usage=ImageUsage(1, 1, 1),
            estimated_cost_usd=Decimal("0.001"),
        )

    def generate_text(self, *, section_id, prompt):
        return self.edit_reference(
            section_id=section_id, reference_path=None, prompt=prompt
        )


def test_integrated_executor_links_story_images_and_html_under_one_run(tmp_path) -> None:
    repository = DataRepository()
    images = _IntegratedImages()
    settings = ImageSettings(reserve_usd_per_call=Decimal("0.5"))
    executor = IntegratedStoryMakerExecutor(
        repository=repository,
        pipeline=_IntegratedPipeline(repository),  # type: ignore[arg-type]
        image_adapter=images,  # type: ignore[arg-type]
        image_settings=settings,
    )
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(b"reference")
    run_dir = tmp_path / "run-integrated"
    result = executor.execute(
        StoryExecutionInput(
            brief=repository.load_brief(),
            template_id="t04_full_campaign",
            run_id="run-integrated",
            output_dir=run_dir,
            reference_image_path=reference,
        )
    )

    assert result["status"] == "complete"
    assert result["images"]["requested"] == 3
    assert result["images"]["succeeded"] == 3
    assert result["images"]["qa_pending"] == 3
    assert images.ledger.reserves[0] == Decimal("1.5")
    assert (run_dir / "story.json").is_file()
    assert (run_dir / "brief.json").is_file()
    assert (run_dir / "images" / "manifest.json").is_file()
    assert (run_dir / "preview.html").is_file()
    assert 'src="images/reference.jpg"' in (run_dir / "preview.html").read_text()
    repository.validate_integrated_story_run(result)
    assert result["input_brief"]["path"] == "brief.json"


def test_integrated_executor_isolates_one_image_failure_as_partial(tmp_path) -> None:
    repository = DataRepository()
    images = _IntegratedImages(fail_section="solution")
    executor = IntegratedStoryMakerExecutor(
        repository=repository,
        pipeline=_IntegratedPipeline(repository),  # type: ignore[arg-type]
        image_adapter=images,  # type: ignore[arg-type]
        image_settings=ImageSettings(),
    )
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(b"reference")
    result = executor.execute(
        StoryExecutionInput(
            brief=repository.load_brief(),
            template_id="t04_full_campaign",
            run_id="run-partial",
            output_dir=tmp_path / "run-partial",
            reference_image_path=reference,
        )
    )
    assert result["status"] == "partial"
    assert result["images"]["failed"] == 1
    assert result["review_required"] is True


def test_integrated_executor_marks_story_warning_as_partial(tmp_path) -> None:
    repository = DataRepository()
    images = _IntegratedImages()
    pipeline = _IntegratedPipeline(
        repository,
        warnings=[
            {
                "code": "unsupported-generated-text",
                "message": "입력에 없는 동작",
                "section_id": "solution",
                "source_fields": ["product.name"],
            }
        ],
    )
    executor = IntegratedStoryMakerExecutor(
        repository=repository,
        pipeline=pipeline,  # type: ignore[arg-type]
        image_adapter=images,  # type: ignore[arg-type]
        image_settings=ImageSettings(),
    )

    result = executor.execute(
        StoryExecutionInput(
            brief=repository.load_brief(),
            template_id="t04_full_campaign",
            run_id="run-warning-partial",
            output_dir=tmp_path / "run-warning-partial",
        )
    )

    assert result["status"] == "partial"
    assert result["warning_count"] == 1
