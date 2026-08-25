import json

from funding_story_ai.data_repository import DataRepository
from funding_story_ai.image_generation import ImageResult, ImageSettings
from funding_story_ai.image_pipeline import generate_section_images, planned_image_sections
from funding_story_ai.preview import render_story_html


def _story(repository: DataRepository) -> dict:
    template = repository.get_template("t02_problem_solution_automation")
    sections = [
        {
            "template_section_id": section["id"],
            "type": section["type"],
            "heading": section["label"],
            "body": "클린포지 R1 입력 사실을 설명합니다.",
            "source_fields": ["product.name"],
            "image_intent": {
                "required": section["image_required"],
                "purpose": "제품 시각화" if section["image_required"] else "",
                "visual_hint": section["visual_hint"] if section["image_required"] else "",
                "source_fields": ["asset_product_hero"]
                if section["image_required"]
                else [],
            },
        }
        for section in template["layout"]
    ]
    return {
        "schema_version": "story-result-v1",
        "language": "ko",
        "template_id": template["id"],
        "template_version": "0.1.0",
        "model": "gemini-3.7-flash",
        "title_candidates": ["클린포지 R1"],
        "sections": sections,
        "warnings": [],
        "automated_validation_passed": True,
        "review_required": True,
    }


class FakeImageAdapter:
    def edit_reference(self, *, section_id, reference_path, prompt):
        return ImageResult(
            section_id=section_id,
            image_bytes=f"image-{section_id}".encode(),
            revised_prompt=None,
        )

    def generate_text(self, *, section_id, prompt):
        return ImageResult(
            section_id=section_id,
            image_bytes=f"generated-{section_id}".encode(),
            revised_prompt=None,
        )


def test_t02_template_plans_its_five_required_images() -> None:
    repository = DataRepository()
    story = _story(repository)
    plans = planned_image_sections(story, repository.get_template(story["template_id"]))
    assert [plan["section_id"] for plan in plans] == [
        "hero",
        "solution",
        "features",
        "social_proof",
        "timeline",
    ]


def test_template_image_count_is_driven_by_the_selected_layout() -> None:
    repository = DataRepository()
    template = repository.get_template("t04_full_campaign")
    story = {
        "sections": [
            {
                "template_section_id": section["id"],
                "image_intent": {
                    "required": section["image_required"],
                    "purpose": "제품 시각화" if section["image_required"] else "",
                },
            }
            for section in template["layout"]
        ]
    }

    plans = planned_image_sections(story, template)

    assert [plan["section_id"] for plan in plans] == [
        "hero", "solution", "features", "funding_plan", "timeline", "team"
    ]
    features_prompt = next(plan["prompt"] for plan in plans if plan["section_id"] == "features")
    assert "입력에 없는 추가 구성품을 넣지 않음" in features_prompt


def test_solution_prompt_keeps_metrics_out_of_raster_image() -> None:
    repository = DataRepository()
    story = _story(repository)
    story["sections"][4]["body"] = "최대 8,000Pa와 180분을 보여 줍니다."
    plans = planned_image_sections(
        story,
        repository.get_template(story["template_id"]),
        {"solution"},
    )
    assert len(plans) == 1
    assert "8,000" not in plans[0]["prompt"]
    assert "타이포그래피를 만들지 않음" in plans[0]["prompt"]


def test_generate_images_writes_valid_manifest_and_preview(tmp_path) -> None:
    repository = DataRepository()
    story = _story(repository)
    story_path = tmp_path / "story.json"
    story_path.write_text(json.dumps(story, ensure_ascii=False), encoding="utf-8")
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(b"reference")
    output = tmp_path / "preview-run"
    settings = ImageSettings()

    manifest = generate_section_images(
        story_path=story_path,
        reference_path=reference,
        output_dir=output,
        repository=repository,
        adapter=FakeImageAdapter(),
        settings=settings,
    )

    assert manifest["requested"] == 5
    assert manifest["succeeded"] == 5
    assert manifest["failed"] == 0
    assert {asset["qa_status"] for asset in manifest["assets"]} == {"pending"}
    repository.validate_story_image_manifest(manifest)

    html = render_story_html(
        story=story,
        template=repository.get_template(story["template_id"]),
        manifest=manifest,
        fallback_image="reference.jpg",
    )
    assert "본문 HTML 복사" in html
    assert 'src="hero.jpeg"' in html
    assert "사람 검토 대기" in html
    manifest["assets"][0]["qa_status"] = "pass"
    html_after_qa = render_story_html(
        story=story,
        template=repository.get_template(story["template_id"]),
        manifest=manifest,
        fallback_image="reference.jpg",
    )
    assert 'src="hero.jpeg"' in html_after_qa
    assert "검토 필수" in html


def test_generate_images_accepts_text_only_input_without_reference(tmp_path) -> None:
    repository = DataRepository()
    story = _story(repository)
    story_path = tmp_path / "story.json"
    story_path.write_text(json.dumps(story, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "text-only-run"

    manifest = generate_section_images(
        story_path=story_path,
        reference_path=None,
        output_dir=output,
        repository=repository,
        adapter=FakeImageAdapter(),
        settings=ImageSettings(),
    )

    assert manifest["reference_sha256"] is None
    assert manifest["input_mode"] == "text-seeded-edit"
    assert manifest["generated_seed_sha256"] is not None
    assert manifest["succeeded"] == 5
    assert (output / "hero.jpeg").read_bytes() == b"generated-hero"
    repository.validate_story_image_manifest(manifest)


def test_text_only_prompt_does_not_claim_a_reference_image() -> None:
    repository = DataRepository()
    story = _story(repository)
    plans = planned_image_sections(
        story,
        repository.get_template(story["template_id"]),
        reference_available=False,
    )

    assert all("참조 이미지" not in plan["prompt"] for plan in plans)
    assert all("입력 제품 설명" in plan["prompt"] for plan in plans)


def test_preview_escapes_generated_markup() -> None:
    repository = DataRepository()
    story = _story(repository)
    story["sections"][0]["body"] = '<script>alert("x")</script>'
    rendered = render_story_html(
        story=story,
        template=repository.get_template(story["template_id"]),
        fallback_image="reference.jpg",
    )
    assert "&lt;script&gt;" in rendered
    assert '<script>alert("x")</script>' not in rendered


def test_preview_renders_safe_structured_markdown_blocks() -> None:
    repository = DataRepository()
    story = _story(repository)
    story["sections"][0]["body"] = (
        "**핵심 성능**과 ==시험 조건==, *검토 안내*\n\n"
        "- 입력 사실\n- 미확인 정보\n\n"
        "| 항목 | 상태 |\n| --- | --- |\n| 인증 | 미입력 |\n\n"
        "> 검토가 필요합니다."
    )

    rendered = render_story_html(
        story=story,
        template=repository.get_template(story["template_id"]),
        fallback_image="reference.jpg",
    )

    assert "<strong>핵심 성능</strong>" in rendered
    assert "<mark>시험 조건</mark>" in rendered
    assert "<em>검토 안내</em>" in rendered
    assert "<ul><li>입력 사실</li><li>미확인 정보</li></ul>" in rendered
    assert "<table><thead>" in rendered
    assert "<blockquote>검토가 필요합니다.</blockquote>" in rendered
