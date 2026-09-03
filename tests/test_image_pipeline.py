import hashlib
from copy import deepcopy

import pytest

from funding_story_ai.data_repository import DataRepository
from funding_story_ai.image_generation import ImageResult, ImageSettings
from funding_story_ai.image_pipeline import (
    build_slot_image_prompt,
    generate_planned_images,
    planned_image_slots,
)
from funding_story_ai.media_projection import build_approved_generation_package
from funding_story_ai.preview import (
    can_render_publishable,
    render_funding_story_html,
)


def _package(repository: DataRepository, reference) -> dict:
    return build_approved_generation_package(
        repository=repository,
        input_id="image-pipeline-test",
        thread_id="image-pipeline-thread",
        state={
            "workflow_stage": "generation-ready",
            "summary_version": 1,
            "approved_summary_version": 1,
            "facts_revision": 1,
            "collection_revision": 1,
            "facts": {},
        },
        brief=repository.load_brief(),
        local_asset_paths={"asset_product_hero": reference},
    )


def _facts_and_plan(package: dict) -> tuple[dict, dict]:
    group_by_kind = {
        "product": "product_identity_outcome",
        "problem": "problem_environment",
        "feature": "cleaning_mechanism",
        "evidence": "evidence_performance",
    }
    selected = {}
    for fact in package["entity_projection"]["facts"]:
        if fact["entity_kind"] in group_by_kind and fact["entity_kind"] not in selected:
            selected[fact["entity_kind"]] = fact
    propositions = []
    facts = []
    for kind, group in group_by_kind.items():
        source = selected[kind]
        proposition_id = "p_" + hashlib.sha256(source["fact_id"].encode()).hexdigest()[:12]
        propositions.append(
            {
                "proposition_id": proposition_id,
                "fact_id": source["fact_id"],
                "text": source["statement"],
                "capability_group": group,
            }
        )
        state = next(
            item
            for item in package["worker_projection"]["fact_states"]
            if item["fact_id"] == source["fact_id"]
        )
        facts.append(
            {
                "fact_id": source["fact_id"],
                "proposition_ids": [proposition_id],
                "source_refs": source["source_refs"],
                "evidence_refs": source["evidence_refs"],
                "asset_refs": source["asset_refs"],
                "reference_roles": source["reference_roles"],
                "availability": state["availability"],
                "support_level": state["support_level"],
                "collection_state": state["collection_state"],
            }
        )
    media_facts = {
        "schema_version": "media-facts-v1",
        "brief_id": package["brief"]["brief_id"],
        "approved_revision": 1,
        "brief_digest": package["brief_digest"],
        "worker_projection_digest": package["worker_projection_digest"],
        "propositions": propositions,
        "facts": facts,
        "sources": package["entity_projection"]["sources"],
        "evidence": package["entity_projection"]["evidence"],
        "assets": package["entity_projection"]["assets"],
        "ignored_fact_ids": [],
    }
    section_by_group = {
        "product_identity_outcome": "introduction",
        "problem_environment": "problem_context",
        "cleaning_mechanism": "cleaning_performance",
        "evidence_performance": "trust",
    }
    slots = []
    for proposition, fact in zip(propositions, facts, strict=True):
        group = proposition["capability_group"]
        refs = (
            fact["asset_refs"]
            if group in {"product_identity_outcome", "cleaning_mechanism"}
            else []
        )
        slots.append(
            {
                "slot_id": f"slot_{group}_01",
                "capability_group": group,
                "grouping_key": group,
                "section_id": section_by_group[group],
                "persuasion_goal": "승인 사실 전달",
                "priority": "required",
                "placement": "section_lead",
                "media_kind": "static",
                "reference_policy": "required" if refs else "none",
                "reference_asset_ids": refs,
                "fact_ids": [fact["fact_id"]],
                "proposition_ids": [proposition["proposition_id"]],
                "scene": {
                    "summary": proposition["text"],
                    "visual_direction": "각 슬롯마다 서로 다른 가로 장면",
                    "text_policy": "allowed_grounded_only",
                },
            }
        )
    plan = {
        "schema_version": "media-plan-v1",
        "brief_id": media_facts["brief_id"],
        "template_id": "t02_problem_solution_automation",
        "media_profile_id": "robotic-floor-cleaner-v1",
        "decision": "ready",
        "publishable": True,
        "generation_allowed": True,
        "active_groups": list(group_by_kind.values()),
        "inactive_groups": [],
        "placeholder_groups": [],
        "missing_reference_roles": [],
        "slots": slots,
        "placeholders": [],
        "reasons": ["검증 가능"],
    }
    return media_facts, plan


class _Images:
    def __init__(self, fail_slot: str | None = None) -> None:
        self.fail_slot = fail_slot
        self.calls = []

    def generate(self, *, slot_id, prompt, reference_paths=()):
        self.calls.append((slot_id, tuple(reference_paths), prompt))
        if slot_id == self.fail_slot:
            raise RuntimeError("image failure")
        return ImageResult(
            slot_id=slot_id,
            image_bytes=f"image-{slot_id}".encode(),
            model="gemini-image-test",
        )


def _story(repository: DataRepository) -> dict:
    template = repository.compose_template(
        template_id="t02_problem_solution_automation",
        brief=repository.load_brief(),
    )
    return {
        "title_candidates": ["클린포지 R1"],
        "sections": [
            {
                "template_section_id": section["id"],
                "heading": section["label"],
                "body": "**승인된 사실**을 설명합니다.",
            }
            for section in template["layout"]
        ],
    }


@pytest.mark.parametrize("image_count", [4, 6, 8])
def test_plan_drives_bounded_independent_images_without_generated_chaining(
    tmp_path, image_count: int
) -> None:
    repository = DataRepository()
    reference = tmp_path / "product.jpg"
    reference.write_bytes(b"reference")
    package = _package(repository, reference)
    facts, plan = _facts_and_plan(package)
    while len(plan["slots"]) < image_count:
        slot = deepcopy(plan["slots"][0])
        slot["slot_id"] = f"slot_product_identity_outcome_{len(plan['slots']) + 1:02d}"
        plan["slots"].append(slot)
    images = _Images()

    manifest = generate_planned_images(
        media_plan=plan,
        media_facts=facts,
        generation_package=package,
        output_dir=tmp_path / "images",
        repository=repository,
        adapter=images,
        settings=ImageSettings(),
        run_id="run-image-test",
    )

    assert manifest["requested"] == image_count
    assert manifest["succeeded"] == image_count
    assert {asset["qa_status"] for asset in manifest["assets"]} == {"pending"}
    assert all("review_checks" in asset for asset in manifest["assets"])
    assert images.calls[1][1] == ()
    assert all(path == reference for _, paths, _ in images.calls for path in paths)
    repository.validate_story_image_manifest(manifest)


def test_prompt_allows_only_grounded_text_and_unknown_slot_is_rejected(tmp_path) -> None:
    repository = DataRepository()
    reference = tmp_path / "product.jpg"
    reference.write_bytes(b"reference")
    package = _package(repository, reference)
    facts, plan = _facts_and_plan(package)

    prompt = build_slot_image_prompt(
        slot=plan["slots"][0], media_facts=facts, reference_available=True
    )
    assert "이미지 내 문자는 허용" in prompt
    assert facts["propositions"][0]["text"] in prompt
    with pytest.raises(ValueError, match="Unknown media slot"):
        planned_image_slots(plan, slot_ids={"slot_unknown_01"})


def test_image_failure_is_isolated_in_manifest(tmp_path) -> None:
    repository = DataRepository()
    reference = tmp_path / "product.jpg"
    reference.write_bytes(b"reference")
    package = _package(repository, reference)
    facts, plan = _facts_and_plan(package)
    manifest = generate_planned_images(
        media_plan=plan,
        media_facts=facts,
        generation_package=package,
        output_dir=tmp_path / "images",
        repository=repository,
        adapter=_Images(fail_slot=plan["slots"][0]["slot_id"]),
        settings=ImageSettings(),
    )
    assert manifest["failed"] == 1
    assert manifest["succeeded"] == 3
    failed = next(asset for asset in manifest["assets"] if asset["status"] == "error")
    assert failed["error_category"] == "unknown"
    assert failed["error_message"] == "image failure"
    assert failed["attempts"] == 1
    assert failed["attempt_history"] == []


def test_draft_is_pure_740px_html_and_publishable_requires_review(tmp_path) -> None:
    repository = DataRepository()
    reference = tmp_path / "product.jpg"
    reference.write_bytes(b"reference")
    package = _package(repository, reference)
    facts, plan = _facts_and_plan(package)
    manifest = generate_planned_images(
        media_plan=plan,
        media_facts=facts,
        generation_package=package,
        output_dir=tmp_path / "images",
        repository=repository,
        adapter=_Images(),
        settings=ImageSettings(),
    )
    for asset in manifest["assets"]:
        asset["path"] = f"images/{asset['path']}"
    story = _story(repository)
    template = repository.compose_template(
        template_id="t02_problem_solution_automation",
        brief=repository.load_brief(),
    )

    draft = render_funding_story_html(
        story=story, template=template, media_plan=plan, manifest=manifest
    )
    assert "max-width:740px" in draft
    assert "@media(max-width:768px)" in draft
    assert "toolbar" not in draft
    assert "출처 필드" not in draft
    assert "&lt;script&gt;" in render_funding_story_html(
        story={**story, "sections": [{**story["sections"][0], "body": "<script>x</script>"}]},
        template=template,
        media_plan={**plan, "slots": []},
        manifest={**manifest, "assets": [], "requested": 0, "succeeded": 0},
    )
    assert can_render_publishable(media_plan=plan, manifest=manifest) is False
    with pytest.raises(ValueError, match="Publishable HTML"):
        render_funding_story_html(
            story=story,
            template=template,
            media_plan=plan,
            manifest=manifest,
            mode="publishable",
        )
    for asset in manifest["assets"]:
        asset["qa_status"] = "pass"
    assert can_render_publishable(media_plan=plan, manifest=manifest) is True
    published = render_funding_story_html(
        story=story,
        template=template,
        media_plan=plan,
        manifest=manifest,
        mode="publishable",
    )
    assert 'data-render-mode="publishable"' in published


def test_missing_optional_content_renders_clearly_labeled_fixed_example(tmp_path) -> None:
    repository = DataRepository()
    reference = tmp_path / "product.jpg"
    reference.write_bytes(b"reference")
    package = _package(repository, reference)
    facts, plan = _facts_and_plan(package)
    manifest = generate_planned_images(
        media_plan=plan,
        media_facts=facts,
        generation_package=package,
        output_dir=tmp_path / "images",
        repository=repository,
        adapter=_Images(),
        settings=ImageSettings(),
    )
    for asset in manifest["assets"]:
        asset["qa_status"] = "pass"
        asset["path"] = f"images/{asset['path']}"
    brief = deepcopy(repository.load_brief())
    brief["unknowns"] = [
        {
            "field": "funding_plan",
            "question": "펀딩금 사용 계획이 무엇인가요?",
            "blocks_sections": ["participation"],
        }
    ]
    template = repository.compose_template(
        template_id="t02_problem_solution_automation",
        brief=brief,
    )
    story = _story(repository)

    draft = render_funding_story_html(
        story=story,
        template=template,
        media_plan=plan,
        manifest=manifest,
        brief=brief,
    )

    assert "작성 예시 · 실제 정보 아님" in draft
    assert "참여에 필요한 정보를 작성해 주세요" in draft
    assert "실제 메이커 정보로 교체" in draft
    assert (
        can_render_publishable(
            media_plan=plan,
            manifest=manifest,
            story=story,
            brief=brief,
            template=template,
        )
        is False
    )
