import hashlib
from typing import Any

from funding_story_ai.data_repository import DataRepository
from funding_story_ai.engine import (
    IntegratedStoryMakerExecutor,
    StoryExecutionInput,
    StoryMakerExecutor,
    review_integrated_story_run,
)
from funding_story_ai.image_generation import ImageResult, ImageSettings
from funding_story_ai.media_projection import build_approved_generation_package
from funding_story_ai.run_store import LocalRunStore


def _generation_package(repository: DataRepository, reference=None):
    return build_approved_generation_package(
        repository=repository,
        input_id="engine-test",
        thread_id="engine-thread",
        state={
            "workflow_stage": "generation-ready",
            "summary_version": 1,
            "approved_summary_version": 1,
            "facts_revision": 1,
            "collection_revision": 1,
            "facts": {},
        },
        brief=repository.load_brief(),
        local_asset_paths=(
            {"asset_product_hero": reference} if reference is not None else {}
        ),
    )


class _Pipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def invoke(self, brief: dict[str, Any], *, template_id: str | None = None):
        self.calls.append({"brief": brief, "template_id": template_id})
        return {"status": "complete"}


def test_story_maker_executor_has_no_mcp_dependency() -> None:
    repository = DataRepository()
    pipeline = _Pipeline()
    executor = StoryMakerExecutor(repository=repository, pipeline=pipeline)  # type: ignore[arg-type]
    brief = repository.load_brief()
    result = executor.execute(
        StoryExecutionInput(
            generation_package=_generation_package(repository),
            template_id="t02_problem_solution_automation",
        )
    )
    assert result == {"status": "complete"}
    assert pipeline.calls == [
        {"brief": brief, "template_id": "t02_problem_solution_automation"}
    ]


class _IntegratedPipeline:
    def __init__(self, repository: DataRepository, warnings=None) -> None:
        self.repository = repository
        self.warnings = warnings or []

    def invoke(self, brief, *, template_id=None):
        template = self.repository.get_template(template_id or "t04_full_campaign")
        return {
            "schema_version": "story-result-v1",
            "language": "ko",
            "template_id": template["id"],
            "template_version": "0.1.0",
            "model": "gemini-test",
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
                        "visual_hint": section["visual_hint"] if section["image_required"] else "",
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
        }


def _media_facts(package: dict) -> dict:
    fact = next(
        item for item in package["entity_projection"]["facts"] if item["entity_kind"] == "product"
    )
    proposition_id = "p_" + hashlib.sha256(fact["fact_id"].encode()).hexdigest()[:12]
    state = next(
        item
        for item in package["worker_projection"]["fact_states"]
        if item["fact_id"] == fact["fact_id"]
    )
    return {
        "schema_version": "media-facts-v1",
        "brief_id": package["brief"]["brief_id"],
        "approved_revision": 1,
        "brief_digest": package["brief_digest"],
        "worker_projection_digest": package["worker_projection_digest"],
        "propositions": [
            {
                "proposition_id": proposition_id,
                "fact_id": fact["fact_id"],
                "text": fact["statement"],
                "capability_group": "product_identity_outcome",
            }
        ],
        "facts": [
            {
                "fact_id": fact["fact_id"],
                "proposition_ids": [proposition_id],
                "source_refs": fact["source_refs"],
                "evidence_refs": fact["evidence_refs"],
                "asset_refs": fact["asset_refs"],
                "reference_roles": fact["reference_roles"],
                "availability": state["availability"],
                "support_level": state["support_level"],
                "collection_state": state["collection_state"],
            }
        ],
        "sources": package["entity_projection"]["sources"],
        "evidence": package["entity_projection"]["evidence"],
        "assets": [
            {
                **asset,
                "generation_available": asset["asset_id"]
                in package["local_asset_paths"],
            }
            for asset in package["entity_projection"]["assets"]
        ],
        "ignored_fact_ids": [],
    }


class _Normalizer:
    def normalize(self, package):
        return _media_facts(package)


class _Planner:
    def __init__(self, *, allow_generation=True) -> None:
        self.allow_generation = allow_generation

    def plan(self, *, media_facts, template, profile):
        proposition = media_facts["propositions"][0]
        fact = media_facts["facts"][0]
        available_assets = {
            asset["asset_id"]
            for asset in media_facts["assets"]
            if asset.get("generation_available", True)
        }
        allowed = self.allow_generation and bool(
            set(fact["asset_refs"]).intersection(available_assets)
        )
        return {
            "schema_version": "media-plan-v1",
            "brief_id": media_facts["brief_id"],
            "template_id": template["id"],
            "media_profile_id": profile["id"],
            "decision": "ready" if allowed else "needs_reference_assets",
            "publishable": allowed,
            "generation_allowed": allowed,
            "active_groups": ["product_identity_outcome"],
            "inactive_groups": [],
            "placeholder_groups": [],
            "missing_reference_roles": [] if allowed else ["product_body"],
            "slots": [
                {
                    "slot_id": "slot_product_identity_outcome_01",
                    "capability_group": "product_identity_outcome",
                    "grouping_key": "identity_outcome",
                    "section_id": "hero",
                    "persuasion_goal": "제품 정체성 전달",
                    "priority": "required",
                    "placement": "section_lead",
                    "media_kind": "static",
                    "reference_policy": "required",
                    "reference_asset_ids": fact["asset_refs"],
                    "fact_ids": [fact["fact_id"]],
                    "proposition_ids": [proposition["proposition_id"]],
                    "scene": {
                        "summary": proposition["text"],
                        "visual_direction": "제품 히어로 장면",
                        "text_policy": "allowed_grounded_only",
                    },
                }
            ],
            "placeholders": [],
            "reasons": ["테스트 계획"],
        }


class _Images:
    def __init__(self, fail=False) -> None:
        self.fail = fail

    def generate(self, *, slot_id, prompt, reference_paths=()):
        if self.fail:
            raise RuntimeError("image failure")
        return ImageResult(
            slot_id=slot_id,
            image_bytes=f"image-{slot_id}".encode(),
            model="gemini-image-test",
        )


def _integrated(repository, images, *, warnings=None, allow_generation=True):
    return IntegratedStoryMakerExecutor(
        repository=repository,
        pipeline=_IntegratedPipeline(repository, warnings=warnings),  # type: ignore[arg-type]
        semantic_normalizer=_Normalizer(),  # type: ignore[arg-type]
        media_planner=_Planner(allow_generation=allow_generation),  # type: ignore[arg-type]
        image_adapter=images,  # type: ignore[arg-type]
        image_settings=ImageSettings(),
    )


def test_integrated_executor_writes_plan_images_and_pure_draft_html(tmp_path) -> None:
    repository = DataRepository()
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(b"reference")
    run_dir = tmp_path / "run-integrated"
    result = _integrated(repository, _Images()).execute(
        StoryExecutionInput(
            generation_package=_generation_package(repository, reference),
            template_id="t04_full_campaign",
            run_id="run-integrated",
            output_dir=run_dir,
        )
    )
    assert result["status"] == "complete"
    assert result["images"]["requested"] == 1
    assert result["images"]["qa_pending"] == 1
    assert (run_dir / "media-facts.json").is_file()
    assert (run_dir / "media-plan.json").is_file()
    assert (run_dir / "draft.html").is_file()
    assert not (run_dir / "publishable.html").exists()
    assert 'src="images/slot_product_identity_outcome_01.jpeg"' in (
        run_dir / "draft.html"
    ).read_text()
    repository.validate_integrated_story_run(result)


def test_integrated_executor_isolates_image_failure_as_partial(tmp_path) -> None:
    repository = DataRepository()
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(b"reference")
    result = _integrated(repository, _Images(fail=True)).execute(
        StoryExecutionInput(
            generation_package=_generation_package(repository, reference),
            run_id="run-partial",
            output_dir=tmp_path / "run-partial",
        )
    )
    assert result["status"] == "partial"
    assert result["images"]["failed"] == 1


def test_integrated_executor_does_not_call_images_when_plan_blocks(tmp_path) -> None:
    repository = DataRepository()

    class _MustNotRun:
        def generate(self, **kwargs):
            raise AssertionError("image adapter must not run")

    result = _integrated(
        repository, _MustNotRun(), allow_generation=False
    ).execute(
        StoryExecutionInput(
            generation_package=_generation_package(repository),
            run_id="run-blocked",
            output_dir=tmp_path / "run-blocked",
        )
    )
    assert result["status"] == "partial"
    assert result["images"]["requested"] == 0
    assert "이미지 자리" in (tmp_path / "run-blocked" / "draft.html").read_text()


def test_integrated_executor_marks_story_warning_as_partial(tmp_path) -> None:
    repository = DataRepository()
    warning = {
        "code": "unsupported-generated-text",
        "message": "입력에 없는 동작",
        "section_id": "solution",
        "source_fields": ["product.name"],
    }
    result = _integrated(repository, _Images(), warnings=[warning]).execute(
        StoryExecutionInput(
            generation_package=_generation_package(repository),
            run_id="run-warning-partial",
            output_dir=tmp_path / "run-warning-partial",
        )
    )
    assert result["status"] == "partial"
    assert result["warning_count"] == 1
    assert result["publishable_html"] is None


def test_review_integrated_run_renders_publishable_html_after_all_checks_pass(tmp_path) -> None:
    repository = DataRepository()
    store = LocalRunStore(tmp_path / "runs")
    record, created = store.begin(
        caller_id="review-test",
        idempotency_key="review-test-key",
        request_payload={"brief_id": "engine-test"},
    )
    assert created is True
    run_id = record["run_id"]
    reference = tmp_path / "reference.jpg"
    reference.write_bytes(b"reference")
    result = _integrated(repository, _Images()).execute(
        StoryExecutionInput(
            generation_package=_generation_package(repository, reference),
            template_id="t04_full_campaign",
            run_id=run_id,
            output_dir=store.root / run_id,
        )
    )
    store.complete(run_id, result)
    review = review_integrated_story_run(
        repository=repository,
        store=store,
        run_id=run_id,
        reviews={
            "slot_product_identity_outcome_01": {
                "qa_status": "pass",
                "review_checks": {
                    "scene_distinctness": "pass",
                    "product_fidelity": "pass",
                    "text_legibility": "pass",
                    "claim_grounding": "pass",
                },
                "qa_notes": ["사람 검토 완료"],
            }
        },
    )
    assert review["publishable"] is True
    assert (store.root / run_id / "publishable.html").is_file()
    updated = review["run"]
    assert updated["result"]["images"]["qa_pending"] == 0
    assert updated["result"]["publishable_html"]["path"] == "publishable.html"
