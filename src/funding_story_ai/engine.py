from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .data_repository import DataRepository
from .image_generation import ImageAdapter, ImageSettings
from .image_pipeline import empty_image_manifest, file_sha256, generate_planned_images
from .media_planning import MediaPlanner
from .pipeline import StoryPipeline
from .preview import can_render_publishable, write_funding_story_html
from .run_store import LocalRunStore
from .semantic_normalization import SemanticNormalizer


@dataclass(frozen=True, slots=True)
class StoryExecutionInput:
    generation_package: dict[str, Any]
    template_id: str | None = None
    run_id: str | None = None
    output_dir: Path | None = None


class StoryExecutor(Protocol):
    def execute(self, value: StoryExecutionInput) -> dict[str, Any]: ...


class StoryMakerExecutor:
    """Deterministic text executor, independent from transport and conversation."""

    def __init__(self, *, repository: DataRepository, pipeline: StoryPipeline) -> None:
        self.repository = repository
        self.pipeline = pipeline

    def execute(self, value: StoryExecutionInput) -> dict[str, Any]:
        self.repository.validate_approved_generation_package(value.generation_package)
        return self.pipeline.invoke(
            value.generation_package["brief"],
            template_id=value.template_id,
        )


class IntegratedStoryMakerExecutor:
    """Own one text, image, and editable-preview execution as an atomic run."""

    def __init__(
        self,
        *,
        repository: DataRepository,
        pipeline: StoryPipeline,
        semantic_normalizer: SemanticNormalizer,
        media_planner: MediaPlanner,
        image_adapter: ImageAdapter,
        image_settings: ImageSettings,
    ) -> None:
        self.repository = repository
        self.pipeline = pipeline
        self.semantic_normalizer = semantic_normalizer
        self.media_planner = media_planner
        self.image_adapter = image_adapter
        self.image_settings = image_settings

    def execute(self, value: StoryExecutionInput) -> dict[str, Any]:
        if value.run_id is None or value.output_dir is None:
            raise ValueError("Integrated execution requires run_id and output_dir")
        self.repository.validate_approved_generation_package(value.generation_package)
        if value.output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite run output: {value.output_dir}")
        package = value.generation_package
        brief = package["brief"]

        story = self.pipeline.invoke(
            brief,
            template_id=value.template_id,
        )
        template = self.repository.get_template(story["template_id"])
        media_facts = self.semantic_normalizer.normalize(package)
        media_profile = self.repository.get_media_profile(template["media_profile_ref"])
        media_plan = self.media_planner.plan(
            media_facts=media_facts,
            template=template,
            profile=media_profile,
        )
        value.output_dir.mkdir(parents=True, exist_ok=False)
        brief_path = value.output_dir / "brief.json"
        brief_path.write_text(
            json.dumps(brief, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        story_path = value.output_dir / "story.json"
        story_path.write_text(
            json.dumps(story, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        facts_path = value.output_dir / "media-facts.json"
        facts_path.write_text(
            json.dumps(media_facts, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        plan_path = value.output_dir / "media-plan.json"
        plan_path.write_text(
            json.dumps(media_plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        images_dir = value.output_dir / "images"
        if media_plan["generation_allowed"]:
            manifest = generate_planned_images(
                media_plan=media_plan,
                media_facts=media_facts,
                generation_package=package,
                output_dir=images_dir,
                repository=self.repository,
                adapter=self.image_adapter,
                settings=self.image_settings,
                run_id=value.run_id,
            )
        else:
            images_dir.mkdir(parents=True, exist_ok=False)
            manifest = empty_image_manifest(
                media_plan=media_plan,
                generation_package=package,
                settings=self.image_settings,
                run_id=value.run_id,
            )
            self.repository.validate_story_image_manifest(manifest)
            (images_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        render_manifest = deepcopy(manifest)
        for asset in render_manifest["assets"]:
            if asset["path"]:
                asset["path"] = f"images/{asset['path']}"
        draft_path = value.output_dir / "draft.html"
        write_funding_story_html(
            target=draft_path,
            story=story,
            template=template,
            media_plan=media_plan,
            manifest=render_manifest,
            mode="draft",
        )
        publishable_path: Path | None = None
        if can_render_publishable(
            media_plan=media_plan, manifest=render_manifest, story=story
        ):
            publishable_path = value.output_dir / "publishable.html"
            write_funding_story_html(
                target=publishable_path,
                story=story,
                template=template,
                media_plan=media_plan,
                manifest=render_manifest,
                mode="publishable",
            )
        manifest_path = images_dir / "manifest.json"
        result = {
            "schema_version": "integrated-story-run-v2",
            "run_id": value.run_id,
            "status": (
                "partial"
                if manifest["failed"]
                or story["warnings"]
                or not media_plan["generation_allowed"]
                else "complete"
            ),
            "template_id": story["template_id"],
            "model": story["model"],
            "input_brief": {"path": "brief.json", "sha256": file_sha256(brief_path)},
            "story": {"path": "story.json", "sha256": file_sha256(story_path)},
            "media_facts": {
                "path": "media-facts.json",
                "sha256": file_sha256(facts_path),
            },
            "media_plan": {
                "path": "media-plan.json",
                "sha256": file_sha256(plan_path),
                "decision": media_plan["decision"],
            },
            "images": {
                "manifest": {
                    "path": "images/manifest.json",
                    "sha256": file_sha256(manifest_path),
                },
                "requested": manifest["requested"],
                "succeeded": manifest["succeeded"],
                "failed": manifest["failed"],
                "qa_pending": sum(
                    asset["qa_status"] == "pending" for asset in manifest["assets"]
                ),
            },
            "draft_html": {"path": "draft.html", "sha256": file_sha256(draft_path)},
            "publishable_html": (
                {
                    "path": "publishable.html",
                    "sha256": file_sha256(publishable_path),
                }
                if publishable_path is not None
                else None
            ),
            "warning_count": len(story["warnings"]),
            "review_required": True,
        }
        self.repository.validate_integrated_story_run(result)
        return result


_REVIEW_CHECKS = (
    "scene_distinctness",
    "product_fidelity",
    "text_legibility",
    "claim_grounding",
)
_REVIEW_VALUES = {"pending", "pass", "fail", "not_applicable"}
_QA_STATUSES = {"pending", "pass", "conditional", "fail"}


def review_integrated_story_run(
    *,
    repository: DataRepository,
    store: LocalRunStore,
    run_id: str,
    reviews: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Persist human image review and render publishable HTML when all gates pass.

    ``reviews`` is keyed by media slot id. Each value may contain ``qa_status``,
    ``review_checks`` and ``qa_notes``; omitted fields retain their current values.
    The function is deliberately separate from generation so a reviewer can update a
    completed run without invoking a model again.
    """
    if not reviews:
        raise ValueError("At least one image review is required")
    record = store.get(run_id)
    if record["status"] != "completed" or not isinstance(record.get("result"), dict):
        raise ValueError("Only completed integrated runs can be reviewed")
    run_dir = store.root / run_id
    manifest_path = run_dir / "images" / "manifest.json"
    story_path = run_dir / "story.json"
    media_plan_path = run_dir / "media-plan.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    story = json.loads(story_path.read_text(encoding="utf-8"))
    media_plan = json.loads(media_plan_path.read_text(encoding="utf-8"))
    repository.validate_story_image_manifest(manifest)
    repository.validate_media_plan(media_plan)
    repository.validate_story_result(story)
    assets_by_slot = {asset["slot_id"]: asset for asset in manifest["assets"]}

    for slot_id, review in reviews.items():
        if slot_id not in assets_by_slot:
            raise ValueError(f"Unknown image slot id: {slot_id}")
        asset = assets_by_slot[slot_id]
        if asset["status"] != "success":
            raise ValueError(f"Image slot cannot be reviewed: {slot_id}")
        qa_status = str(review.get("qa_status", asset["qa_status"]))
        if qa_status not in _QA_STATUSES:
            raise ValueError(f"Invalid qa_status for {slot_id}: {qa_status}")
        checks = dict(asset["review_checks"])
        provided_checks = review.get("review_checks", {})
        if not isinstance(provided_checks, dict):
            raise ValueError(f"review_checks must be an object for {slot_id}")
        unknown_checks = set(provided_checks) - set(_REVIEW_CHECKS)
        if unknown_checks:
            raise ValueError(f"Unknown review checks for {slot_id}: {sorted(unknown_checks)}")
        for check in _REVIEW_CHECKS:
            if check in provided_checks:
                value = str(provided_checks[check])
                if value not in _REVIEW_VALUES:
                    raise ValueError(f"Invalid {check} review for {slot_id}: {value}")
                checks[check] = value
        if qa_status == "pass" and any(checks[name] != "pass" for name in _REVIEW_CHECKS):
            raise ValueError(f"A passed image requires all review checks to pass: {slot_id}")
        notes = review.get("qa_notes")
        if notes is not None:
            if not isinstance(notes, list) or not all(str(note).strip() for note in notes):
                raise ValueError(f"qa_notes must be a non-empty string list for {slot_id}")
            asset["qa_notes"] = [str(note) for note in notes]
        asset["qa_status"] = qa_status
        asset["review_checks"] = checks

    repository.validate_story_image_manifest(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    render_manifest = deepcopy(manifest)
    for asset in render_manifest["assets"]:
        if asset["path"]:
            asset["path"] = f"images/{asset['path']}"
    publishable_path = run_dir / "publishable.html"
    if can_render_publishable(
        media_plan=media_plan, manifest=render_manifest, story=story
    ):
        template = repository.get_template(story["template_id"])
        write_funding_story_html(
            target=publishable_path,
            story=story,
            template=template,
            media_plan=media_plan,
            manifest=render_manifest,
            mode="publishable",
        )
    elif publishable_path.exists():
        publishable_path.unlink()

    result = deepcopy(record["result"])
    result["images"]["qa_pending"] = sum(
        asset["qa_status"] == "pending" for asset in manifest["assets"]
    )
    result["publishable_html"] = (
        {"path": "publishable.html", "sha256": file_sha256(publishable_path)}
        if publishable_path.is_file()
        else None
    )
    repository.validate_integrated_story_run(result)
    updated = store.update_result(run_id, result)
    return {
        "run": updated,
        "publishable": publishable_path.is_file(),
        "manifest": manifest,
    }
