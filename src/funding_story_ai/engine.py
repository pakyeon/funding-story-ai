from __future__ import annotations

import json
import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .data_repository import DataRepository
from .image_generation import ImageSettings, OpenAIImageAdapter
from .image_pipeline import file_sha256, generate_section_images, planned_image_sections
from .pipeline import StoryPipeline
from .preview import write_story_preview


@dataclass(frozen=True, slots=True)
class StoryExecutionInput:
    brief: dict[str, Any]
    template_id: str | None = None
    category_profile_id: str | None = None
    run_id: str | None = None
    output_dir: Path | None = None
    reference_image_path: Path | None = None
    generate_images: bool = True


class StoryExecutor(Protocol):
    def execute(self, value: StoryExecutionInput) -> dict[str, Any]: ...


class StoryMakerExecutor:
    """Deterministic text executor, independent from transport and conversation."""

    def __init__(self, *, repository: DataRepository, pipeline: StoryPipeline) -> None:
        self.repository = repository
        self.pipeline = pipeline

    def execute(self, value: StoryExecutionInput) -> dict[str, Any]:
        self.repository.validate_story_brief(value.brief)
        return self.pipeline.invoke(
            value.brief,
            template_id=value.template_id,
            category_profile_id=value.category_profile_id,
        )


class IntegratedStoryMakerExecutor:
    """Own one text, image, and editable-preview execution as an atomic run."""

    def __init__(
        self,
        *,
        repository: DataRepository,
        pipeline: StoryPipeline,
        image_adapter: OpenAIImageAdapter,
        image_settings: ImageSettings,
    ) -> None:
        self.repository = repository
        self.pipeline = pipeline
        self.image_adapter = image_adapter
        self.image_settings = image_settings

    def execute(self, value: StoryExecutionInput) -> dict[str, Any]:
        if value.run_id is None or value.output_dir is None:
            raise ValueError("Integrated execution requires run_id and output_dir")
        self.repository.validate_story_brief(value.brief)
        if value.output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite run output: {value.output_dir}")
        if value.reference_image_path is not None and not value.reference_image_path.is_file():
            raise FileNotFoundError(value.reference_image_path)

        story = self.pipeline.invoke(
            value.brief,
            template_id=value.template_id,
            category_profile_id=value.category_profile_id,
        )
        template = self.repository.get_template(story["template_id"])
        value.output_dir.mkdir(parents=True, exist_ok=False)
        brief_path = value.output_dir / "brief.json"
        brief_path.write_text(
            json.dumps(value.brief, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        story_path = value.output_dir / "story.json"
        story_path.write_text(
            json.dumps(story, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if not value.generate_images:
            raise ValueError("Integrated execution requires section image generation")
        visual_identity = " / ".join(
            [
                value.brief["product"]["name"],
                value.brief["product"]["summary"],
                *[
                    asset["description"]
                    for asset in value.brief["assets"]
                    if asset["asset_type"] in {"product", "brand"}
                ],
            ]
        )
        plans = planned_image_sections(
            story,
            template,
            reference_available=value.reference_image_path is not None,
            visual_identity=visual_identity,
        )
        self.image_adapter.ledger.assert_can_call(
            self.image_settings.reserve_usd_per_call * len(plans)
        )
        images_dir = value.output_dir / "images"
        manifest = generate_section_images(
            story_path=story_path,
            reference_path=value.reference_image_path,
            output_dir=images_dir,
            repository=self.repository,
            adapter=self.image_adapter,
            settings=self.image_settings,
            run_id=value.run_id,
            visual_identity=visual_identity,
        )

        fallback_image: str | None = None
        if value.reference_image_path is not None:
            suffix = value.reference_image_path.suffix.lower() or ".bin"
            target = images_dir / f"reference{suffix}"
            shutil.copy2(value.reference_image_path, target)
            fallback_image = f"images/{target.name}"

        render_manifest = deepcopy(manifest)
        for asset in render_manifest["assets"]:
            if asset["path"]:
                asset["path"] = f"images/{asset['path']}"
        preview_path = value.output_dir / "preview.html"
        write_story_preview(
            target=preview_path,
            story=story,
            template=template,
            manifest=render_manifest,
            fallback_image=fallback_image,
        )
        manifest_path = images_dir / "manifest.json"
        result = {
            "schema_version": "integrated-story-run-v1",
            "run_id": value.run_id,
            "status": "partial" if manifest["failed"] or story["warnings"] else "complete",
            "template_id": story["template_id"],
            "model": story["model"],
            "input_brief": {"path": "brief.json", "sha256": file_sha256(brief_path)},
            "story": {"path": "story.json", "sha256": file_sha256(story_path)},
            "images": {
                "manifest": {
                    "path": "images/manifest.json",
                    "sha256": file_sha256(manifest_path),
                },
                "requested": manifest["requested"],
                "succeeded": manifest["succeeded"],
                "failed": manifest["failed"],
                "estimated_cost_usd": manifest["estimated_cost_usd"],
                "qa_pending": sum(
                    asset["qa_status"] == "pending" for asset in manifest["assets"]
                ),
            },
            "preview": {"path": "preview.html", "sha256": file_sha256(preview_path)},
            "warning_count": len(story["warnings"]),
            "review_required": True,
        }
        self.repository.validate_integrated_story_run(result)
        return result
