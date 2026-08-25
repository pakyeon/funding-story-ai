from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from .adapter import GeminiAdapter
from .data_repository import DataRepository
from .image_generation import (
    GeminiImageAdapter,
    ImageSettings,
    OpenAIImageAdapter,
    RetryingFallbackImageAdapter,
)
from .image_pipeline import generate_section_images, planned_image_sections
from .preview import write_story_preview
from .smoke import build_runtime


def _default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts/previews") / f"preview-{timestamp}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate section images and HTML preview")
    parser.add_argument("--story", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=Path,
        help="Optional product reference image; omit to create a seed image",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--section",
        action="append",
        dest="sections",
        help="Generate only the named image-required section; may be repeated",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    args = parser.parse_args()

    load_dotenv(dotenv_path=Path(".env"), override=False)
    repository = DataRepository()
    story = repository.load_story_result(args.story)
    template = repository.get_template(story["template_id"])
    section_ids = set(args.sections) if args.sections else None
    plans = planned_image_sections(
        story,
        template,
        section_ids,
        reference_available=args.reference is not None,
    )
    settings = ImageSettings(
        model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2"),
        size=os.getenv("OPENAI_IMAGE_SIZE", "1536x1024"),
        quality=os.getenv("OPENAI_IMAGE_QUALITY", "low"),
        output_format=os.getenv("OPENAI_IMAGE_OUTPUT_FORMAT", "jpeg"),
        output_compression=int(os.getenv("OPENAI_IMAGE_OUTPUT_COMPRESSION", "85")),
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry-run",
                    "model": settings.model,
                    "size": settings.size,
                    "quality": settings.quality,
                    "section_ids": [plan["section_id"] for plan in plans],
                    "image_count": len(plans),
                    "api_key_present": bool(os.getenv("OPENAI_API_KEY", "").strip()),
                },
                ensure_ascii=False,
            )
        )
        return

    live_settings = ImageSettings.from_env()
    output_dir = args.output_dir or _default_output_dir()
    gemini = GeminiAdapter(build_runtime())
    adapters = []
    if os.getenv("OPENAI_API_KEY", "").strip():
        adapters.append(OpenAIImageAdapter(live_settings))
    adapters.append(GeminiImageAdapter(live_settings, client=gemini.client))
    adapter = RetryingFallbackImageAdapter(
        adapters,
        attempts_per_provider=live_settings.attempts_per_provider,
    )
    manifest = generate_section_images(
        story_path=args.story,
        reference_path=args.reference,
        output_dir=output_dir,
        repository=repository,
        adapter=adapter,
        settings=live_settings,
        section_ids=section_ids,
    )
    reference_target = None
    if args.reference is not None:
        reference_target = output_dir / f"reference{args.reference.suffix.lower()}"
        shutil.copy2(args.reference, reference_target)
    fallback_image = (
        reference_target.name
        if reference_target is not None
        else next(
            (
                asset["path"]
                for asset in manifest["assets"]
                if asset["status"] == "success" and asset["path"]
            ),
            "",
        )
    )
    preview_path = output_dir / "preview.html"
    write_story_preview(
        target=preview_path,
        story=story,
        template=template,
        manifest=manifest,
        fallback_image=fallback_image,
    )
    print(
        json.dumps(
            {
                "status": "ok" if manifest["failed"] == 0 else "partial",
                "output_dir": str(output_dir),
                "preview": str(preview_path),
                "manifest": str(output_dir / "manifest.json"),
                "requested": manifest["requested"],
                "succeeded": manifest["succeeded"],
                "failed": manifest["failed"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
