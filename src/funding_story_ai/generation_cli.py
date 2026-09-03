from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from .adapter import GeminiAdapter
from .data_repository import DataRepository
from .pipeline import StoryPipeline
from .selector import TemplateSelector
from .smoke import build_runtime


def _default_output_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("artifacts/generations") / f"story-{timestamp}.json"


def build_dry_run_summary(
    repository: DataRepository,
    brief_name: str | None,
    template_id: str | None,
    brief_path: Path | None = None,
) -> dict:
    load_dotenv(dotenv_path=Path(".env"), override=False)
    settings = build_runtime(require_project=False)
    brief = _load_brief(repository, brief_name=brief_name, brief_path=brief_path)
    if template_id:
        selected_template_id = repository.get_template(template_id)["id"]
        selection_scores = {template_id: 0}
        selection_reasons = ["explicit template request"]
    else:
        selection = TemplateSelector().select(brief, repository.load_templates())
        selected_template_id = selection.template_id
        selection_scores = selection.scores
        selection_reasons = list(selection.reasons)
    template = repository.compose_template(
        template_id=selected_template_id,
        brief=brief,
    )
    return {
        "mode": "dry-run",
        "brief_id": brief["brief_id"],
        "template_id": template["id"],
        "template_version": repository.get_template_version(template["id"]),
        "category_module_id": template["category_module_id"],
        "media_profile_id": template["media_profile_ref"],
        "selection_scores": selection_scores,
        "selection_reasons": selection_reasons,
        "model": settings.primary_model,
    }


def _load_brief(
    repository: DataRepository,
    *,
    brief_name: str | None,
    brief_path: Path | None,
) -> dict:
    if brief_path is not None:
        return repository.load_brief_path(brief_path)
    return repository.load_brief(brief_name or "robot-vacuum/brief.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a structured crowdfunding story")
    brief_source = parser.add_mutually_exclusive_group()
    brief_source.add_argument(
        "--brief",
        help="File name under examples (default: robot-vacuum/brief.json)",
    )
    brief_source.add_argument(
        "--brief-path",
        type=Path,
        help="Path to a schema-validated story brief",
    )
    parser.add_argument("--template", help="Explicit template id; omit for selection")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate and select only")
    mode.add_argument("--live", action="store_true", help="Call Vertex AI and save JSON")
    parser.add_argument("--output", type=Path, help="New output path for --live")
    args = parser.parse_args()

    repository = DataRepository()
    if args.dry_run:
        print(
            json.dumps(
                build_dry_run_summary(
                    repository,
                    args.brief,
                    args.template,
                    brief_path=args.brief_path,
                ),
                ensure_ascii=False,
            )
        )
        return

    load_dotenv(dotenv_path=Path(".env"), override=False)
    settings = build_runtime()
    adapter = GeminiAdapter(settings)
    pipeline = StoryPipeline(repository=repository, adapter=adapter)
    result = pipeline.invoke(
        _load_brief(
            repository,
            brief_name=args.brief,
            brief_path=args.brief_path,
        ),
        template_id=args.template,
    )
    output_path = args.output or _default_output_path()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output_path),
                "model": result["model"],
                "template_id": result["template_id"],
                "review_required": result["review_required"],
                "warning_count": len(result["warnings"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
