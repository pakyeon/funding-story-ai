from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource


class DataValidationError(ValueError):
    """Raised when a repository artifact violates its public contract."""


class DataRepository:
    """Load and validate templates, examples, and generated artifacts."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path(__file__).resolve().parents[2]
        self.schemas_dir = self.root / "schemas"
        self.templates_dir = self.root / "templates"
        self.examples_dir = self.root / "examples"
        self._schemas = {
            "section": self._read_json(
                self.schemas_dir / "story-template-section.schema.json"
            ),
            "template": self._read_json(
                self.schemas_dir / "story-template.schema.json"
            ),
            "brief": self._read_json(self.schemas_dir / "story-brief.schema.json"),
            "story_generation_content": self._read_json(
                self.schemas_dir / "story-generation-content.schema.json"
            ),
            "story_result": self._read_json(
                self.schemas_dir / "story-result.schema.json"
            ),
            "story_image_manifest": self._read_json(
                self.schemas_dir / "story-image-manifest.schema.json"
            ),
            "integrated_story_run": self._read_json(
                self.schemas_dir / "integrated-story-run.schema.json"
            ),
            "template_retrieval_index": self._read_json(
                self.schemas_dir / "template-retrieval-index.schema.json"
            ),
            "catalog": self._read_json(
                self.schemas_dir / "template-catalog.schema.json"
            ),
            "intake_semantic_state": self._read_json(
                self.schemas_dir / "story-intake-semantic-state.schema.json"
            ),
        }
        self._registry = Registry().with_resource(
            self._schemas["section"]["$id"],
            Resource.from_contents(self._schemas["section"]),
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open(encoding="utf-8") as source_file:
            value = json.load(source_file)
        if not isinstance(value, dict):
            raise DataValidationError(f"JSON object required: {path}")
        return value

    def _validate(self, value: dict[str, Any], schema_name: str, label: str) -> None:
        validator = Draft202012Validator(
            self._schemas[schema_name],
            registry=self._registry,
            format_checker=FormatChecker(),
        )
        errors = sorted(validator.iter_errors(value), key=lambda error: list(error.path))
        if errors:
            details = "; ".join(
                f"{'.'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
                for error in errors
            )
            raise DataValidationError(f"{label} failed schema validation: {details}")

    def check_schemas(self) -> None:
        for name, schema in self._schemas.items():
            try:
                Draft202012Validator.check_schema(schema)
            except Exception as exc:
                raise DataValidationError(f"Invalid {name} schema") from exc

    def load_templates(self) -> list[dict[str, Any]]:
        templates = [
            self._read_json(path)
            for path in sorted(self.templates_dir.glob("t[0-9][0-9]_*.json"))
        ]
        if not templates:
            raise DataValidationError("No templates found")
        for template in templates:
            self._validate(template, "template", template.get("id", "template"))
            self._require_unique(
                (section["id"] for section in template["layout"]),
                f"{template['id']} layout section id",
            )
        self._require_unique((template["id"] for template in templates), "template id")
        return templates

    def get_template(self, template_id: str) -> dict[str, Any]:
        for template in self.load_templates():
            if template["id"] == template_id:
                return template
        raise DataValidationError(f"Unknown template id: {template_id}")

    def load_catalog(self) -> dict[str, Any]:
        catalog = self._read_json(self.templates_dir / "catalog.json")
        self._validate(catalog, "catalog", "template catalog")
        self._require_unique(
            (entry["template_id"] for entry in catalog["templates"]),
            "catalog template id",
        )
        return catalog

    def get_template_version(self, template_id: str) -> str:
        for entry in self.load_catalog()["templates"]:
            if entry["template_id"] == template_id:
                return str(entry["template_version"])
        raise DataValidationError(f"No catalog entry for template id: {template_id}")

    def load_brief(self, name: str = "robot-vacuum/brief.json") -> dict[str, Any]:
        return self.load_brief_path(self.examples_dir / name)

    def load_brief_path(self, path: Path) -> dict[str, Any]:
        brief = self._read_json(path)
        self.validate_story_brief(brief)
        return brief

    def load_story_result(self, path: Path) -> dict[str, Any]:
        story = self._read_json(path)
        self.validate_story_result(story)
        return story

    def load_story_image_manifest(self, path: Path) -> dict[str, Any]:
        manifest = self._read_json(path)
        self.validate_story_image_manifest(manifest)
        return manifest

    def validate_story_brief(self, brief: dict[str, Any]) -> None:
        self._validate(brief, "brief", brief.get("brief_id", "story brief"))
        self._validate_brief_links(brief)

    def validate_intake_semantic_state(self, value: dict[str, Any]) -> None:
        self._validate(
            value,
            "intake_semantic_state",
            value.get("input_id", "story intake semantic state"),
        )

    def story_brief_schema(self) -> dict[str, Any]:
        return deepcopy(self._schemas["brief"])

    def intake_semantic_state_schema(self) -> dict[str, Any]:
        return deepcopy(self._schemas["intake_semantic_state"])

    def story_generation_content_schema(self) -> dict[str, Any]:
        return deepcopy(self._schemas["story_generation_content"])

    def validate_story_generation_content(self, value: dict[str, Any]) -> None:
        self._validate(value, "story_generation_content", "story generation content")

    def validate_story_result(self, value: dict[str, Any]) -> None:
        self._validate(value, "story_result", "story result")

    def validate_story_image_manifest(self, value: dict[str, Any]) -> None:
        self._validate(value, "story_image_manifest", "story image manifest")

    def validate_integrated_story_run(self, value: dict[str, Any]) -> None:
        self._validate(value, "integrated_story_run", "integrated story run")

    def load_template_retrieval_index(self) -> dict[str, Any]:
        value = self._read_json(self.templates_dir / "retrieval-index.json")
        self._validate(value, "template_retrieval_index", "template retrieval index")
        self._require_unique(
            (candidate["candidate_id"] for candidate in value["candidates"]),
            "retrieval candidate id",
        )
        executable = {
            candidate["executable_template_id"]
            for candidate in value["candidates"]
            if candidate["executable_template_id"] is not None
        }
        available = {template["id"] for template in self.load_templates()}
        missing = executable - available
        if missing:
            raise DataValidationError(
                f"Retrieval index references unavailable templates: {sorted(missing)}"
            )
        return value

    def validate_catalog_links(self) -> None:
        template_ids = {template["id"] for template in self.load_templates()}
        catalog_ids = {
            entry["template_id"] for entry in self.load_catalog()["templates"]
        }
        if template_ids != catalog_ids:
            raise DataValidationError(
                f"Template/catalog id mismatch: templates={template_ids}, "
                f"catalog={catalog_ids}"
            )
        self.load_template_retrieval_index()

    @staticmethod
    def _require_unique(values: Iterable[str], label: str) -> None:
        values = list(values)
        if len(values) != len(set(values)):
            raise DataValidationError(f"Duplicate {label}")

    def _validate_brief_links(self, brief: dict[str, Any]) -> None:
        source_ids = {source["source_id"] for source in brief["source"]["refs"]}
        fact_ids = {fact["id"] for fact in brief["product"]["facts"]}
        claim_ids = {claim["id"] for claim in brief["claims"]}
        evidence_ids = {evidence["id"] for evidence in brief["evidence"]}
        entity_groups = [
            brief["product"]["facts"],
            brief["audiences"],
            brief["problems"],
            brief["features"],
            brief["claims"],
            brief["evidence"],
            brief["assets"],
            brief["rewards"],
        ]
        all_ids = [entity["id"] for group in entity_groups for entity in group]
        self._require_unique(all_ids, "brief entity id")

        for entity in [entity for group in entity_groups for entity in group]:
            self._require_subset(entity.get("source_refs", []), source_ids, entity["id"])
        self._require_subset(
            brief["schedule_policy"]["source_refs"], source_ids, "schedule_policy"
        )
        for feature in brief["features"]:
            self._require_subset(
                feature["fact_ids"], fact_ids, f"{feature['id']}.fact_ids"
            )
            self._require_subset(
                feature["evidence_ids"],
                evidence_ids,
                f"{feature['id']}.evidence_ids",
            )
        for claim in brief["claims"]:
            self._require_subset(
                claim["evidence_ids"], evidence_ids, f"{claim['id']}.evidence_ids"
            )
        for evidence in brief["evidence"]:
            self._require_subset(
                evidence["supports_claim_ids"],
                claim_ids,
                f"{evidence['id']}.supports_claim_ids",
            )

    @staticmethod
    def _require_subset(values: Iterable[str], allowed: set[str], label: str) -> None:
        missing = set(values) - allowed
        if missing:
            raise DataValidationError(f"Unknown references in {label}: {sorted(missing)}")
