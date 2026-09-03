from __future__ import annotations

import json
import re
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
        self.category_modules_dir = self.root / "category_modules"
        self.media_profiles_dir = self.root / "media_profiles"
        self.examples_dir = self.root / "examples"
        self._schemas = {
            "section": self._read_json(self.schemas_dir / "story-template-section.schema.json"),
            "template": self._read_json(self.schemas_dir / "story-template.schema.json"),
            "category_module": self._read_json(
                self.schemas_dir / "story-category-module.schema.json"
            ),
            "brief": self._read_json(self.schemas_dir / "story-brief.schema.json"),
            "story_generation_content": self._read_json(
                self.schemas_dir / "story-generation-content.schema.json"
            ),
            "story_result": self._read_json(self.schemas_dir / "story-result.schema.json"),
            "story_image_manifest": self._read_json(
                self.schemas_dir / "story-image-manifest.schema.json"
            ),
            "integrated_story_run": self._read_json(
                self.schemas_dir / "integrated-story-run.schema.json"
            ),
            "template_retrieval_index": self._read_json(
                self.schemas_dir / "template-retrieval-index.schema.json"
            ),
            "catalog": self._read_json(self.schemas_dir / "template-catalog.schema.json"),
            "intake_semantic_state": self._read_json(
                self.schemas_dir / "story-intake-semantic-state.schema.json"
            ),
            "approved_generation_package": self._read_json(
                self.schemas_dir / "approved-generation-package.schema.json"
            ),
            "media_facts": self._read_json(self.schemas_dir / "media-facts.schema.json"),
            "media_profile": self._read_json(self.schemas_dir / "media-profile.schema.json"),
            "media_plan": self._read_json(self.schemas_dir / "media-plan.schema.json"),
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
            self._read_json(path) for path in sorted(self.templates_dir.glob("t[0-9][0-9]_*.json"))
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

    def load_category_modules(self) -> list[dict[str, Any]]:
        modules = [
            self._read_json(path) for path in sorted(self.category_modules_dir.glob("*.json"))
        ]
        if not modules:
            raise DataValidationError("No story category modules found")
        for module in modules:
            self._validate(
                module,
                "category_module",
                module.get("id", "story category module"),
            )
            self._require_unique(
                (entry["section"]["id"] for entry in module["section_modules"]),
                f"{module['id']} section module id",
            )
        self._require_unique((module["id"] for module in modules), "category module id")
        return modules

    def get_category_module(self, module_id: str) -> dict[str, Any]:
        for module in self.load_category_modules():
            if module["id"] == module_id:
                return module
        raise DataValidationError(f"Unknown story category module id: {module_id}")

    def resolve_category_module(self, brief: dict[str, Any]) -> dict[str, Any]:
        category = brief["product"]["category"]
        product_type = re.sub(r"\s+", "", brief["product"]["product_type"]).casefold()
        candidates = []
        for module in self.load_category_modules():
            if module["category"] != category:
                continue
            normalized_types = [
                re.sub(r"\s+", "", value).casefold() for value in module["product_types"]
            ]
            if any(value in product_type or product_type in value for value in normalized_types):
                candidates.append(module)
        if len(candidates) != 1:
            raise DataValidationError(
                "Expected exactly one story category module for "
                f"category={category!r}, product_type={brief['product']['product_type']!r}; "
                f"found={len(candidates)}"
            )
        return candidates[0]

    @staticmethod
    def _active_capability_groups(media_facts: dict[str, Any] | None) -> set[str] | None:
        if media_facts is None:
            return None
        fact_by_id = {fact["fact_id"]: fact for fact in media_facts["facts"]}
        active: set[str] = set()
        for proposition in media_facts["propositions"]:
            fact = fact_by_id[proposition["fact_id"]]
            if fact["availability"] == "explicitly_absent":
                continue
            if fact["support_level"] == "rejected":
                continue
            if (
                proposition["capability_group"] == "evidence_performance"
                and fact["availability"] == "provided"
                and fact["support_level"] != "supported"
            ):
                continue
            active.add(proposition["capability_group"])
        return active

    def compose_template(
        self,
        *,
        template_id: str,
        brief: dict[str, Any],
        media_facts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Combine one persuasion template with one product-family module.

        Purpose templates always contribute the four public story stages. Product-family
        sections are inserted only when normalized facts activate them. Without normalized
        facts, all module sections are included so the standalone text executor remains useful.
        """

        template = deepcopy(self.get_template(template_id))
        module = self.resolve_category_module(brief)
        active_groups = self._active_capability_groups(media_facts)
        unknown_fields = {item["field"] for item in brief.get("unknowns", [])}
        module_entries_by_stage: dict[str, list[dict[str, Any]]] = {}
        for entry in module["section_modules"]:
            placeholder = entry["content_placeholder"]
            placeholder_fields = set(placeholder["fields"]) if placeholder else set()
            include_for_placeholder = bool(placeholder_fields.intersection(unknown_fields))
            include_for_fact = active_groups is None or bool(
                set(entry["activation_capability_groups"]).intersection(active_groups)
            )
            if include_for_fact or include_for_placeholder:
                module_entries_by_stage.setdefault(entry["after_stage"], []).append(entry)

        layout: list[dict[str, Any]] = []
        placeholders = deepcopy(template["content_placeholders"])
        for stage in template["layout"]:
            layout.append(deepcopy(stage))
            for entry in module_entries_by_stage.get(stage["id"], []):
                section = deepcopy(entry["section"])
                layout.append(section)
                if entry["content_placeholder"] is not None:
                    placeholders[section["id"]] = deepcopy(entry["content_placeholder"])

        template["layout"] = layout
        template["content_placeholders"] = placeholders
        template["category"] = module["category"]
        template["category_module_id"] = module["id"]
        template["category_module_version"] = module["version"]
        template["media_profile_ref"] = module["media_profile_ref"]
        template["media_section_bindings"] = deepcopy(module["media_section_bindings"])
        return template

    def load_catalog(self) -> dict[str, Any]:
        catalog = self._read_json(self.templates_dir / "catalog.json")
        self._validate(catalog, "catalog", "template catalog")
        self._require_unique(
            (entry["template_id"] for entry in catalog["templates"]),
            "catalog template id",
        )
        return catalog

    def load_media_profiles(self) -> list[dict[str, Any]]:
        profiles = [
            self._read_json(path) for path in sorted(self.media_profiles_dir.glob("*.json"))
        ]
        if not profiles:
            raise DataValidationError("No media profiles found")
        for profile in profiles:
            self.validate_media_profile(profile)
            self._require_unique(
                (group["id"] for group in profile["capability_groups"]),
                f"{profile['id']} capability group id",
            )
        self._require_unique((profile["id"] for profile in profiles), "media profile id")
        return profiles

    def get_media_profile(self, profile_id: str) -> dict[str, Any]:
        for profile in self.load_media_profiles():
            if profile["id"] == profile_id:
                return profile
        raise DataValidationError(f"Unknown media profile id: {profile_id}")

    def validate_template_media_profile_links(self) -> None:
        profiles = {profile["id"]: profile for profile in self.load_media_profiles()}
        templates = self.load_templates()
        for module in self.load_category_modules():
            profile_id = module["media_profile_ref"]
            if profile_id not in profiles:
                raise DataValidationError(
                    f"{module['id']} references unknown media profile: {profile_id}"
                )
            for template in templates:
                section_by_id = {
                    section["id"]: section
                    for section in [
                        *template["layout"],
                        *(entry["section"] for entry in module["section_modules"]),
                    ]
                }
                for group in profiles[profile_id]["capability_groups"]:
                    section_id = module["media_section_bindings"][group["id"]]
                    section = section_by_id.get(section_id)
                    if section is None:
                        raise DataValidationError(
                            f"{module['id']} binds {group['id']} to unknown section {section_id}"
                        )
                    if section["type"] not in group["allowed_section_types"]:
                        raise DataValidationError(
                            f"{module['id']} binding for {group['id']} uses incompatible "
                            f"section type {section['type']}"
                        )

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
        if value["requested"] != len(value["assets"]):
            raise DataValidationError("image manifest requested count does not match assets")
        if value["succeeded"] + value["failed"] != value["requested"]:
            raise DataValidationError("image manifest success and failure counts do not add up")
        self._require_unique(
            (asset["slot_id"] for asset in value["assets"]),
            "image manifest slot id",
        )

    def validate_integrated_story_run(self, value: dict[str, Any]) -> None:
        self._validate(value, "integrated_story_run", "integrated story run")

    def validate_approved_generation_package(self, value: dict[str, Any]) -> None:
        self._validate(
            value,
            "approved_generation_package",
            value.get("input_id", "approved generation package"),
        )
        self.validate_story_brief(value["brief"])

    def validate_media_facts(self, value: dict[str, Any]) -> None:
        self._validate(value, "media_facts", value.get("brief_id", "media facts"))
        proposition_ids = {item["proposition_id"] for item in value["propositions"]}
        fact_ids = {item["fact_id"] for item in value["facts"]}
        if len(proposition_ids) != len(value["propositions"]):
            raise DataValidationError("media facts contain duplicate proposition ids")
        if len(fact_ids) != len(value["facts"]):
            raise DataValidationError("media facts contain duplicate fact ids")
        for proposition in value["propositions"]:
            if proposition["fact_id"] not in fact_ids:
                raise DataValidationError("media proposition references an unknown fact")
        linked = {
            proposition_id for fact in value["facts"] for proposition_id in fact["proposition_ids"]
        }
        if linked != proposition_ids:
            raise DataValidationError("media fact proposition links are incomplete")

    def validate_media_profile(self, value: dict[str, Any]) -> None:
        self._validate(value, "media_profile", value.get("id", "media profile"))
        self._require_unique(
            (group["id"] for group in value["capability_groups"]),
            f"{value['id']} capability group id",
        )
        for group in value["capability_groups"]:
            bounds = group["cardinality"]
            if bounds["min"] > bounds["max"]:
                raise DataValidationError(
                    f"{value['id']} {group['id']} cardinality min exceeds max"
                )

    def validate_media_plan(self, value: dict[str, Any]) -> None:
        self._validate(value, "media_plan", value.get("brief_id", "media plan"))
        self._require_unique((slot["slot_id"] for slot in value["slots"]), "media plan slot id")

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
        catalog_ids = {entry["template_id"] for entry in self.load_catalog()["templates"]}
        if template_ids != catalog_ids:
            raise DataValidationError(
                f"Template/catalog id mismatch: templates={template_ids}, catalog={catalog_ids}"
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
        self._require_subset(brief["schedule_policy"]["source_refs"], source_ids, "schedule_policy")
        for feature in brief["features"]:
            self._require_subset(feature["fact_ids"], fact_ids, f"{feature['id']}.fact_ids")
            self._require_subset(
                feature["evidence_ids"],
                evidence_ids,
                f"{feature['id']}.evidence_ids",
            )
        for claim in brief["claims"]:
            self._require_subset(claim["evidence_ids"], evidence_ids, f"{claim['id']}.evidence_ids")
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
