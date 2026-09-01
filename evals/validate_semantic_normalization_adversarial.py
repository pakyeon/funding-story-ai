from __future__ import annotations

import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jsonschema

EVAL_ROOT = Path(__file__).parent
BASE_VALIDATOR_PATH = EVAL_ROOT / "validate_semantic_normalization_dataset.py"
BASE_SPEC = importlib.util.spec_from_file_location(
    "semantic_dataset_validator", BASE_VALIDATOR_PATH
)
assert BASE_SPEC is not None and BASE_SPEC.loader is not None
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)

ROOT = EVAL_ROOT / "datasets" / "robot-vacuum-semantic-normalization-adversarial-v1"
CASES_PATH = ROOT / "adversarial-cases.json"
MANIFEST_PATH = ROOT / "dataset-manifest.json"
DEV_ROOT = EVAL_ROOT / "datasets" / "robot-vacuum-semantic-normalization-v1"
HOLDOUT_ROOT = EVAL_ROOT / "datasets" / "robot-vacuum-semantic-normalization-holdout-v1"

LANGUAGE_TAGS = {"language_ko": 11, "language_en": 11, "language_mixed": 10}
POSITION_TAGS = {"position_front": 11, "position_middle": 11, "position_end": 10}
FORMAT_TAGS = {
    "format_role_alias": 7,
    "format_json_record": 7,
    "format_markdown_quote": 6,
    "format_html_fragment": 6,
    "format_tool_quote": 6,
}
FORGED_TARGET_TAGS = {
    "ref_target_entity_source_refs": 4,
    "ref_target_fact_source_refs": 4,
    "ref_target_entity_evidence_refs": 4,
    "ref_target_fact_evidence_refs": 4,
    "ref_target_entity_asset_refs": 4,
    "ref_target_fact_asset_refs": 4,
    "ref_target_evidence_source_refs": 4,
    "ref_target_asset_source_refs": 4,
}


def _normalize(value: str) -> str:
    return " ".join(value.strip().rstrip(".!?。 ").split())


def _texts(cases: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    return (
        {
            _normalize(fact["statement"])
            for case in cases
            for fact in case["model_view"]["facts"]
        },
        {
            _normalize(proposition["text"])
            for case in cases
            for proposition in case["expected"]["atomic_propositions"]
        },
    )


def _all_ids(cases: list[dict[str, Any]]) -> dict[str, set[str]]:
    ids: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        projection = case["input"]["approved_entity_projection"]
        for key, collection, field in (
            ("entity", projection["entities"], "entity_id"),
            ("fact", projection["facts"], "fact_id"),
            ("source", projection["sources"], "source_id"),
            ("evidence", projection["evidence"], "evidence_id"),
            ("asset", projection["assets"], "asset_id"),
            ("proposition", case["expected"]["atomic_propositions"], "proposition_id"),
        ):
            ids[key].update(item[field] for item in collection)
    return ids


def validate_adversarial(cases: list[dict[str, Any]]) -> dict[str, Any]:
    schema = base._load(base.SCHEMA_PATH)
    schema_validator = jsonschema.Draft202012Validator(schema)
    base._require(len(cases) == 64, "adversarial: expected 64 cases")
    base._require(
        {case["case_id"] for case in cases}
        == {f"rv_semantic_adversarial_{index:03d}" for index in range(1, 65)},
        "adversarial: case ID sequence",
    )
    base._require(
        all(case["case_kind"] == "defensive" for case in cases),
        "adversarial: case kind",
    )
    variants = Counter(case["variant"] for case in cases)
    base._require(set(variants.values()) == {16}, "adversarial: unbalanced variants")

    summaries = [base._validate_case(case, schema_validator) for case in cases]
    decisions = Counter(summary["decision"] for summary in summaries)
    base._require(
        decisions == {"accepted": 32, "reject_forged_reference": 32},
        "adversarial: decision distribution",
    )
    prompt_cases = [case for case in cases if "prompt_injection" in case["tags"]]
    forged_cases = [case for case in cases if "forged_link" in case["tags"]]
    base._require(len(prompt_cases) == 32, "adversarial: prompt injection count")
    base._require(len(forged_cases) == 32, "adversarial: forged link count")
    base._require(
        all(case["expected"]["adapter"]["decision"] == "accepted" for case in prompt_cases),
        "adversarial: injection data must cross the valid boundary",
    )
    base._require(
        all(
            not case["expected"]["atomic_propositions"]
            and not case["expected"]["ignored_fact_ids"]
            and not case["expected"]["media_facts"]
            for case in forged_cases
        ),
        "adversarial: rejected references invoked the semantic model",
    )

    def _tag_counts(selected: list[dict[str, Any]], keys: set[str]) -> dict[str, int]:
        counts = Counter(tag for case in selected for tag in case["tags"] if tag in keys)
        return dict(sorted(counts.items()))

    language_counts = _tag_counts(prompt_cases, set(LANGUAGE_TAGS))
    position_counts = _tag_counts(prompt_cases, set(POSITION_TAGS))
    format_counts = _tag_counts(prompt_cases, set(FORMAT_TAGS))
    forged_target_counts = _tag_counts(forged_cases, set(FORGED_TARGET_TAGS))
    base._require(language_counts == LANGUAGE_TAGS, "adversarial: language distribution")
    base._require(position_counts == POSITION_TAGS, "adversarial: position distribution")
    base._require(format_counts == FORMAT_TAGS, "adversarial: format distribution")
    base._require(
        forged_target_counts == FORGED_TARGET_TAGS,
        "adversarial: forged target distribution",
    )

    comparison = [
        *base._load(DEV_ROOT / "normal-cases.json"),
        *base._load(DEV_ROOT / "defensive-cases.json"),
        *base._load(HOLDOUT_ROOT / "holdout-cases.json"),
    ]
    adversarial_ids = _all_ids(cases)
    comparison_ids = _all_ids(comparison)
    id_overlaps = {
        key: len(adversarial_ids[key] & comparison_ids[key]) for key in adversarial_ids
    }
    base._require(not any(id_overlaps.values()), "adversarial IDs overlap another split")
    fact_texts, proposition_texts = _texts(cases)
    comparison_facts, comparison_propositions = _texts(comparison)
    fact_overlap = fact_texts & comparison_facts
    proposition_overlap = proposition_texts & comparison_propositions
    base._require(not fact_overlap, "adversarial fact text overlaps another split")
    base._require(not proposition_overlap, "adversarial proposition overlaps another split")

    return {
        "case_count": len(cases),
        "variant_counts": dict(sorted(variants.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "prompt_injection_count": len(prompt_cases),
        "forged_reference_count": len(forged_cases),
        "language_counts": language_counts,
        "position_counts": position_counts,
        "format_counts": format_counts,
        "forged_target_counts": forged_target_counts,
        "rejected_semantic_invocation_count": 0,
        "unresolved_case_count": sum(bool(summary["unresolved"]) for summary in summaries),
        "unresolved_reference_count": sum(len(summary["unresolved"]) for summary in summaries),
        "cross_split_id_overlap_count": dict(sorted(id_overlaps.items())),
        "cross_split_fact_overlap_count": len(fact_overlap),
        "cross_split_proposition_overlap_count": len(proposition_overlap),
    }


def main() -> None:
    cases = base._load(CASES_PATH)
    summary = validate_adversarial(cases)
    if MANIFEST_PATH.exists():
        manifest = base._load(MANIFEST_PATH)
        base._require(manifest["summary"] == summary, "stale adversarial manifest")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
