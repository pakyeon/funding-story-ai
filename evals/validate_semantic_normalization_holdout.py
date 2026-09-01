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

HOLDOUT_ROOT = EVAL_ROOT / "datasets" / "robot-vacuum-semantic-normalization-holdout-v1"
HOLDOUT_PATH = HOLDOUT_ROOT / "holdout-cases.json"
MANIFEST_PATH = HOLDOUT_ROOT / "dataset-manifest.json"
DEV_ROOT = EVAL_ROOT / "datasets" / "robot-vacuum-semantic-normalization-v1"


def _normalize(value: str) -> str:
    return " ".join(value.strip().rstrip(".!?。 ").split())


def _char_ngrams(value: str, n: int = 4) -> set[str]:
    compact = "".join(_normalize(value).split())
    return {compact[index : index + n] for index in range(max(0, len(compact) - n + 1))}


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 1.0


def _all_ids(cases: list[dict[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
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
            result[key].update(item[field] for item in collection)
    return result


def _texts(cases: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    facts = [
        fact["statement"]
        for case in cases
        for fact in case["model_view"]["facts"]
    ]
    propositions = [
        proposition["text"]
        for case in cases
        for proposition in case["expected"]["atomic_propositions"]
    ]
    return facts, propositions


def validate_holdout(cases: list[dict[str, Any]]) -> dict[str, Any]:
    schema = base._load(base.SCHEMA_PATH)
    validator = jsonschema.Draft202012Validator(schema)
    summary = base._validate_collection(
        cases,
        kind="normal",
        id_prefix="rv_semantic_holdout_",
        validator=validator,
    )
    base._require(summary["decision_counts"] == {"accepted": 64}, "holdout decisions")
    base._require(summary["fact_count_min"] >= 4, "holdout minimum fact count")
    base._require(summary["fact_count_max"] <= 14, "holdout maximum fact count")

    claim_groups: Counter[str] = Counter()
    distractor_cases = 0
    for case in cases:
        claim_ids = {
            fact["fact_id"]
            for fact in case["model_view"]["facts"]
            if fact["entity_kind"] == "claim"
        }
        base._require(len(claim_ids) == 1, f"{case['case_id']}: one claim required")
        groups = {
            proposition["capability_group"]
            for proposition in case["expected"]["atomic_propositions"]
            if proposition["fact_id"] in claim_ids
        }
        base._require(len(groups) == 1, f"{case['case_id']}: claim spans groups")
        claim_groups[next(iter(groups))] += 1
        distractor_cases += any(
            fact["entity_kind"] == "distractor" for fact in case["model_view"]["facts"]
        )
    base._require(
        claim_groups == Counter({group: 8 for group in base.CAPABILITY_GROUPS}),
        "holdout claim groups",
    )
    base._require(distractor_cases == 16, "holdout distractor case count")
    base._require(16 <= summary["split_fact_count"] <= 24, "holdout split fact count")

    dev_cases = [
        *base._load(DEV_ROOT / "normal-cases.json"),
        *base._load(DEV_ROOT / "defensive-cases.json"),
    ]
    holdout_ids = _all_ids(cases)
    dev_ids = _all_ids(dev_cases)
    for id_kind in holdout_ids:
        base._require(
            not (holdout_ids[id_kind] & dev_ids[id_kind]),
            f"holdout {id_kind} IDs overlap development data",
        )

    holdout_facts, holdout_props = _texts(cases)
    dev_facts, dev_props = _texts(dev_cases)
    exact_fact_overlap = set(map(_normalize, holdout_facts)) & set(map(_normalize, dev_facts))
    exact_prop_overlap = set(map(_normalize, holdout_props)) & set(map(_normalize, dev_props))
    base._require(not exact_fact_overlap, "holdout fact text overlaps development data")
    base._require(not exact_prop_overlap, "holdout proposition text overlaps development data")

    dev_ngrams = [(text, _char_ngrams(text)) for text in [*dev_facts, *dev_props]]
    max_surface_overlap = 0.0
    max_surface_pair: tuple[str, str] | None = None
    for text in [*holdout_facts, *holdout_props]:
        grams = _char_ngrams(text)
        for dev_text, dev_grams in dev_ngrams:
            score = _jaccard(grams, dev_grams)
            if score > max_surface_overlap:
                max_surface_overlap = score
                max_surface_pair = text, dev_text
    base._require(
        max_surface_overlap < 0.80,
        f"holdout has near-duplicate development text: {max_surface_pair}",
    )

    return {
        **summary,
        "claim_group_counts": dict(sorted(claim_groups.items())),
        "distractor_case_count": distractor_cases,
        "exact_development_fact_overlap_count": len(exact_fact_overlap),
        "exact_development_proposition_overlap_count": len(exact_prop_overlap),
        "maximum_development_surface_jaccard_4gram": max_surface_overlap,
        "maximum_development_surface_pair": list(max_surface_pair) if max_surface_pair else None,
        "development_id_overlap_count": {
            key: len(holdout_ids[key] & dev_ids[key]) for key in sorted(holdout_ids)
        },
    }


def main() -> None:
    cases = base._load(HOLDOUT_PATH)
    summary = validate_holdout(cases)
    if MANIFEST_PATH.exists():
        manifest = base._load(MANIFEST_PATH)
        base._require(manifest["summary"] == summary, "stale holdout manifest")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
