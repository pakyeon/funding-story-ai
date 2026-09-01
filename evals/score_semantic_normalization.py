from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jsonschema

DATASET_ROOT = Path(__file__).parent / "datasets" / "robot-vacuum-semantic-normalization-v1"
CASE_SCHEMA_PATH = DATASET_ROOT / "semantic-normalization-case.schema.json"
NORMAL_PATH = DATASET_ROOT / "normal-cases.json"
DEFENSIVE_PATH = DATASET_ROOT / "defensive-cases.json"
HOLDOUT_PATH = (
    Path(__file__).parent
    / "datasets"
    / "robot-vacuum-semantic-normalization-holdout-v1"
    / "holdout-cases.json"
)
ADVERSARIAL_PATH = (
    Path(__file__).parent
    / "datasets"
    / "robot-vacuum-semantic-normalization-adversarial-v1"
    / "adversarial-cases.json"
)

ALL_GROUPS = (
    "product_identity_outcome",
    "problem_environment",
    "cleaning_mechanism",
    "mobility_coverage",
    "automation_return",
    "control_personalization",
    "evidence_performance",
    "configuration_maintenance",
)
FEATURE_GROUPS = (
    "cleaning_mechanism",
    "mobility_coverage",
    "automation_return",
    "control_personalization",
    "configuration_maintenance",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_validator() -> jsonschema.Draft202012Validator:
    case_schema = _load(CASE_SCHEMA_PATH)
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": case_schema["$defs"],
        "type": "object",
        "additionalProperties": False,
        "required": ["schema_version", "predictions"],
        "properties": {
            "schema_version": {"const": "semantic-normalization-predictions-v1"},
            "predictions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["case_id", "output"],
                    "properties": {
                        "case_id": {"type": "string"},
                        "output": {"$ref": "#/$defs/expected"},
                    },
                },
            },
        },
    }
    return jsonschema.Draft202012Validator(schema)


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().rstrip(".!?。").split())


def _proposition_key(value: dict[str, Any]) -> tuple[str, str]:
    return value["fact_id"], _normalize_text(value["text"])


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _set_metrics(
    predicted: set[tuple[str, str]], gold: set[tuple[str, str]]
) -> dict[str, float | int]:
    true_positive = len(predicted & gold)
    precision = _ratio(true_positive, len(predicted))
    recall = _ratio(true_positive, len(gold))
    return {
        "true_positive": true_positive,
        "false_positive": len(predicted - gold),
        "false_negative": len(gold - predicted),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _group_metrics(
    *,
    gold_labels: dict[tuple[str, str], str],
    predicted_labels: dict[tuple[str, str], str],
    fact_kinds: dict[str, str],
    included_kinds: set[str],
    labels: tuple[str, ...],
) -> dict[str, Any]:
    per_group: dict[str, dict[str, float | int]] = {}
    for label in labels:
        gold = {
            key
            for key, value in gold_labels.items()
            if fact_kinds.get(key[0]) in included_kinds and value == label
        }
        predicted = {
            key
            for key, value in predicted_labels.items()
            if fact_kinds.get(key[0]) in included_kinds and value == label
        }
        per_group[label] = _set_metrics(predicted, gold)
    return {
        "macro_f1": sum(float(value["f1"]) for value in per_group.values()) / len(per_group),
        "per_group": per_group,
    }


def _media_views(
    media_facts: list[dict[str, Any]],
    proposition_by_id: dict[str, tuple[str, str]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    full: dict[str, Any] = {}
    references: dict[str, Any] = {}
    states: dict[str, Any] = {}
    invalid_proposition_refs = 0
    for item in media_facts:
        proposition_keys = []
        for proposition_id in item["proposition_ids"]:
            if proposition_id not in proposition_by_id:
                invalid_proposition_refs += 1
                proposition_keys.append(("<invalid>", proposition_id))
            else:
                proposition_keys.append(proposition_by_id[proposition_id])
        fact_id = item["fact_id"]
        references[fact_id] = (
            tuple(sorted(item["source_refs"])),
            tuple(sorted(item["evidence_refs"])),
            tuple(sorted(item["asset_refs"])),
        )
        states[fact_id] = (
            item["availability"],
            item["support_level"],
            item["collection_state"],
        )
        full[fact_id] = (
            tuple(sorted(proposition_keys)),
            references[fact_id],
            states[fact_id],
        )
    return full, references, states, invalid_proposition_refs


def score_predictions(
    predictions: dict[str, Any], gold_cases: list[dict[str, Any]]
) -> dict[str, Any]:
    errors = sorted(
        _prediction_validator().iter_errors(predictions), key=lambda error: list(error.path)
    )
    if errors:
        path = ".".join(str(item) for item in errors[0].absolute_path)
        raise ValueError(f"prediction schema error at {path}: {errors[0].message}")

    predicted_cases = predictions["predictions"]
    predicted_ids = [item["case_id"] for item in predicted_cases]
    if len(predicted_ids) != len(set(predicted_ids)):
        raise ValueError("prediction case IDs must be unique")
    gold_by_id = {case["case_id"]: case for case in gold_cases}
    if set(predicted_ids) != set(gold_by_id):
        missing = sorted(set(gold_by_id) - set(predicted_ids))
        extra = sorted(set(predicted_ids) - set(gold_by_id))
        raise ValueError(f"prediction coverage mismatch: missing={missing} extra={extra}")

    gold_keys: set[tuple[str, str]] = set()
    predicted_keys: set[tuple[str, str]] = set()
    gold_labels: dict[tuple[str, str], str] = {}
    predicted_labels: dict[tuple[str, str], str] = {}
    fact_kinds: dict[str, str] = {}
    duplicate_propositions = 0
    duplicate_proposition_ids = 0
    invented_fact_ids: set[str] = set()
    ignored_exact_cases = 0
    decomposition_exact_cases = 0
    decision_exact_cases = 0
    accepted_cases = 0
    media_full_exact = 0
    media_reference_exact = 0
    media_state_exact = 0
    rejected_media_violations = 0
    rejected_semantic_invocations = 0
    invalid_proposition_refs = 0

    prediction_by_id = {item["case_id"]: item["output"] for item in predicted_cases}
    for case_id, case in gold_by_id.items():
        output = prediction_by_id[case_id]
        gold = case["expected"]
        case_fact_kinds = {
            item["fact_id"]: item["entity_kind"] for item in case["model_view"]["facts"]
        }
        fact_kinds.update(case_fact_kinds)
        gold_props = gold["atomic_propositions"]
        predicted_props = output["atomic_propositions"]
        case_gold_keys = {_proposition_key(item) for item in gold_props}
        case_predicted_list = [_proposition_key(item) for item in predicted_props]
        case_predicted_keys = set(case_predicted_list)
        duplicate_propositions += len(case_predicted_list) - len(case_predicted_keys)
        predicted_prop_id_list = [item["proposition_id"] for item in predicted_props]
        duplicate_proposition_ids += len(predicted_prop_id_list) - len(
            set(predicted_prop_id_list)
        )
        invented_fact_ids.update(
            key[0] for key in case_predicted_keys if key[0] not in case_fact_kinds
        )
        gold_keys.update(case_gold_keys)
        predicted_keys.update(case_predicted_keys)
        gold_labels.update(
            {_proposition_key(item): item["capability_group"] for item in gold_props}
        )
        for item in predicted_props:
            predicted_labels.setdefault(_proposition_key(item), item["capability_group"])
        decomposition_exact_cases += case_predicted_keys == case_gold_keys
        ignored_exact_cases += set(output["ignored_fact_ids"]) == set(gold["ignored_fact_ids"])
        invented_fact_ids.update(
            fact_id for fact_id in output["ignored_fact_ids"] if fact_id not in case_fact_kinds
        )

        decision_exact_cases += output["adapter"] == gold["adapter"]
        gold_prop_ids = {item["proposition_id"]: _proposition_key(item) for item in gold_props}
        predicted_prop_ids = {
            item["proposition_id"]: _proposition_key(item) for item in predicted_props
        }
        gold_full, gold_refs, gold_states, gold_invalid = _media_views(
            gold["media_facts"], gold_prop_ids
        )
        predicted_full, predicted_refs, predicted_states, predicted_invalid = _media_views(
            output["media_facts"], predicted_prop_ids
        )
        invalid_proposition_refs += gold_invalid + predicted_invalid
        invented_fact_ids.update(
            fact_id for fact_id in predicted_full if fact_id not in case_fact_kinds
        )
        if gold["adapter"]["decision"] == "accepted":
            accepted_cases += 1
            media_full_exact += predicted_full == gold_full
            media_reference_exact += predicted_refs == gold_refs
            media_state_exact += predicted_states == gold_states
        else:
            if output["media_facts"]:
                rejected_media_violations += 1
            if output["atomic_propositions"] or output["ignored_fact_ids"]:
                rejected_semantic_invocations += 1

    decomposition = _set_metrics(predicted_keys, gold_keys)
    all_group = _group_metrics(
        gold_labels=gold_labels,
        predicted_labels=predicted_labels,
        fact_kinds=fact_kinds,
        included_kinds={"product", "problem", "feature", "claim", "evidence"},
        labels=ALL_GROUPS,
    )
    feature_group = _group_metrics(
        gold_labels=gold_labels,
        predicted_labels=predicted_labels,
        fact_kinds=fact_kinds,
        included_kinds={"feature"},
        labels=FEATURE_GROUPS,
    )
    claim_group = _group_metrics(
        gold_labels=gold_labels,
        predicted_labels=predicted_labels,
        fact_kinds=fact_kinds,
        included_kinds={"claim"},
        labels=ALL_GROUPS,
    )
    case_count = len(gold_cases)
    result = {
        "case_count": case_count,
        "decomposition": decomposition,
        "decomposition_exact_case_rate": _ratio(decomposition_exact_cases, case_count),
        "ignored_fact_exact_case_rate": _ratio(ignored_exact_cases, case_count),
        "capability_group_all": all_group,
        "capability_group_feature": feature_group,
        "capability_group_claim": claim_group,
        "adapter_decision_exact_rate": _ratio(decision_exact_cases, case_count),
        "accepted_case_count": accepted_cases,
        "accepted_media_full_exact_rate": _ratio(media_full_exact, accepted_cases),
        "accepted_reference_join_exact_rate": _ratio(media_reference_exact, accepted_cases),
        "accepted_state_join_exact_rate": _ratio(media_state_exact, accepted_cases),
        "duplicate_proposition_count": duplicate_propositions,
        "duplicate_proposition_id_count": duplicate_proposition_ids,
        "invented_fact_id_count": len(invented_fact_ids),
        "invalid_proposition_reference_count": invalid_proposition_refs,
        "rejected_media_violation_count": rejected_media_violations,
        "rejected_semantic_invocation_count": rejected_semantic_invocations,
    }
    result["semantic_gate"] = bool(
        decomposition["f1"] >= 0.85
        and feature_group["macro_f1"] >= 0.90
        and claim_group["macro_f1"] >= 0.90
        and duplicate_propositions == 0
        and duplicate_proposition_ids == 0
        and not invented_fact_ids
        and invalid_proposition_refs == 0
    )
    result["boundary_safety_gate"] = bool(
        result["adapter_decision_exact_rate"] == 1.0
        and rejected_media_violations == 0
        and rejected_semantic_invocations == 0
    )
    result["adapter_join_gate"] = bool(
        result["accepted_media_full_exact_rate"] == 1.0
        and result["accepted_reference_join_exact_rate"] == 1.0
        and result["accepted_state_join_exact_rate"] == 1.0
        and invalid_proposition_refs == 0
    )
    result["adapter_safety_gate"] = (
        result["boundary_safety_gate"] and result["adapter_join_gate"]
    )
    result["meets_gate"] = result["semantic_gate"] and result["adapter_safety_gate"]
    return result


def _gold_cases(split: str) -> list[dict[str, Any]]:
    normal = _load(NORMAL_PATH)
    defensive = _load(DEFENSIVE_PATH)
    if split == "normal":
        return normal
    if split == "defensive":
        return defensive
    if split == "holdout":
        return _load(HOLDOUT_PATH)
    if split == "adversarial":
        return _load(ADVERSARIAL_PATH)
    return [*normal, *defensive]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score semantic-normalization predictions")
    parser.add_argument("predictions", type=Path)
    parser.add_argument(
        "--split",
        choices=("normal", "defensive", "holdout", "adversarial", "all"),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    result = score_predictions(_load(args.predictions), _gold_cases(args.split))
    if args.output is not None:
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.fail_on_gate and not result["meets_gate"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
