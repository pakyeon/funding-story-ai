from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).parent / "datasets" / "robot-vacuum-semantic-normalization-v1"
SCHEMA_PATH = ROOT / "semantic-normalization-case.schema.json"
NORMAL_PATH = ROOT / "normal-cases.json"
DEFENSIVE_PATH = ROOT / "defensive-cases.json"
MANIFEST_PATH = ROOT / "dataset-manifest.json"

CAPABILITY_GROUPS = {
    "product_identity_outcome",
    "problem_environment",
    "cleaning_mechanism",
    "mobility_coverage",
    "automation_return",
    "control_personalization",
    "evidence_performance",
    "configuration_maintenance",
}
KIND_GROUPS = {
    "product": {"product_identity_outcome"},
    "problem": {"problem_environment"},
    "evidence": {"evidence_performance"},
    "feature": {
        "cleaning_mechanism",
        "mobility_coverage",
        "automation_return",
        "control_personalization",
        "configuration_maintenance",
    },
    "claim": CAPABILITY_GROUPS,
    "distractor": set(),
}
MODEL_VIEW_KEYS = {"facts"}
FORBIDDEN_MODEL_KEYS = {
    "case_id",
    "case_kind",
    "variant",
    "tags",
    "expected",
    "notes",
    "template_id",
    "source_boundary",
    "approved_revision",
    "revision",
    "brief_digest",
    "worker_projection_digest",
    "availability",
    "support_level",
    "collection_state",
    "capability_group",
}
DEFENSIVE_TYPES = {
    "conflict": 32,
    "stale_revision": 8,
    "digest_mismatch": 8,
    "forged_link": 8,
    "prompt_injection": 8,
}
FAILURE_CODE_BY_DECISION = {
    "accepted": None,
    "reject_conflict": "conflicting_worker_state",
    "reject_stale_revision": "stale_revision",
    "reject_digest_mismatch": "digest_mismatch",
    "reject_forged_reference": "forged_reference",
}
GROUP_SOURCE_TYPES = {
    "product_identity_outcome": "maker-input",
    "problem_environment": "maker-input",
    "cleaning_mechanism": "document",
    "mobility_coverage": "document",
    "automation_return": "document",
    "control_personalization": "document",
    "evidence_performance": "test-report",
    "configuration_maintenance": "document",
}
GROUP_ASSET_TYPES = {
    "product_identity_outcome": "product",
    "problem_environment": "lifestyle",
    "cleaning_mechanism": "feature",
    "mobility_coverage": "feature",
    "automation_return": "feature",
    "control_personalization": "app",
    "evidence_performance": "evidence",
    "configuration_maintenance": "feature",
}
INJECTION_PREFIX = re.compile(r'^인용된 메모에는 ".*?"라고 적혀 있다\.\s*')
QUOTED_FACT = re.compile(r"인용된 사실은\s*(.+?[.!?。])(?:\s|$)")


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_all_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value), set())
    return set()


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _all_strings(child)]
    if isinstance(value, list):
        return [item for child in value for item in _all_strings(child)]
    return [value] if isinstance(value, str) else []


def _ids(items: list[dict[str, Any]], key: str) -> set[str]:
    values = [item[key] for item in items]
    _require(len(values) == len(set(values)), f"duplicate {key}")
    return set(values)


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _normalize_clause(value: str) -> str:
    return value.strip().rstrip(".!?。").strip()


def _is_grounded(proposition: str, statement: str) -> bool:
    """Require a complete clause rather than accepting any matching substring."""
    normalized = _normalize_clause(proposition)
    body = INJECTION_PREFIX.sub("", statement)
    clauses = {_normalize_clause(clause) for clause in body.split(" 그리고 ")}
    clauses.update(_normalize_clause(match) for match in QUOTED_FACT.findall(statement))
    return len(normalized) >= 10 and normalized in clauses


def _unresolved_links(case: dict[str, Any]) -> set[str]:
    projection = case["input"]["approved_entity_projection"]
    source_ids = _ids(projection["sources"], "source_id")
    evidence_ids = _ids(projection["evidence"], "evidence_id")
    asset_ids = _ids(projection["assets"], "asset_id")
    unresolved: set[str] = set()

    for item in [*projection["entities"], *projection["facts"]]:
        unresolved.update(set(item["source_refs"]) - source_ids)
        unresolved.update(set(item["evidence_refs"]) - evidence_ids)
        unresolved.update(set(item["asset_refs"]) - asset_ids)
    for item in [*projection["evidence"], *projection["assets"]]:
        unresolved.update(set(item["source_refs"]) - source_ids)
    return unresolved


def _expected_decision(case: dict[str, Any], unresolved: set[str]) -> str:
    input_value = case["input"]
    worker = input_value["worker_projection"]
    if any(item["availability"] == "conflicting" for item in worker["fact_states"]):
        return "reject_conflict"
    if worker["revision"] != input_value["approved_revision"]:
        return "reject_stale_revision"
    if worker["brief_digest"] != input_value["brief_digest"]:
        return "reject_digest_mismatch"
    if unresolved:
        return "reject_forged_reference"
    return "accepted"


def _validate_case(
    case: dict[str, Any], validator: jsonschema.Draft202012Validator
) -> dict[str, Any]:
    errors = sorted(validator.iter_errors(case), key=lambda error: list(error.path))
    if errors:
        raise AssertionError(f"{case.get('case_id', '<unknown>')}: schema: {errors[0].message}")
    case_id = case["case_id"]
    projection = case["input"]["approved_entity_projection"]
    worker = case["input"]["worker_projection"]
    model_view = case["model_view"]
    expected = case["expected"]

    _require(
        case["input"]["brief_digest"] == _digest(projection),
        f"{case_id}: brief digest is not canonical",
    )
    _require(
        case["input"]["worker_projection_digest"] == _digest(worker),
        f"{case_id}: worker projection digest is not canonical",
    )
    _require(
        all(state["revision"] == worker["revision"] for state in worker["fact_states"]),
        f"{case_id}: fact-state revision differs from worker revision",
    )

    _require(set(model_view) == MODEL_VIEW_KEYS, f"{case_id}: unexpected model-view keys")
    leaked = FORBIDDEN_MODEL_KEYS & _all_keys(model_view)
    _require(not leaked, f"{case_id}: model-view leakage {sorted(leaked)}")
    leaked_group_tokens = {
        group
        for group in CAPABILITY_GROUPS
        if any(group in value for value in _all_strings(model_view))
    }
    _require(
        not leaked_group_tokens,
        f"{case_id}: capability-group token leaked into model text",
    )
    expected_model_facts = [
        {
            "fact_id": fact["fact_id"],
            "entity_kind": fact["entity_kind"],
            "statement": fact["statement"],
        }
        for fact in projection["facts"]
    ]
    _require(
        model_view["facts"] == expected_model_facts,
        f"{case_id}: model-view facts differ from the sanitized projection",
    )

    entity_ids = _ids(projection["entities"], "entity_id")
    fact_ids = _ids(projection["facts"], "fact_id")
    state_ids = _ids(worker["fact_states"], "fact_id")
    _require(fact_ids == state_ids, f"{case_id}: fact/state join mismatch")
    _require(
        all(item["entity_id"] in entity_ids for item in projection["facts"]),
        f"{case_id}: fact points to an unknown entity",
    )
    _require(
        len({item["statement"] for item in projection["facts"]}) == len(projection["facts"]),
        f"{case_id}: duplicate fact statements inflate the denominator",
    )

    unresolved = _unresolved_links(case)
    decision = _expected_decision(case, unresolved)
    _require(expected["adapter"]["decision"] == decision, f"{case_id}: adapter decision")
    _require(
        expected["adapter"]["failure_code"] == FAILURE_CODE_BY_DECISION[decision],
        f"{case_id}: adapter failure code leaks metadata or differs from the boundary decision",
    )
    if decision == "reject_conflict":
        _require(
            case["input"]["source_boundary"] == "defensive_boundary_injection",
            f"{case_id}: conflict is not marked as a defensive injection",
        )
    else:
        _require(
            case["input"]["source_boundary"] == "approved_story_brief_worker_projection",
            f"{case_id}: invalid approved source boundary",
        )

    propositions = expected["atomic_propositions"]
    proposition_ids = _ids(propositions, "proposition_id")
    fact_by_id = {item["fact_id"]: item for item in projection["facts"]}
    source_types = {item["source_id"]: item["source_type"] for item in projection["sources"]}
    evidence_types = {item["evidence_id"]: item["evidence_type"] for item in projection["evidence"]}
    asset_types = {item["asset_id"]: item["asset_type"] for item in projection["assets"]}
    propositions_by_fact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposition in propositions:
        fact_id = proposition["fact_id"]
        _require(fact_id in fact_ids, f"{case_id}: proposition points to an unknown fact")
        fact = fact_by_id[fact_id]
        _require(
            proposition["capability_group"] in KIND_GROUPS[fact["entity_kind"]],
            f"{case_id}: {fact['entity_kind']} fact has incompatible group "
            f"{proposition['capability_group']}",
        )
        _require(
            _is_grounded(proposition["text"], fact["statement"]),
            f"{case_id}: proposition not grounded",
        )
        group = proposition["capability_group"]
        resolved_sources = [source_types[ref] for ref in fact["source_refs"] if ref in source_types]
        resolved_evidence = [
            evidence_types[ref] for ref in fact["evidence_refs"] if ref in evidence_types
        ]
        resolved_assets = [asset_types[ref] for ref in fact["asset_refs"] if ref in asset_types]
        if fact["entity_kind"] != "claim":
            _require(
                all(value == GROUP_SOURCE_TYPES[group] for value in resolved_sources),
                f"{case_id}: source type is inconsistent with {group}",
            )
            expected_evidence_type = (
                "test-report" if group == "evidence_performance" else "demonstration"
            )
            _require(
                all(value == expected_evidence_type for value in resolved_evidence),
                f"{case_id}: evidence type is inconsistent with {group}",
            )
            _require(
                all(value == GROUP_ASSET_TYPES[group] for value in resolved_assets),
                f"{case_id}: asset type is inconsistent with {group}",
            )
        propositions_by_fact[fact_id].append(proposition)

    ignored = set(expected["ignored_fact_ids"])
    _require(ignored <= fact_ids, f"{case_id}: ignored unknown fact")
    _require(not (ignored & set(propositions_by_fact)), f"{case_id}: ignored fact was classified")
    if decision == "accepted":
        _require(
            set(propositions_by_fact) | ignored == fact_ids,
            f"{case_id}: facts are not partitioned into propositions and ignored facts",
        )
        _require(
            all(fact_by_id[fact_id]["entity_kind"] == "distractor" for fact_id in ignored),
            f"{case_id}: non-distractor fact was ignored",
        )
    else:
        _require(
            not propositions and not ignored,
            f"{case_id}: rejected boundary invoked semantic classification",
        )
    _require(
        all(len(items) <= 2 for items in propositions_by_fact.values()),
        f"{case_id}: one fact has more than two atomic propositions",
    )
    split_count = sum(len(items) > 1 for items in propositions_by_fact.values())
    _require(split_count <= 3, f"{case_id}: compound pattern affects too many facts")

    media_facts = expected["media_facts"]
    media_fact_ids = _ids(media_facts, "fact_id") if media_facts else set()
    states = {item["fact_id"]: item for item in worker["fact_states"]}
    for fact_id, state in states.items():
        fact = fact_by_id[fact_id]
        if state["availability"] != "provided":
            expected_support = "none"
        elif fact["evidence_refs"]:
            expected_support = "supported"
        else:
            expected_support = "maker_stated"
        _require(
            state["support_level"] == expected_support,
            f"{case_id}: support level is not grounded in evidence availability",
        )
    if decision == "accepted":
        _require(
            media_fact_ids == set(propositions_by_fact),
            f"{case_id}: accepted media facts do not cover classified facts",
        )
        for media_fact in media_facts:
            fact = fact_by_id[media_fact["fact_id"]]
            state = states[media_fact["fact_id"]]
            for key in ("availability", "support_level", "collection_state"):
                _require(media_fact[key] == state[key], f"{case_id}: state join changed {key}")
            _require(
                set(media_fact["proposition_ids"])
                == {item["proposition_id"] for item in propositions_by_fact[media_fact["fact_id"]]},
                f"{case_id}: media fact proposition join mismatch",
            )
            _require(
                set(media_fact["source_refs"]) == set(fact["source_refs"])
                and set(media_fact["evidence_refs"]) == set(fact["evidence_refs"])
                and set(media_fact["asset_refs"]) == set(fact["asset_refs"]),
                f"{case_id}: media fact does not preserve approved references",
            )
    else:
        _require(not media_facts, f"{case_id}: rejected input produced media facts")

    return {
        "decision": decision,
        "unresolved": unresolved,
        "fact_count": len(projection["facts"]),
        "entity_count": len(projection["entities"]),
        "split_fact_count": split_count,
        "groups": [item["capability_group"] for item in propositions],
        "entity_kinds": [item["entity_kind"] for item in projection["facts"]],
        "fact_order": tuple(item["fact_id"] for item in model_view["facts"]),
        "ids": {
            "entity": entity_ids,
            "fact": fact_ids,
            "source": _ids(projection["sources"], "source_id"),
            "evidence": _ids(projection["evidence"], "evidence_id"),
            "asset": _ids(projection["assets"], "asset_id"),
            "proposition": proposition_ids,
        },
    }


def _validate_collection(
    cases: list[dict[str, Any]],
    *,
    kind: str,
    id_prefix: str,
    validator: jsonschema.Draft202012Validator,
) -> dict[str, Any]:
    _require(len(cases) == 64, f"{kind}: expected 64 cases")
    _require(
        {case["case_id"] for case in cases}
        == {f"{id_prefix}{index:03d}" for index in range(1, 65)},
        f"{kind}: case ID sequence",
    )
    _require(all(case["case_kind"] == kind for case in cases), f"{kind}: case kind")
    variant_counts = Counter(case["variant"] for case in cases)
    _require(set(variant_counts.values()) == {16}, f"{kind}: unbalanced variants")

    summaries = [_validate_case(case, validator) for case in cases]
    global_ids: dict[str, set[str]] = defaultdict(set)
    for summary in summaries:
        for key, values in summary["ids"].items():
            _require(not (global_ids[key] & values), f"{kind}: repeated global {key} ID")
            global_ids[key].update(values)

    return {
        "case_count": len(cases),
        "variant_counts": dict(sorted(variant_counts.items())),
        "decision_counts": dict(sorted(Counter(item["decision"] for item in summaries).items())),
        "fact_count_min": min(item["fact_count"] for item in summaries),
        "fact_count_max": max(item["fact_count"] for item in summaries),
        "entity_count_min": min(item["entity_count"] for item in summaries),
        "entity_count_max": max(item["entity_count"] for item in summaries),
        "split_fact_count": sum(item["split_fact_count"] for item in summaries),
        "unique_model_fact_orders": len({item["fact_order"] for item in summaries}),
        "capability_group_counts": dict(
            sorted(Counter(group for item in summaries for group in item["groups"]).items())
        ),
        "entity_kind_counts": dict(
            sorted(Counter(kind for item in summaries for kind in item["entity_kinds"]).items())
        ),
        "unresolved_case_count": sum(bool(item["unresolved"]) for item in summaries),
        "unresolved_reference_count": len(set().union(*(item["unresolved"] for item in summaries))),
    }


def main() -> None:
    schema = _load(SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    normal = _load(NORMAL_PATH)
    defensive = _load(DEFENSIVE_PATH)
    manifest = _load(MANIFEST_PATH)

    normal_summary = _validate_collection(
        normal,
        kind="normal",
        id_prefix="rv_semantic_normal_",
        validator=validator,
    )
    defensive_summary = _validate_collection(
        defensive,
        kind="defensive",
        id_prefix="rv_semantic_defensive_",
        validator=validator,
    )

    _require(normal_summary["decision_counts"] == {"accepted": 64}, "normal decisions")
    _require(normal_summary["fact_count_min"] == 4, "normal minimum fact count")
    _require(normal_summary["fact_count_max"] == 14, "normal maximum fact count")
    _require(normal_summary["entity_kind_counts"].get("claim") == 64, "normal claim slice")

    def claim_group_counts(cases: list[dict[str, Any]]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for case in cases:
            if case["expected"]["adapter"]["decision"] != "accepted":
                continue
            claim_ids = {
                fact["fact_id"]
                for fact in case["model_view"]["facts"]
                if fact["entity_kind"] == "claim"
            }
            _require(len(claim_ids) == 1, f"{case['case_id']}: expected one claim fact")
            groups = {
                proposition["capability_group"]
                for proposition in case["expected"]["atomic_propositions"]
                if proposition["fact_id"] in claim_ids
            }
            _require(len(groups) == 1, f"{case['case_id']}: claim spans multiple groups")
            counts[next(iter(groups))] += 1
        return counts

    expected_claim_groups = Counter(
        {
            "automation_return": 8,
            "cleaning_mechanism": 8,
            "configuration_maintenance": 9,
            "control_personalization": 8,
            "evidence_performance": 7,
            "mobility_coverage": 8,
            "problem_environment": 8,
            "product_identity_outcome": 8,
        }
    )
    _require(claim_group_counts(normal) == expected_claim_groups, "normal claim groups")
    _require(
        claim_group_counts(defensive) == Counter({group: 1 for group in CAPABILITY_GROUPS}),
        "accepted defensive claim groups",
    )
    _require(not normal_summary["unresolved_case_count"], "normal unresolved links")
    _require(
        normal_summary["unique_model_fact_orders"] == 64,
        "normal fact-order diversity",
    )

    primary_counts = Counter()
    for case in defensive:
        primary = set(case["tags"]) & set(DEFENSIVE_TYPES)
        _require(len(primary) == 1, f"{case['case_id']}: defensive primary type")
        primary_counts[next(iter(primary))] += 1
    _require(primary_counts == Counter(DEFENSIVE_TYPES), "defensive type distribution")
    _require(
        defensive_summary["decision_counts"]
        == {
            "accepted": 8,
            "reject_conflict": 32,
            "reject_digest_mismatch": 8,
            "reject_forged_reference": 8,
            "reject_stale_revision": 8,
        },
        "defensive decisions",
    )
    _require(defensive_summary["unresolved_case_count"] == 8, "forged-link cases")
    _require(defensive_summary["unresolved_reference_count"] == 8, "forged references")

    for key, summary in (("normal", normal_summary), ("defensive", defensive_summary)):
        for field in (
            "case_count",
            "variant_counts",
            "fact_count_min",
            "fact_count_max",
            "entity_count_min",
            "entity_count_max",
            "unique_model_fact_orders",
            "capability_group_counts",
        ):
            _require(manifest[key][field] == summary[field], f"manifest: stale {key}.{field}")

    print(json.dumps({"normal": normal_summary, "defensive": defensive_summary}, indent=2))


if __name__ == "__main__":
    main()
