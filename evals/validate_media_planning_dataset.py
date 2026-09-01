from __future__ import annotations

import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).parent / "datasets" / "robot-vacuum-media-planning-v1"
SCHEMA_PATH = ROOT / "media-planning-case.schema.json"
VARIANT_PATH = ROOT / "product-variant-cases.json"
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
REQUIRED_GROUPS = {
    "product_identity_outcome",
    "cleaning_mechanism",
    "mobility_coverage",
}
PROFILE = {
    "product_identity_outcome": (1, 1, "required"),
    "problem_environment": (1, 1, "recommended"),
    "cleaning_mechanism": (1, 2, "required"),
    "mobility_coverage": (1, 2, "required"),
    "automation_return": (1, 1, "recommended"),
    "control_personalization": (1, 1, "recommended"),
    "evidence_performance": (1, 2, "recommended"),
    "configuration_maintenance": (1, 1, "optional"),
}
DEFENSIVE_TAGS = {
    "unknown_required_identity",
    "unknown_required_cleaning",
    "unknown_required_mobility",
    "optional_unknown_placeholder",
    "explicit_absence",
    "unsupported_evidence",
    "conflict_boundary",
    "missing_reference_assets",
}

PLANNER_INPUT_KEYS = {"input"}
HIDDEN_EVALUATION_KEYS = {
    "case_id",
    "case_kind",
    "variant",
    "template_id",
    "tags",
    "expected",
    "notes",
}
BOUNDARY_ONLY_INPUT_KEYS = {"source_boundary", "worker_revision", "template_revision"}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_planner_evaluation_views(
    case: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create separated planner-input and scorer-only views for one case.

    Boundary/revision checks and state normalization happen before this projection.
    The planner therefore receives authoritative capability groups and planner states,
    not raw fields from which an LLM could appear to infer them. Facts are rotated and
    assigned new opaque IDs so authoring position cannot affect the downstream plan.
    The scorer-only expected value is returned separately and must never be added to
    the planner payload.
    """

    case_number = int(case["case_id"].rsplit("_", maxsplit=1)[1])
    input_view = deepcopy(case["input"])
    expected_view = deepcopy(case["expected"])
    raw_mapping = {
        mapping["fact_id"]: mapping["capability_group"]
        for mapping in case["expected"]["fact_mappings"]
    }
    for fact in input_view["facts"]:
        fact["capability_group"] = raw_mapping[fact["fact_id"]]
        fact["planner_state"] = _planner_state(fact)
        for field in ("availability", "support_level", "revision"):
            fact.pop(field)
    for field in BOUNDARY_ONLY_INPUT_KEYS:
        input_view.pop(field)

    def permute_and_relabel(items: list[dict[str, Any]], prefix: str) -> dict[str, str]:
        if not items:
            return {}
        offset = (case_number - 1) % len(items)
        order = list(range(len(items)))
        if ((case_number - 1) // len(items)) % 2:
            order.reverse()
        order = order[offset:] + order[:offset]
        reordered = [items[index] for index in order]
        id_map: dict[str, str] = {}
        for index, item in enumerate(reordered, start=1):
            key = f"{prefix}_id"
            new_id = f"{prefix}_{index:02d}"
            id_map[item[key]] = new_id
            item[key] = new_id
        items[:] = reordered
        return id_map

    fact_id_map = permute_and_relabel(input_view["facts"], "fact")
    permute_and_relabel(input_view["available_assets"], "asset")
    for mapping in expected_view["fact_mappings"]:
        mapping["fact_id"] = fact_id_map[mapping["fact_id"]]
    planner_input = {"input": input_view}
    return planner_input, expected_view


def _planner_state(fact: dict[str, Any]) -> str:
    availability = fact["availability"]
    support = fact["support_level"]
    if availability == "conflicting":
        return "conflicting"
    if availability == "unknown":
        return "unknown"
    if availability == "explicitly_absent" or support == "rejected":
        return "inactive"
    if support == "supported":
        return "verified"
    return "unverified"


def _logic_signature(planner_view: dict[str, Any]) -> str:
    group_states: dict[str, list[str]] = defaultdict(list)
    for fact in planner_view["input"]["facts"]:
        group_states[fact["capability_group"]].append(fact["planner_state"])
    value = {
        "group_states": {key: sorted(values) for key, values in sorted(group_states.items())},
        "template_groups": sorted(planner_view["input"]["template_capability_groups"]),
        "assets": sorted(asset["role"] for asset in planner_view["input"]["available_assets"]),
        "collection_states": sorted(
            (fact["planner_state"], fact["collection_state"])
            for fact in planner_view["input"]["facts"]
        ),
    }
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _validate_case(case: dict[str, Any], validator: jsonschema.Draft202012Validator) -> None:
    errors = sorted(validator.iter_errors(case), key=lambda error: list(error.path))
    if errors:
        first = errors[0]
        raise AssertionError(f"{case.get('case_id', '<unknown>')}: schema: {first.message}")

    facts = case["input"]["facts"]
    mappings = case["expected"]["fact_mappings"]
    fact_ids = [fact["fact_id"] for fact in facts]
    mapping_ids = [mapping["fact_id"] for mapping in mappings]
    assert len(fact_ids) == len(set(fact_ids)), f"{case['case_id']}: duplicate fact_id"
    assert len(mapping_ids) == len(set(mapping_ids)), f"{case['case_id']}: duplicate mapping"
    assert set(fact_ids) == set(mapping_ids), f"{case['case_id']}: fact/mapping mismatch"

    mapping_by_id = {mapping["fact_id"]: mapping for mapping in mappings}
    worker_revision = case["input"]["worker_revision"]
    for fact in facts:
        actual = mapping_by_id[fact["fact_id"]]["planner_state"]
        expected = _planner_state(fact)
        assert actual == expected, (
            f"{case['case_id']}: {fact['fact_id']} state {actual} != {expected}"
        )
        if fact["availability"] == "provided" and fact["support_level"] == "supported":
            assert fact["source_refs"], f"{case['case_id']}: supported fact without source"
        assert fact["revision"] <= worker_revision, f"{case['case_id']}: future fact revision"
        if fact["availability"] in {"provided", "explicitly_absent"}:
            assert fact["collection_state"] == "resolved", (
                f"{case['case_id']}: resolved availability without resolved collection state"
            )
        if fact["availability"] == "unknown":
            assert fact["collection_state"] in {"not_offered", "requested", "skipped"}, (
                f"{case['case_id']}: invalid unknown collection state"
            )

    conflict = any(mapping["planner_state"] == "conflicting" for mapping in mappings)
    if case["input"]["source_boundary"] == "approved_worker_projection":
        assert not conflict, f"{case['case_id']}: conflict crossed approved worker boundary"
    else:
        assert conflict, f"{case['case_id']}: defensive injection without conflict"

    assets = case["input"]["available_assets"]
    asset_ids = [asset["asset_id"] for asset in assets]
    assert len(asset_ids) == len(set(asset_ids)), f"{case['case_id']}: duplicate asset_id"
    asset_signatures = [(asset["role"], tuple(sorted(asset["source_refs"]))) for asset in assets]
    assert len(asset_signatures) == len(set(asset_signatures)), (
        f"{case['case_id']}: duplicate asset role/source"
    )

    active = set(case["expected"]["active_groups"])
    placeholders = set(case["expected"]["placeholder_groups"])
    inactive = set(case["expected"]["inactive_groups"])
    assert not (active & placeholders or active & inactive or placeholders & inactive), (
        f"{case['case_id']}: group sets overlap"
    )
    assert active | placeholders | inactive == CAPABILITY_GROUPS, (
        f"{case['case_id']}: groups do not partition all capabilities"
    )
    template_groups = set(case["input"]["template_capability_groups"])
    assert active <= template_groups, f"{case['case_id']}: active group outside template"
    assert placeholders <= template_groups, f"{case['case_id']}: placeholder outside template"

    slot_bounds = case["expected"]["slot_bounds"]
    slot_groups = [slot["capability_group"] for slot in slot_bounds]
    assert len(slot_groups) == len(set(slot_groups)), f"{case['case_id']}: duplicate slot bounds"
    assert set(slot_groups) == active, f"{case['case_id']}: slot bounds != active groups"
    available_assets = {asset["role"] for asset in assets}
    required_assets: set[str] = set()
    for slot in slot_bounds:
        group = slot["capability_group"]
        expected_min, expected_max, expected_priority = PROFILE[group]
        assert (slot["min"], slot["max"], slot["priority"]) == (
            expected_min,
            expected_max,
            expected_priority,
        ), f"{case['case_id']}: invalid profile bounds for {group}"
        if slot["reference_policy"] == "required":
            assert slot["reference_asset_roles"], (
                f"{case['case_id']}: required reference without roles"
            )
            required_assets.update(slot["reference_asset_roles"])
    expected_missing = required_assets - available_assets
    assert set(case["expected"]["missing_reference_roles"]) == expected_missing, (
        f"{case['case_id']}: missing reference roles mismatch"
    )

    states_by_group: dict[str, set[str]] = defaultdict(set)
    fact_asset_roles_by_group: dict[str, set[str]] = defaultdict(set)
    for mapping in mappings:
        states_by_group[mapping["capability_group"]].add(mapping["planner_state"])
    for fact in facts:
        group = mapping_by_id[fact["fact_id"]]["capability_group"]
        fact_asset_roles_by_group[group].update(fact["asset_roles"])
    for group in active:
        allowed = {"verified"} if group == "evidence_performance" else {"verified", "unverified"}
        assert states_by_group[group] & allowed, f"{case['case_id']}: active {group} lacks support"
    for slot in slot_bounds:
        if slot["reference_policy"] == "required":
            assert (
                set(slot["reference_asset_roles"])
                <= fact_asset_roles_by_group[slot["capability_group"]]
            ), f"{case['case_id']}: slot reference role is not grounded in group facts"
    for group in placeholders:
        assert "unknown" in states_by_group[group], (
            f"{case['case_id']}: placeholder {group} lacks unknown fact"
        )
        assert any(
            mapping_by_id[fact["fact_id"]]["capability_group"] == group
            and fact["availability"] == "unknown"
            and fact["collection_state"] == "skipped"
            for fact in facts
        ), f"{case['case_id']}: placeholder {group} was not explicitly skipped"

    required_unknown = any("unknown" in states_by_group[group] for group in REQUIRED_GROUPS)
    if conflict:
        decision = "reject_conflict"
    elif required_unknown:
        decision = "needs_required_information"
    elif expected_missing:
        decision = "needs_reference_assets"
    elif placeholders:
        decision = "draft_with_placeholders"
    else:
        decision = "ready"
    assert case["expected"]["decision"] == decision, f"{case['case_id']}: decision mismatch"

    expected_flags = {
        "ready": (True, True),
        "draft_with_placeholders": (False, True),
        "needs_required_information": (False, False),
        "needs_reference_assets": (False, False),
        "reject_conflict": (False, False),
    }[decision]
    assert (
        case["expected"]["publishable"],
        case["expected"]["generation_allowed"],
    ) == expected_flags, f"{case['case_id']}: publishable/generation flags mismatch"


def _validate_collection(
    cases: list[dict[str, Any]],
    *,
    kind: str,
    id_prefix: str,
    minimum_input_signatures: int,
    validator: jsonschema.Draft202012Validator,
) -> dict[str, Any]:
    assert len(cases) == 64, f"{kind}: expected 64 cases"
    expected_ids = {f"{id_prefix}{index:03d}" for index in range(1, 65)}
    assert {case["case_id"] for case in cases} == expected_ids, f"{kind}: case ID sequence"
    assert all(case["case_kind"] == kind for case in cases), f"{kind}: wrong case_kind"
    visible_id_group_counts: dict[str, Counter[str]] = defaultdict(Counter)
    signatures: set[str] = set()
    for case in cases:
        planner_view, scorer_view = build_planner_evaluation_views(case)
        assert set(planner_view) == PLANNER_INPUT_KEYS
        assert not (HIDDEN_EVALUATION_KEYS & planner_view.keys())
        assert not (BOUNDARY_ONLY_INPUT_KEYS & planner_view["input"].keys())
        scorer_mapping = {
            mapping["fact_id"]: mapping["capability_group"]
            for mapping in scorer_view["fact_mappings"]
        }
        for fact in planner_view["input"]["facts"]:
            assert fact["capability_group"] == scorer_mapping[fact["fact_id"]]
            assert "availability" not in fact
            assert "support_level" not in fact
            assert "revision" not in fact
            visible_id_group_counts[fact["fact_id"]][scorer_mapping[fact["fact_id"]]] += 1
        signatures.add(_logic_signature(planner_view))
        _validate_case(case, validator)

    variant_counts = Counter(case["variant"] for case in cases)
    assert set(variant_counts.values()) == {16}, f"{kind}: unbalanced variants {variant_counts}"
    template_counts = Counter(case["template_id"] for case in cases)
    assert all(8 <= count <= 12 for count in template_counts.values()), (
        f"{kind}: unbalanced templates {template_counts}"
    )
    assert len(signatures) >= minimum_input_signatures, (
        f"{kind}: only {len(signatures)} input-derived logical combinations"
    )
    assert all(set(counts) == CAPABILITY_GROUPS for counts in visible_id_group_counts.values()), (
        f"{kind}: visible fact IDs reveal capability groups {visible_id_group_counts}"
    )
    return {
        "count": len(cases),
        "variant_counts": dict(sorted(variant_counts.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "decision_counts": dict(
            sorted(Counter(case["expected"]["decision"] for case in cases).items())
        ),
        "input_signature_count": len(signatures),
    }


def main() -> None:
    if not __debug__:
        raise RuntimeError("dataset validation must run without Python optimization (-O)")
    schema = _load(SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    variants = _load(VARIANT_PATH)
    defensive = _load(DEFENSIVE_PATH)
    manifest = _load(MANIFEST_PATH)

    assert all(
        case["input"]["template_revision"] == "robotic-floor-cleaner-v1"
        for case in [*variants, *defensive]
    ), "dataset: template revision must not encode the collection kind"

    variant_summary = _validate_collection(
        variants,
        kind="product_variant",
        id_prefix="rv_variant_",
        minimum_input_signatures=48,
        validator=validator,
    )
    assert variant_summary["decision_counts"] == {"ready": 64}
    assert all(
        not (
            {"unknown", "conflicting"}
            & {item["planner_state"] for item in case["expected"]["fact_mappings"]}
        )
        for case in variants
    ), "product_variant: unknown or conflicting state"
    assert all(
        case["input"]["source_boundary"] == "approved_worker_projection" for case in variants
    )
    assert all(
        any(item["planner_state"] == "unverified" for item in case["expected"]["fact_mappings"])
        for case in variants
    ), "product_variant: every case needs an unverified maker statement"
    assert (
        sum(
            item["planner_state"] == "unverified"
            for case in variants
            for item in case["expected"]["fact_mappings"]
        )
        >= 64
    ), "product_variant: insufficient unverified fact coverage"

    defensive_summary = _validate_collection(
        defensive,
        kind="defensive",
        id_prefix="rv_defensive_",
        minimum_input_signatures=24,
        validator=validator,
    )
    tag_counts = Counter()
    per_tag_variants: dict[str, Counter[str]] = defaultdict(Counter)
    for case in defensive:
        primary = DEFENSIVE_TAGS & set(case["tags"])
        assert len(primary) == 1, f"{case['case_id']}: expected one defensive primary tag"
        tag = next(iter(primary))
        tag_counts[tag] += 1
        per_tag_variants[tag][case["variant"]] += 1
    assert tag_counts == Counter({tag: 8 for tag in DEFENSIVE_TAGS}), tag_counts
    assert all(set(counts.values()) == {2} for counts in per_tag_variants.values()), (
        per_tag_variants
    )
    assert defensive_summary["decision_counts"] == {
        "draft_with_placeholders": 8,
        "needs_reference_assets": 8,
        "needs_required_information": 24,
        "ready": 16,
        "reject_conflict": 8,
    }
    missing_role_counts = Counter(
        role
        for case in defensive
        if "missing_reference_assets" in case["tags"]
        for role in case["expected"]["missing_reference_roles"]
    )
    assert missing_role_counts == Counter(
        {
            "product_body": 2,
            "dock": 2,
            "accessory": 2,
            "control_interface": 1,
            "provided_evidence": 1,
        }
    ), missing_role_counts
    placeholder_group_counts = Counter(
        group
        for case in defensive
        if "optional_unknown_placeholder" in case["tags"]
        for group in case["expected"]["placeholder_groups"]
    )
    assert placeholder_group_counts == Counter(
        {
            "problem_environment": 2,
            "automation_return": 2,
            "control_personalization": 2,
            "evidence_performance": 1,
            "configuration_maintenance": 1,
        }
    ), placeholder_group_counts

    for key, summary in (
        ("product_variant", variant_summary),
        ("defensive", defensive_summary),
    ):
        assert manifest[key]["case_count"] == summary["count"], f"manifest: stale {key} case_count"
        assert manifest[key]["input_signature_count"] == summary["input_signature_count"], (
            f"manifest: stale {key} input_signature_count"
        )
        assert manifest[key]["decision_counts"] == summary["decision_counts"], (
            f"manifest: stale {key} decision_counts"
        )

    print(
        json.dumps(
            {"product_variant": variant_summary, "defensive": defensive_summary},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
