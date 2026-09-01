from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .data_repository import DataRepository


class GenerationBoundaryError(ValueError):
    """Raised when an input does not belong to one approved worker revision."""


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _stable_id(prefix: str, *parts: str) -> str:
    encoded = "\0".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(encoded).hexdigest()[:12]}"


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _asset_roles(asset: dict[str, Any]) -> list[str]:
    asset_type = asset["asset_type"]
    description = asset["description"].lower()
    roles: list[str] = []
    if asset_type in {"product", "lifestyle"}:
        roles.append("product_body")
    if asset_type in {"feature", "reward"}:
        roles.append("accessory")
    if asset_type == "app":
        roles.append("control_interface")
    if asset_type == "evidence":
        roles.append("provided_evidence")
    if asset_type == "product" and any(
        keyword in description for keyword in ("도크", "스테이션", "dock")
    ):
        roles.append("dock")
    return _unique(roles)


def _support_level(kind: str, entity: dict[str, Any], sources: set[str]) -> str:
    if kind == "claim":
        status = entity.get("status")
        if status == "supported":
            return "supported"
        if status == "rejected":
            return "rejected"
        return "supported" if sources else "maker_stated"
    if kind == "evidence":
        return "supported"
    return "supported" if any(ref in sources for ref in entity["source_refs"]) else "maker_stated"


def _fact_row(
    *,
    brief_id: str,
    entity_id: str,
    entity_kind: str,
    statement: str,
    source_refs: list[str],
    evidence_refs: list[str] | None = None,
    asset_refs: list[str] | None = None,
    reference_roles: list[str] | None = None,
) -> dict[str, Any]:
    fact_id = _stable_id("f", brief_id, entity_id, statement)
    return {
        "fact_id": fact_id,
        "entity_id": entity_id,
        "entity_kind": entity_kind,
        "statement": statement,
        "source_refs": _unique(source_refs),
        "evidence_refs": _unique(evidence_refs or []),
        "asset_refs": _unique(asset_refs or []),
        "reference_roles": _unique(reference_roles or []),
    }


def _reference_roles(entity_kind: str, statement: str) -> list[str]:
    text = statement.lower()
    roles: list[str] = []
    if entity_kind == "product":
        roles.append("product_body")
    if any(word in text for word in ("도크", "스테이션", "dock")):
        roles.extend(("product_body", "dock"))
    if any(
        word in text
        for word in ("브러시", "걸레", "필터", "먼지통", "물통", "구성품", "부속")
    ):
        roles.append("accessory")
    if any(word in text for word in ("앱", "리모컨", "버튼", "제어 화면")):
        roles.append("control_interface")
    if entity_kind == "evidence" or any(
        word in text for word in ("시험", "인증", "비교표", "측정 결과")
    ):
        roles.append("provided_evidence")
    if entity_kind == "feature" and not roles:
        roles.append("product_body")
    return _unique(roles)


def project_brief_entities(brief: dict[str, Any]) -> dict[str, Any]:
    brief_id = brief["brief_id"]
    sources = [
        {
            "source_id": source["source_id"],
            "source_type": source["source_type"],
            "description": source["location"],
        }
        for source in brief["source"]["refs"]
    ]
    source_ids = {source["source_id"] for source in sources}
    evidence = [
        {
            "evidence_id": item["id"],
            "evidence_type": item["evidence_type"],
            "description": item["description"],
            "source_refs": item["source_refs"],
        }
        for item in brief["evidence"]
    ]
    assets = [
        {
            "asset_id": item["id"],
            "roles": roles,
            "description": item["description"],
            "source_refs": item["source_refs"],
        }
        for item in brief["assets"]
        if (roles := _asset_roles(item))
    ]
    product_asset_ids = [
        item["asset_id"]
        for item in assets
        if set(item["roles"]).intersection({"product_body", "dock"})
    ]
    accessory_asset_ids = [
        item["asset_id"] for item in assets if "accessory" in item["roles"]
    ]
    evidence_asset_ids = [
        item["asset_id"] for item in assets if "provided_evidence" in item["roles"]
    ]

    facts: list[dict[str, Any]] = []
    product = brief["product"]
    product_sources = _unique(
        ref
        for item in product["facts"]
        for ref in item["source_refs"]
    ) or [brief["source"]["refs"][0]["source_id"]]
    product_statement = (
        f"{product['name']}은(는) {product['product_type']}이며 "
        f"{product['summary']}"
    )
    facts.append(
        _fact_row(
            brief_id=brief_id,
            entity_id="product_identity",
            entity_kind="product",
            statement=product_statement,
            source_refs=product_sources,
            asset_refs=product_asset_ids,
            reference_roles=_reference_roles("product", product_statement),
        )
    )
    for item in product["facts"]:
        value = f"{item['value']}{item['unit'] or ''}"
        facts.append(
            _fact_row(
                brief_id=brief_id,
                entity_id=item["id"],
                entity_kind="claim",
                statement=f"{item['name']}은(는) {value}이다.",
                source_refs=item["source_refs"],
                asset_refs=product_asset_ids,
                reference_roles=_reference_roles(
                    "claim", f"{item['name']}은(는) {value}이다."
                ),
            )
        )
    for item in brief["problems"]:
        facts.append(
            _fact_row(
                brief_id=brief_id,
                entity_id=item["id"],
                entity_kind="problem",
                statement=item["description"],
                source_refs=item["source_refs"],
                reference_roles=_reference_roles("problem", item["description"]),
            )
        )
    for item in brief["features"]:
        facts.append(
            _fact_row(
                brief_id=brief_id,
                entity_id=item["id"],
                entity_kind="feature",
                statement=item["description"],
                source_refs=item["source_refs"],
                evidence_refs=item["evidence_ids"],
                asset_refs=[*product_asset_ids, *accessory_asset_ids],
                reference_roles=_reference_roles("feature", item["description"]),
            )
        )
    for item in brief["claims"]:
        facts.append(
            _fact_row(
                brief_id=brief_id,
                entity_id=item["id"],
                entity_kind="claim",
                statement=item["statement"],
                source_refs=item["source_refs"],
                evidence_refs=item["evidence_ids"],
                asset_refs=evidence_asset_ids if item["evidence_ids"] else [],
                reference_roles=_reference_roles("claim", item["statement"]),
            )
        )
    for item in brief["evidence"]:
        facts.append(
            _fact_row(
                brief_id=brief_id,
                entity_id=item["id"],
                entity_kind="evidence",
                statement=item["description"],
                source_refs=item["source_refs"],
                evidence_refs=[item["id"]],
                asset_refs=evidence_asset_ids,
                reference_roles=_reference_roles("evidence", item["description"]),
            )
        )

    if not facts:
        raise GenerationBoundaryError("Approved brief has no projectable product facts")
    referenced_sources = {
        ref for fact in facts for ref in fact["source_refs"]
    } | {ref for item in assets for ref in item["source_refs"]}
    if not referenced_sources.issubset(source_ids):
        raise GenerationBoundaryError("Entity projection contains an unknown source reference")
    return {
        "schema_version": "approved-entity-projection-v1",
        "brief_id": brief_id,
        "facts": facts,
        "sources": sources,
        "evidence": evidence,
        "assets": assets,
    }


def build_approved_generation_package(
    *,
    repository: DataRepository,
    input_id: str,
    thread_id: str,
    state: dict[str, Any],
    brief: dict[str, Any],
    local_asset_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    if state.get("workflow_stage") != "generation-ready":
        raise GenerationBoundaryError("Generation package requires generation-ready state")
    summary_version = state.get("summary_version")
    if summary_version is None or state.get("approved_summary_version") != summary_version:
        raise GenerationBoundaryError("Approved summary version does not match current summary")
    if any(value.get("status") == "conflicting" for value in state.get("facts", {}).values()):
        raise GenerationBoundaryError("Conflicting facts must be resolved before generation")

    repository.validate_story_brief(brief)
    brief_digest = canonical_digest(brief)
    entity_projection = project_brief_entities(brief)
    source_types = {
        source["source_id"]: source["source_type"]
        for source in entity_projection["sources"]
    }
    claims_by_id = {item["id"]: item for item in brief["claims"]}
    facts_revision = int(state.get("facts_revision", 0))
    fact_states = []
    for fact in entity_projection["facts"]:
        entity_kind = fact["entity_kind"]
        entity = claims_by_id.get(fact["entity_id"], {"source_refs": fact["source_refs"]})
        supported_sources = {
            ref
            for ref in fact["source_refs"]
            if source_types.get(ref) in {"document", "test-report"}
        }
        fact_states.append(
            {
                "fact_id": fact["fact_id"],
                "availability": "provided",
                "support_level": _support_level(entity_kind, entity, supported_sources),
                "collection_state": "resolved",
                "revision": facts_revision,
            }
        )
    worker_projection = {
        "approval_status": "approved",
        "summary_version": int(summary_version),
        "facts_revision": facts_revision,
        "collection_revision": int(state.get("collection_revision", 0)),
        "brief_digest": brief_digest,
        "fact_states": fact_states,
    }
    paths = {
        asset_id: str(path)
        for asset_id, path in (local_asset_paths or {}).items()
    }
    asset_ids = {asset["asset_id"] for asset in entity_projection["assets"]}
    if not set(paths).issubset(asset_ids):
        raise GenerationBoundaryError("Local path references an unknown approved asset")
    package = {
        "schema_version": "approved-generation-package-v1",
        "input_id": input_id,
        "thread_id": thread_id,
        "approval": {
            "status": "approved",
            "summary_version": int(summary_version),
            "facts_revision": facts_revision,
            "collection_revision": int(state.get("collection_revision", 0)),
        },
        "brief": brief,
        "brief_digest": brief_digest,
        "worker_projection": worker_projection,
        "worker_projection_digest": canonical_digest(worker_projection),
        "entity_projection": entity_projection,
        "entity_projection_digest": canonical_digest(entity_projection),
        "local_asset_paths": paths,
    }
    repository.validate_approved_generation_package(package)
    validate_approved_generation_package(repository=repository, package=package)
    return package


def validate_approved_generation_package(
    *, repository: DataRepository, package: dict[str, Any]
) -> None:
    repository.validate_approved_generation_package(package)
    if canonical_digest(package["brief"]) != package["brief_digest"]:
        raise GenerationBoundaryError("Brief digest mismatch")
    projection = package["worker_projection"]
    if projection["brief_digest"] != package["brief_digest"]:
        raise GenerationBoundaryError("Worker projection references a different brief")
    if canonical_digest(projection) != package["worker_projection_digest"]:
        raise GenerationBoundaryError("Worker projection digest mismatch")
    approval = package["approval"]
    for field in ("summary_version", "facts_revision", "collection_revision"):
        if approval[field] != projection[field]:
            raise GenerationBoundaryError(f"Approval and worker projection disagree on {field}")

    entity_projection = package["entity_projection"]
    if canonical_digest(entity_projection) != package["entity_projection_digest"]:
        raise GenerationBoundaryError("Entity projection digest mismatch")
    facts = {fact["fact_id"]: fact for fact in entity_projection["facts"]}
    states = {state["fact_id"]: state for state in projection["fact_states"]}
    if len(states) != len(projection["fact_states"]) or set(states) != set(facts):
        raise GenerationBoundaryError("Fact projection and authority states must match exactly")
    sources = {item["source_id"] for item in entity_projection["sources"]}
    evidence = {item["evidence_id"] for item in entity_projection["evidence"]}
    assets = {item["asset_id"] for item in entity_projection["assets"]}
    for fact in facts.values():
        if not set(fact["source_refs"]).issubset(sources):
            raise GenerationBoundaryError(f"{fact['fact_id']} has an unknown source reference")
        if not set(fact["evidence_refs"]).issubset(evidence):
            raise GenerationBoundaryError(f"{fact['fact_id']} has an unknown evidence reference")
        if not set(fact["asset_refs"]).issubset(assets):
            raise GenerationBoundaryError(f"{fact['fact_id']} has an unknown asset reference")
        if states[fact["fact_id"]]["revision"] != approval["facts_revision"]:
            raise GenerationBoundaryError(f"{fact['fact_id']} has a stale fact revision")
    if not set(package["local_asset_paths"]).issubset(assets):
        raise GenerationBoundaryError("Local path references an unknown approved asset")
