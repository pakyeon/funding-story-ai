from copy import deepcopy

import pytest

from funding_story_ai.data_repository import DataRepository
from funding_story_ai.media_projection import (
    GenerationBoundaryError,
    build_approved_generation_package,
    canonical_digest,
    validate_approved_generation_package,
)


def _state() -> dict:
    return {
        "workflow_stage": "generation-ready",
        "summary_version": 3,
        "approved_summary_version": 3,
        "facts_revision": 4,
        "collection_revision": 2,
        "facts": {
            "product_name": {"status": "provided", "values": ["클린포지 R1"]},
        },
    }


def _package(repository: DataRepository) -> dict:
    return build_approved_generation_package(
        repository=repository,
        input_id="robot-vacuum-one",
        thread_id="thread-one",
        state=_state(),
        brief=repository.load_brief(),
        local_asset_paths={
            "asset_product_hero": repository.root
            / "examples"
            / "robot-vacuum"
            / "product-reference.png"
        },
    )


def test_build_projection_preserves_approval_revisions_and_catalog_links() -> None:
    repository = DataRepository()

    package = _package(repository)

    assert package["approval"] == {
        "status": "approved",
        "summary_version": 3,
        "facts_revision": 4,
        "collection_revision": 2,
    }
    assert package["worker_projection"]["brief_digest"] == package["brief_digest"]
    assert len(package["entity_projection"]["facts"]) >= 20
    assert {
        role
        for asset in package["entity_projection"]["assets"]
        for role in asset["roles"]
    } >= {"product_body", "dock"}
    assert {
        state["revision"] for state in package["worker_projection"]["fact_states"]
    } == {4}


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("workflow_stage", "generation-ready"),
        ("approved_summary_version", "summary version"),
    ],
)
def test_projection_rejects_unapproved_state(field: str, message: str) -> None:
    repository = DataRepository()
    state = _state()
    state[field] = "collecting" if field == "workflow_stage" else 2

    with pytest.raises(GenerationBoundaryError, match=message):
        build_approved_generation_package(
            repository=repository,
            input_id="robot-vacuum-one",
            thread_id="thread-one",
            state=state,
            brief=repository.load_brief(),
        )


def test_projection_rejects_conflict_before_building_package() -> None:
    repository = DataRepository()
    state = _state()
    state["facts"]["product_name"]["status"] = "conflicting"

    with pytest.raises(GenerationBoundaryError, match="Conflicting"):
        build_approved_generation_package(
            repository=repository,
            input_id="robot-vacuum-one",
            thread_id="thread-one",
            state=state,
            brief=repository.load_brief(),
        )


def test_boundary_rejects_digest_revision_and_reference_tampering() -> None:
    repository = DataRepository()
    package = _package(repository)

    digest_tampered = deepcopy(package)
    digest_tampered["brief"]["product"]["name"] = "변조 제품"
    with pytest.raises(GenerationBoundaryError, match="Brief digest mismatch"):
        validate_approved_generation_package(
            repository=repository, package=digest_tampered
        )

    stale = deepcopy(package)
    stale["worker_projection"]["fact_states"][0]["revision"] = 3
    stale["worker_projection_digest"] = canonical_digest(stale["worker_projection"])
    with pytest.raises(GenerationBoundaryError, match="stale fact revision"):
        validate_approved_generation_package(repository=repository, package=stale)

    forged = deepcopy(package)
    forged["entity_projection"]["facts"][0]["asset_refs"] = ["asset_forged"]
    forged["entity_projection_digest"] = canonical_digest(forged["entity_projection"])
    with pytest.raises(GenerationBoundaryError, match="unknown asset reference"):
        validate_approved_generation_package(repository=repository, package=forged)


def test_projection_rejects_unknown_local_asset_path() -> None:
    repository = DataRepository()

    with pytest.raises(GenerationBoundaryError, match="unknown approved asset"):
        build_approved_generation_package(
            repository=repository,
            input_id="robot-vacuum-one",
            thread_id="thread-one",
            state=_state(),
            brief=repository.load_brief(),
            local_asset_paths={"asset_forged": repository.root / "forged.png"},
        )
