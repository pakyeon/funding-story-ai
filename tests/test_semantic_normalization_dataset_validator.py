from __future__ import annotations

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import jsonschema
import pytest

MODULE_PATH = Path(__file__).parents[1] / "evals" / "validate_semantic_normalization_dataset.py"
SPEC = spec_from_file_location("semantic_dataset_validator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
semantic_validator = module_from_spec(SPEC)
SPEC.loader.exec_module(semantic_validator)


@pytest.fixture
def validator() -> jsonschema.Draft202012Validator:
    schema = semantic_validator._load(semantic_validator.SCHEMA_PATH)
    return jsonschema.Draft202012Validator(schema)


@pytest.fixture
def normal_case() -> dict:
    return deepcopy(semantic_validator._load(semantic_validator.NORMAL_PATH)[0])


def test_valid_case_passes(normal_case: dict, validator: jsonschema.Draft202012Validator) -> None:
    semantic_validator._validate_case(normal_case, validator)


@pytest.mark.parametrize("mutation", ["digest", "revision", "short_proposition"])
def test_integrity_mutations_are_rejected(
    normal_case: dict,
    validator: jsonschema.Draft202012Validator,
    mutation: str,
) -> None:
    if mutation == "digest":
        normal_case["input"]["brief_digest"] = "sha256:" + "0" * 64
    elif mutation == "revision":
        normal_case["input"]["worker_projection"]["fact_states"][0]["revision"] = 999
        normal_case["input"]["worker_projection_digest"] = semantic_validator._digest(
            normal_case["input"]["worker_projection"]
        )
    else:
        normal_case["expected"]["atomic_propositions"][0]["text"] = "짧은 문구"

    with pytest.raises(AssertionError):
        semantic_validator._validate_case(normal_case, validator)


def test_media_reference_mutation_is_rejected(
    normal_case: dict, validator: jsonschema.Draft202012Validator
) -> None:
    media_fact = next(item for item in normal_case["expected"]["media_facts"] if item["asset_refs"])
    media_fact["asset_refs"] = []

    with pytest.raises(AssertionError):
        semantic_validator._validate_case(normal_case, validator)


def test_duplicate_media_fact_is_rejected(
    normal_case: dict, validator: jsonschema.Draft202012Validator
) -> None:
    normal_case["expected"]["media_facts"].append(
        deepcopy(normal_case["expected"]["media_facts"][0])
    )

    with pytest.raises(AssertionError):
        semantic_validator._validate_case(normal_case, validator)
