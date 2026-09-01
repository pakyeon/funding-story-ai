from __future__ import annotations

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _module(name: str, filename: str):
    path = Path(__file__).parents[1] / "evals" / filename
    spec = spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


holdout_validator = _module(
    "semantic_holdout_validator_test", "validate_semantic_normalization_holdout.py"
)
adversarial_validator = _module(
    "semantic_adversarial_validator_test",
    "validate_semantic_normalization_adversarial.py",
)
scorer = _module("semantic_extended_scorer_test", "score_semantic_normalization.py")


def _perfect_predictions(cases: list[dict]) -> dict:
    return {
        "schema_version": "semantic-normalization-predictions-v1",
        "predictions": [
            {"case_id": case["case_id"], "output": deepcopy(case["expected"])}
            for case in cases
        ],
    }


def test_sealed_holdout_contract_and_cross_split_separation() -> None:
    cases = holdout_validator.base._load(holdout_validator.HOLDOUT_PATH)
    summary = holdout_validator.validate_holdout(cases)

    assert summary["case_count"] == 64
    assert summary["exact_development_fact_overlap_count"] == 0
    assert summary["exact_development_proposition_overlap_count"] == 0
    assert not any(summary["development_id_overlap_count"].values())


def test_adversarial_contract_covers_injection_and_reference_locations() -> None:
    cases = adversarial_validator.base._load(adversarial_validator.CASES_PATH)
    summary = adversarial_validator.validate_adversarial(cases)

    assert summary["prompt_injection_count"] == 32
    assert summary["forged_reference_count"] == 32
    assert summary["rejected_semantic_invocation_count"] == 0
    assert set(summary["forged_target_counts"].values()) == {4}


def test_scorer_accepts_perfect_holdout_predictions() -> None:
    cases = scorer._gold_cases("holdout")

    result = scorer.score_predictions(_perfect_predictions(cases), cases)

    assert result["case_count"] == 64
    assert result["meets_gate"] is True


def test_scorer_accepts_perfect_adversarial_predictions() -> None:
    cases = scorer._gold_cases("adversarial")

    result = scorer.score_predictions(_perfect_predictions(cases), cases)

    assert result["case_count"] == 64
    assert result["accepted_case_count"] == 32
    assert result["rejected_semantic_invocation_count"] == 0
    assert result["meets_gate"] is True
