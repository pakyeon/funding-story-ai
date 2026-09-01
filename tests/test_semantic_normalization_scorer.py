from __future__ import annotations

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "evals" / "score_semantic_normalization.py"
SPEC = spec_from_file_location("semantic_normalization_scorer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
scorer = module_from_spec(SPEC)
SPEC.loader.exec_module(scorer)


@pytest.fixture
def normal_cases() -> list[dict]:
    return scorer._load(scorer.NORMAL_PATH)


@pytest.fixture
def perfect_predictions(normal_cases: list[dict]) -> dict:
    return {
        "schema_version": "semantic-normalization-predictions-v1",
        "predictions": [
            {"case_id": case["case_id"], "output": deepcopy(case["expected"])}
            for case in normal_cases
        ],
    }


def test_perfect_predictions_meet_gate(normal_cases: list[dict], perfect_predictions: dict) -> None:
    result = scorer.score_predictions(perfect_predictions, normal_cases)

    assert result["meets_gate"] is True
    assert result["decomposition"]["f1"] == 1.0
    assert result["capability_group_feature"]["macro_f1"] == 1.0
    assert result["capability_group_claim"]["macro_f1"] == 1.0
    assert result["accepted_reference_join_exact_rate"] == 1.0
    assert result["semantic_gate"] is True
    assert result["adapter_safety_gate"] is True


def test_claim_group_error_lowers_semantic_score(
    normal_cases: list[dict], perfect_predictions: dict
) -> None:
    case = normal_cases[0]
    claim_id = next(
        fact["fact_id"] for fact in case["model_view"]["facts"] if fact["entity_kind"] == "claim"
    )
    output = perfect_predictions["predictions"][0]["output"]
    proposition = next(
        item for item in output["atomic_propositions"] if item["fact_id"] == claim_id
    )
    proposition["capability_group"] = (
        "automation_return"
        if proposition["capability_group"] != "automation_return"
        else "cleaning_mechanism"
    )

    result = scorer.score_predictions(perfect_predictions, normal_cases)

    assert result["capability_group_claim"]["macro_f1"] < 1.0


def test_reference_join_error_fails_gate(
    normal_cases: list[dict], perfect_predictions: dict
) -> None:
    media_fact = next(
        item
        for prediction in perfect_predictions["predictions"]
        for item in prediction["output"]["media_facts"]
        if item["asset_refs"]
    )
    media_fact["asset_refs"] = []

    result = scorer.score_predictions(perfect_predictions, normal_cases)

    assert result["accepted_reference_join_exact_rate"] < 1.0
    assert result["adapter_safety_gate"] is False
    assert result["meets_gate"] is False


def test_duplicate_proposition_fails_gate(
    normal_cases: list[dict], perfect_predictions: dict
) -> None:
    proposition = deepcopy(
        perfect_predictions["predictions"][0]["output"]["atomic_propositions"][0]
    )
    proposition["proposition_id"] = "p_aaaaaaaaaaaa"
    perfect_predictions["predictions"][0]["output"]["atomic_propositions"].append(proposition)

    result = scorer.score_predictions(perfect_predictions, normal_cases)

    assert result["duplicate_proposition_count"] == 1
    assert result["meets_gate"] is False


def test_duplicate_proposition_id_fails_gate(
    normal_cases: list[dict], perfect_predictions: dict
) -> None:
    output = perfect_predictions["predictions"][0]["output"]
    duplicate_id = output["atomic_propositions"][0]["proposition_id"]
    output["atomic_propositions"][1]["proposition_id"] = duplicate_id

    result = scorer.score_predictions(perfect_predictions, normal_cases)

    assert result["duplicate_proposition_id_count"] == 1
    assert result["meets_gate"] is False


def test_invented_fact_id_fails_gate(normal_cases: list[dict], perfect_predictions: dict) -> None:
    perfect_predictions["predictions"][0]["output"]["atomic_propositions"].append(
        {
            "proposition_id": "p_bbbbbbbbbbbb",
            "fact_id": "f_aaaaaaaaaaaa",
            "text": "승인된 입력에 없는 기능을 추가한다.",
            "capability_group": "automation_return",
        }
    )

    result = scorer.score_predictions(perfect_predictions, normal_cases)

    assert result["invented_fact_id_count"] == 1
    assert result["meets_gate"] is False


def test_missing_prediction_case_is_rejected(
    normal_cases: list[dict], perfect_predictions: dict
) -> None:
    perfect_predictions["predictions"].pop()

    with pytest.raises(ValueError, match="coverage mismatch"):
        scorer.score_predictions(perfect_predictions, normal_cases)


def test_rejected_case_cannot_invoke_semantic_model() -> None:
    defensive_cases = scorer._load(scorer.DEFENSIVE_PATH)
    predictions = {
        "schema_version": "semantic-normalization-predictions-v1",
        "predictions": [
            {"case_id": case["case_id"], "output": deepcopy(case["expected"])}
            for case in defensive_cases
        ],
    }
    rejected = next(
        prediction
        for prediction in predictions["predictions"]
        if prediction["output"]["adapter"]["decision"] != "accepted"
    )
    case = next(
        case for case in defensive_cases if case["case_id"] == rejected["case_id"]
    )
    fact_id = case["model_view"]["facts"][0]["fact_id"]
    rejected["output"]["atomic_propositions"] = [
        {
            "proposition_id": "p_cccccccccccc",
            "fact_id": fact_id,
            "text": case["model_view"]["facts"][0]["statement"],
            "capability_group": "product_identity_outcome",
        }
    ]

    result = scorer.score_predictions(predictions, defensive_cases)

    assert result["rejected_semantic_invocation_count"] == 1
    assert result["meets_gate"] is False
