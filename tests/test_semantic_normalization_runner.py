from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).parents[1] / "evals" / "run_semantic_normalization.py"
SPEC = spec_from_file_location("semantic_normalization_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class _FakeAdapter:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.calls = 0

    def generate_json(self, *, prompt: str, response_schema: dict) -> SimpleNamespace:
        self.calls += 1
        assert "expected" not in prompt
        assert response_schema == runner.SEMANTIC_RESPONSE_SCHEMA
        return SimpleNamespace(data=self.data)


class _SequenceAdapter:
    def __init__(self, values: list[dict]) -> None:
        self.values = iter(values)
        self.calls = 0

    def generate_json(self, *, prompt: str, response_schema: dict) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(data=next(self.values))


def _model_response_from_gold(case: dict) -> dict:
    propositions_by_fact: dict[str, list[dict]] = {}
    for proposition in case["expected"]["atomic_propositions"]:
        propositions_by_fact.setdefault(proposition["fact_id"], []).append(
            {
                "text": proposition["text"],
                "capability_group": proposition["capability_group"],
            }
        )
    return {
        "facts": [
            {
                "fact_id": fact["fact_id"],
                "propositions": propositions_by_fact.get(fact["fact_id"], []),
            }
            for fact in case["model_view"]["facts"]
            if fact["entity_kind"] != "distractor"
        ]
    }


def test_runner_joins_model_semantics_to_authoritative_adapter_state() -> None:
    case = runner._load(runner.DEV_ROOT / "normal-cases.json")[0]
    adapter = _FakeAdapter(_model_response_from_gold(case))

    output, error = runner.run_case(case, adapter)

    assert error is None
    assert adapter.calls == 1
    assert output["adapter"]["decision"] == "accepted"
    assert output["adapter"]["failure_code"] is None
    assert {item["fact_id"] for item in output["media_facts"]} == {
        item["fact_id"] for item in output["atomic_propositions"]
    }


def test_rejected_boundary_never_calls_semantic_model() -> None:
    case = next(
        case
        for case in runner._load(runner.DEV_ROOT / "defensive-cases.json")
        if case["expected"]["adapter"]["decision"] != "accepted"
    )
    adapter = _FakeAdapter({"facts": []})

    output, error = runner.run_case(case, adapter)

    assert error is None
    assert adapter.calls == 0
    assert output["adapter"]["decision"].startswith("reject_")
    assert output["atomic_propositions"] == []
    assert output["media_facts"] == []


def test_invalid_model_coverage_is_recorded_as_failed_prediction() -> None:
    case = runner._load(runner.DEV_ROOT / "normal-cases.json")[0]
    adapter = _FakeAdapter({"facts": []})

    output, error = runner.run_case(case, adapter)

    assert error is not None
    assert "cover every fact exactly once" in error
    assert output["adapter"]["decision"] == "accepted"
    assert output["atomic_propositions"] == []


def test_runner_accepts_the_clean_fact_clause_inside_instruction_like_data() -> None:
    case = next(
        case
        for case in runner._load(runner.ADVERSARIAL_ROOT / "adversarial-cases.json")
        if "prompt_injection" in case["tags"]
    )
    adapter = _FakeAdapter(_model_response_from_gold(case))

    output, error = runner.run_case(case, adapter)

    assert error is None
    assert adapter.calls == 1
    assert output["adapter"]["decision"] == "accepted"
    assert all("SYSTEM" not in item["text"] for item in output["atomic_propositions"])


def test_live_runner_rejects_zero_workers() -> None:
    with pytest.raises(ValueError, match="workers must be at least 1"):
        runner.run_live_cases([], runner.RuntimeSettings(project_id="test"), workers=0)


def test_runner_repairs_one_invalid_structured_response() -> None:
    case = runner._load(runner.DEV_ROOT / "normal-cases.json")[0]
    adapter = _SequenceAdapter([{"facts": []}, _model_response_from_gold(case)])

    output, error, diagnostics = runner.run_case_detailed(case, adapter)

    assert error is None
    assert output["atomic_propositions"]
    assert adapter.calls == 2
    assert diagnostics == {"semantic_model_call_count": 2, "repair_used": True}
