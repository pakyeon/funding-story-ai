from __future__ import annotations

from pathlib import Path

import pytest

from funding_story_ai.config import RuntimeSettings
from funding_story_ai.conversation import FactPatch, TurnUnderstanding
from funding_story_ai.field_evaluation import (
    RequestRateLimiter,
    _precompute_live_understandings,
    evaluate_field_cases,
    load_field_evaluation_dataset,
)

DATASET = (
    Path(__file__).parents[1]
    / "evals"
    / "datasets"
    / "story-worker-field-evaluation-v1.json"
)


class _Model:
    def __init__(self, outputs: dict[str, TurnUnderstanding]) -> None:
        self.outputs = outputs

    def understand_turn(self, *, message, messages, facts, image_path):
        return self.outputs[message["content"]]


class _OneErrorModel(_Model):
    def __init__(self, outputs: dict[str, TurnUnderstanding], failed_message: str) -> None:
        super().__init__(outputs)
        self.failed_message = failed_message

    def understand_turn(self, *, message, messages, facts, image_path):
        if message["content"] == self.failed_message:
            raise ValueError("truncated structured response")
        return super().understand_turn(
            message=message,
            messages=messages,
            facts=facts,
            image_path=image_path,
        )


def test_field_evaluation_scores_exact_outputs() -> None:
    dataset = load_field_evaluation_dataset(DATASET)
    outputs = {case.message: case.expected for case in dataset.cases}
    result = evaluate_field_cases(dataset, _Model(outputs))
    assert result["metrics"]["case_count"] == len(dataset.cases)
    assert result["metrics"]["field_selection_f1"] == 1.0
    assert result["metrics"]["case_state_exact_rate"] == 1.0
    assert result["metrics"]["latest_value_accuracy"] == 1.0
    assert result["metrics"]["stale_value_retention_rate"] == 0.0
    assert set(result["per_field_metrics"]) == {
        "product_name",
        "product_type",
        "category",
        "key_strengths",
        "target_supporters",
        "problem_context",
        "trust_elements",
        "maker_team_intro",
        "rewards",
        "funding_end",
        "shipping_start",
        "refund_policy",
        "as_policy",
        "funding_plan",
        "risk_response",
    }
    assert result["per_field_metrics"]["product_name"]["selection_recall"] == 1.0


def test_field_evaluation_separates_unsupported_and_missing_fields() -> None:
    dataset = load_field_evaluation_dataset(DATASET)
    outputs = {case.message: case.expected for case in dataset.cases}
    first = dataset.cases[0]
    outputs[first.message] = TurnUnderstanding(
        intent="provide_information",
        fact_patches=[
            FactPatch(field="trust_elements", operation="replace", values=["근거 없는 인증"]),
        ],
    )
    result = evaluate_field_cases(dataset, _Model(outputs))
    assert result["counts"]["field_false_positive"] == 1
    assert result["counts"]["field_false_negative"] == 1
    assert result["metrics"]["unsupported_field_fill_rate"] > 0
    assert result["metrics"]["case_state_exact_rate"] < 1


def test_field_evaluation_records_model_error_and_continues() -> None:
    dataset = load_field_evaluation_dataset(DATASET)
    outputs = {case.message: case.expected for case in dataset.cases}
    result = evaluate_field_cases(
        dataset,
        _OneErrorModel(outputs, dataset.cases[0].message),
    )
    assert result["counts"]["model_response_error"] == 1
    assert result["counts"]["model_response_success"] == len(dataset.cases) - 1
    assert result["metrics"]["model_response_success_rate"] == pytest.approx(
        (len(dataset.cases) - 1) / len(dataset.cases)
    )
    assert result["cases"][0]["model_error"]["type"] == "ValueError"


def test_request_rate_limiter_spaces_requests(monkeypatch) -> None:
    clock = [0.0]
    sleeps: list[float] = []

    monkeypatch.setattr(
        "funding_story_ai.field_evaluation.time.monotonic",
        lambda: clock[0],
    )

    def sleep(delay: float) -> None:
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr("funding_story_ai.field_evaluation.time.sleep", sleep)
    limiter = RequestRateLimiter(60)
    limiter.wait()
    limiter.wait()
    limiter.wait()
    assert sleeps == [1.0, 1.0]


def test_live_precompute_checkpoint_resumes_only_successful_cases(
    monkeypatch, tmp_path
) -> None:
    dataset = load_field_evaluation_dataset(DATASET)

    class FakeAdapter:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class FakeModel:
        calls = 0

        def __init__(self, adapter) -> None:
            self.last_model = "fake-model"
            self.last_call_count = 1

        def understand_turn(self, *, message, messages, facts, image_path):
            type(self).calls += 1
            return TurnUnderstanding(intent="provide_information")

    monkeypatch.setattr("funding_story_ai.field_evaluation.GeminiAdapter", FakeAdapter)
    monkeypatch.setattr(
        "funding_story_ai.field_evaluation.GeminiConversationModel", FakeModel
    )
    monkeypatch.setattr(
        "funding_story_ai.field_evaluation.RequestRateLimiter.wait", lambda self: None
    )
    checkpoint = tmp_path / "field-evaluation.jsonl"
    settings = RuntimeSettings(project_id="test-project")

    _precompute_live_understandings(
        dataset,
        settings,
        workers=3,
        checkpoint_path=checkpoint,
        resume=False,
        requests_per_minute=60,
    )
    assert FakeModel.calls == len(dataset.cases)
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == len(dataset.cases) + 1

    FakeModel.calls = 0
    _precompute_live_understandings(
        dataset,
        settings,
        workers=3,
        checkpoint_path=checkpoint,
        resume=True,
        requests_per_minute=60,
    )
    assert FakeModel.calls == 0

    with pytest.raises(ValueError, match="does not match"):
        _precompute_live_understandings(
            dataset,
            RuntimeSettings(project_id="test-project", thinking_level="HIGH"),
            workers=3,
            checkpoint_path=checkpoint,
            resume=True,
            requests_per_minute=60,
        )
