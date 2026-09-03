from types import SimpleNamespace

import pytest

from funding_story_ai.adapter import GeminiAdapter
from funding_story_ai.config import RuntimeSettings


class AccessError(RuntimeError):
    code = 503


class MissingModelError(RuntimeError):
    code = 404


class TransportError(RuntimeError):
    pass


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.models = []

    def generate_content(self, *, model, contents, config):
        self.models.append(model)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def response():
    return SimpleNamespace(
        text='{"status":"ok"}',
        parsed={"status": "ok"},
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(value="STOP"))],
    )


def adapter(
    outcomes,
    attempts=5,
    *,
    allow_fallback=True,
    before_call=None,
    sleep=None,
):
    settings = RuntimeSettings(
        project_id="test-project",
        primary_access_attempts=attempts,
    )
    models = FakeModels(outcomes)
    return GeminiAdapter(
        settings,
        client=SimpleNamespace(models=models),
        sleep=sleep or (lambda _: None),
        jitter=lambda _start, _end: 0.0,
        before_call=before_call,
        allow_fallback=allow_fallback,
    ), models


def test_success_uses_primary() -> None:
    target, models = adapter([response()])
    result = target.generate_json(prompt="test", response_schema={"type": "object"})
    assert result.model == "gemini-3.8-flash"
    assert models.models == ["gemini-3.8-flash"]


def test_fallback_after_five_access_errors() -> None:
    target, models = adapter([AccessError()] * 5 + [response()])
    result = target.generate_json(prompt="test", response_schema={"type": "object"})
    assert result.model == "gemini-3.6-flash"
    assert models.models == ["gemini-3.8-flash"] * 5 + ["gemini-3.6-flash"]


def test_non_access_error_does_not_fallback() -> None:
    target, models = adapter([ValueError("invalid response")])
    with pytest.raises(ValueError, match="invalid response"):
        target.generate_json(prompt="test", response_schema={"type": "object"})
    assert models.models == ["gemini-3.8-flash"]


def test_evaluation_mode_does_not_fallback() -> None:
    target, models = adapter([AccessError()] * 5, allow_fallback=False)
    with pytest.raises(AccessError):
        target.generate_json(prompt="test", response_schema={"type": "object"})
    assert models.models == ["gemini-3.8-flash"] * 5


def test_before_call_runs_for_every_actual_model_request() -> None:
    calls: list[str] = []
    target, models = adapter(
        [AccessError(), response()],
        attempts=2,
        before_call=lambda: calls.append("called"),
    )
    target.generate_json(prompt="test", response_schema={"type": "object"})
    assert calls == ["called", "called"]
    assert models.models == ["gemini-3.8-flash", "gemini-3.8-flash"]


def test_non_retryable_model_access_error_falls_back_without_repeating() -> None:
    target, models = adapter([MissingModelError(), response()])
    result = target.generate_json(prompt="test", response_schema={"type": "object"})
    assert result.model == "gemini-3.6-flash"
    assert models.models == ["gemini-3.8-flash", "gemini-3.6-flash"]


def test_retry_after_header_controls_retry_delay() -> None:
    error = AccessError()
    error.response = SimpleNamespace(headers={"Retry-After": "3"})
    delays: list[float] = []
    target, _ = adapter(
        [error, response()],
        attempts=2,
        sleep=delays.append,
    )
    target.generate_json(prompt="test", response_schema={"type": "object"})
    assert delays == [3.0]


def test_transport_error_is_retried() -> None:
    target, models = adapter([TransportError("DNS lookup failed"), response()], attempts=2)
    result = target.generate_json(prompt="test", response_schema={"type": "object"})
    assert result.model == "gemini-3.8-flash"
    assert models.models == ["gemini-3.8-flash", "gemini-3.8-flash"]


def test_wrapped_504_message_is_retried() -> None:
    target, models = adapter(
        [RuntimeError("ServerError: 504 DEADLINE_EXCEEDED"), response()], attempts=2
    )
    result = target.generate_json(prompt="test", response_schema={"type": "object"})
    assert result.model == "gemini-3.8-flash"
    assert models.models == ["gemini-3.8-flash", "gemini-3.8-flash"]
