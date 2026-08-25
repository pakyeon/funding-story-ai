import pytest

from funding_story_ai.run_store import IdempotencyConflict, LocalRunStore


def test_run_store_replays_running_and_completed_result(tmp_path) -> None:
    store = LocalRunStore(tmp_path)
    payload = {"brief": {"brief_id": "one"}}
    record, created = store.begin(
        caller_id="worker",
        idempotency_key="stable-key",
        request_payload=payload,
    )
    assert created is True
    running, created = store.begin(
        caller_id="worker",
        idempotency_key="stable-key",
        request_payload=payload,
    )
    assert created is False
    assert running["status"] == "running"

    completed = store.complete(record["run_id"], {"template_id": "t01"})
    replay, created = store.begin(
        caller_id="worker",
        idempotency_key="stable-key",
        request_payload=payload,
    )
    assert created is False
    assert replay == completed
    with pytest.raises(IdempotencyConflict):
        store.begin(
            caller_id="worker",
            idempotency_key="stable-key",
            request_payload={"brief": {"brief_id": "different"}},
        )
