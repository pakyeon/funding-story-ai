from __future__ import annotations

import asyncio
import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from funding_story_ai.conversation import (
    ApprovalDecision,
    FactPatch,
    QuestionPlan,
    StorySummary,
    TurnUnderstanding,
    provided_facts,
)
from funding_story_ai.data_repository import DataRepository
from funding_story_ai.worker import (
    GroundedBriefBuilder,
    StoryMakerWorker,
    WorkerRequest,
    graph_state_to_semantic_state,
    validate_worker_request,
)


def _patches(**values: list[str]) -> list[FactPatch]:
    return [
        FactPatch(field=field, operation="replace", values=value)
        for field, value in values.items()
    ]


def _complete_understanding() -> TurnUnderstanding:
    return TurnUnderstanding(
        intent="provide_information",
        fact_patches=_patches(
            product_name=["OrbitClean V3"],
            product_type=["로봇청소기"],
            category=["테크·가전"],
            key_strengths=["얇은 본체", "자동 먼지 비움"],
            target_supporters=["가구 아래 청소가 필요한 사용자"],
            problem_context=["가구 아래 반복 청소"],
        ),
    )


class _Model:
    def understand_turn(self, *, message, messages, facts, image_path):
        if message["content"] == "완전한 제품 입력":
            return _complete_understanding()
        if message["content"] == "타깃 변경":
            return TurnUnderstanding(
                intent="revise_information",
                fact_patches=_patches(target_supporters=["반려동물 가구"]),
            )
        return TurnUnderstanding(intent="request_generation")

    def plan_questions(
        self,
        *,
        messages,
        facts,
        missing_required_fields,
        asked_topics,
        turn_understanding,
    ):
        return QuestionPlan(
            requested_fields=missing_required_fields[:2],
            question="부족한 제품 정보를 알려주세요.",
            rationale="필수 정보가 부족합니다.",
        )

    def build_summary(self, *, messages, facts):
        confirmed = provided_facts(facts)
        return StorySummary(
            headline="입력 요약",
            confirmed_facts=confirmed,
            unconfirmed_fields=[
                field for field, value in facts.items() if value["status"] != "provided"
            ],
            summary_text=" / ".join(
                value for values in confirmed.values() for value in values
            ),
            confirmation_question="이 내용으로 스토리를 생성할까요?",
        )

    def classify_approval(self, *, message, summary, messages):
        if message["content"] == "이 내용으로 생성해줘":
            return ApprovalDecision(decision="approve", reason="명시적 승인")
        if message["content"] == "네":
            return ApprovalDecision(decision="ambiguous", reason="대상이 불명확")
        return ApprovalDecision(decision="revise", reason="수정 입력")


class _BriefBuilder:
    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository
        self.calls = 0

    def build(self, request, semantic_state):
        self.calls += 1
        return self.repository.load_brief()


class _Tool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"run_id": "run-test", "status": "accepted"}


def _worker(*, checkpointer=None, connection=None):
    repository = DataRepository()
    tool = _Tool()
    builder = _BriefBuilder(repository)
    worker = StoryMakerWorker(
        repository=repository,
        conversation_model=_Model(),
        brief_builder=builder,
        generation_tool=tool,
        checkpointer=checkpointer or InMemorySaver(),
        checkpoint_connection=connection,
    )
    return worker, tool, builder


def _request(thread_id: str, message: str, index: int = 1) -> WorkerRequest:
    return WorkerRequest(
        thread_id=thread_id,
        input_id=thread_id,
        message=message,
        message_id=f"message-{index}",
    )


def test_worker_requires_summary_and_explicit_approval_before_mcp() -> None:
    worker, tool, builder = _worker()
    first = asyncio.run(worker.handle(_request("worker-flow", "완전한 제품 입력")))
    assert first.status == "awaiting-approval"
    assert first.summary_version == 1
    assert first.approved_summary_version is None
    assert tool.calls == []
    assert builder.calls == 0

    ambiguous = asyncio.run(worker.handle(_request("worker-flow", "네", 2)))
    assert ambiguous.status == "awaiting-approval"
    assert tool.calls == []

    submitted = asyncio.run(
        worker.handle(_request("worker-flow", "이 내용으로 생성해줘", 3))
    )
    assert submitted.status == "submitted"
    assert submitted.generation_start_trigger == "explicit-confirmation"
    assert submitted.approved_summary_version == submitted.summary_version
    assert builder.calls == 1
    assert len(tool.calls) == 1
    assert tool.calls[0]["idempotency_key"] == "worker-worker-flow-summary-1"


def test_worker_never_treats_initial_generation_request_as_approval() -> None:
    worker, tool, _ = _worker()
    outcome = asyncio.run(worker.handle(_request("worker-missing", "바로 생성해줘")))
    assert outcome.status == "awaiting-input"
    assert outcome.requested_fields
    assert tool.calls == []


def test_revision_after_summary_requires_new_approval() -> None:
    worker, tool, _ = _worker()
    first = asyncio.run(worker.handle(_request("worker-revise", "완전한 제품 입력")))
    assert first.summary_version == 1
    revised = asyncio.run(worker.handle(_request("worker-revise", "타깃 변경", 2)))
    assert revised.status == "awaiting-approval"
    assert revised.summary_version == 2
    assert revised.approved_summary_version is None
    assert revised.facts["target_supporters"]["values"] == ["반려동물 가구"]
    assert tool.calls == []


def test_sqlite_checkpointer_restores_state_across_worker_instances(tmp_path) -> None:
    path = tmp_path / "conversation.sqlite3"
    connection_one = sqlite3.connect(path, check_same_thread=False)
    saver_one = SqliteSaver(connection_one)
    saver_one.setup()
    first_worker, first_tool, _ = _worker(
        checkpointer=saver_one,
        connection=connection_one,
    )
    first = asyncio.run(first_worker.handle(_request("worker-persist", "완전한 제품 입력")))
    assert first.status == "awaiting-approval"
    assert first_tool.calls == []
    first_worker.close()

    connection_two = sqlite3.connect(path, check_same_thread=False)
    saver_two = SqliteSaver(connection_two)
    saver_two.setup()
    second_worker, second_tool, _ = _worker(
        checkpointer=saver_two,
        connection=connection_two,
    )
    submitted = asyncio.run(
        second_worker.handle(_request("worker-persist", "이 내용으로 생성해줘", 2))
    )
    assert submitted.status == "submitted"
    assert len(second_tool.calls) == 1
    second_worker.close()


def test_delete_thread_starts_a_fresh_session() -> None:
    worker, _, _ = _worker()
    first = asyncio.run(worker.handle(_request("worker-reset", "완전한 제품 입력")))
    assert first.facts["product_name"]["status"] == "provided"
    worker.delete_thread("worker-reset")
    restarted = asyncio.run(worker.handle(_request("worker-reset", "바로 생성해줘", 2)))
    assert restarted.facts["product_name"]["status"] == "unknown"
    assert restarted.summary_version == 0


def test_graph_state_adapter_preserves_current_values_without_image_discard() -> None:
    worker, _, _ = _worker()
    asyncio.run(worker.handle(_request("worker-adapter", "완전한 제품 입력")))
    asyncio.run(worker.handle(_request("worker-adapter", "타깃 변경", 2)))
    state = worker.get_state("worker-adapter")
    semantic = graph_state_to_semantic_state(input_id="worker-adapter", state=state)
    assert semantic["slots"]["target_supporters"]["values"] == ["반려동물 가구"]
    assert semantic["fact_conflict"]["status"] == "none"
    DataRepository().validate_intake_semantic_state(semantic)


def test_grounded_brief_preserves_optional_input_without_domain_expansion() -> None:
    repository = DataRepository()
    worker, _, _ = _worker()
    asyncio.run(worker.handle(_request("robot-grounded", "완전한 제품 입력")))
    state = deepcopy(worker.get_state("robot-grounded"))
    state["facts"]["key_strengths"]["values"] = [
        "최대 6,800Pa",
        "먼지 비움과 충전만 지원하는 도크",
    ]
    state["facts"]["rewards"] = {
        "status": "provided",
        "values": ["본품 얼리버드 1개"],
        "source_message_ids": ["message-1"],
        "updated_at_turn": 1,
    }
    state["facts"]["funding_end"] = {
        "status": "provided",
        "values": ["2026-09-30"],
        "source_message_ids": ["message-1"],
        "updated_at_turn": 1,
    }
    semantic = graph_state_to_semantic_state(input_id="robot-grounded", state=state)
    brief = GroundedBriefBuilder(repository=repository).build(
        _request("robot-grounded", "최신 사양만 사용해줘", 2),
        semantic,
    )
    serialized = json.dumps(brief, ensure_ascii=False)
    assert "6,800Pa" in serialized
    assert "본품 얼리버드 1개" in serialized
    assert brief["schedule_policy"]["funding_end"] == "2026-09-30"
    assert "카펫" not in serialized
    assert "문턱" not in serialized
    repository.validate_story_brief(brief)


def test_worker_enforces_public_text_and_image_input_limits(tmp_path) -> None:
    try:
        validate_worker_request(_request("too-long", "x" * 1_001))
    except ValueError as exc:
        assert "1,000" in str(exc)
    else:
        raise AssertionError("Expected the 1,000 character limit")

    unsupported = tmp_path / "product.gif"
    unsupported.write_bytes(b"gif")
    try:
        validate_worker_request(
            WorkerRequest(
                thread_id="unsupported-image",
                input_id="unsupported-image",
                message="product",
                image_path=Path(unsupported),
            )
        )
    except ValueError as exc:
        assert "JPG, PNG, or WEBP" in str(exc)
    else:
        raise AssertionError("Expected the image format limit")
