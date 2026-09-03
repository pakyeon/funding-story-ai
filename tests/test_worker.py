from __future__ import annotations

import asyncio
import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from funding_story_ai.adapter import GenerationResult
from funding_story_ai.conversation import (
    FACT_FIELDS,
    OPTIONAL_FACT_FIELDS,
    REQUIRED_FACT_FIELDS,
    ApprovalDecision,
    CollectionDirective,
    FactPatch,
    OptionalCollection,
    QuestionPlan,
    StorySummary,
    TurnUnderstanding,
    explicitly_absent_fields,
    initial_facts,
    initial_optional_collection,
    provided_facts,
    skipped_optional_fields,
)
from funding_story_ai.data_repository import DataRepository
from funding_story_ai.worker import (
    GeminiConversationModel,
    GroundedBriefBuilder,
    StoryGenerationDispatcher,
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


def _required_understanding() -> TurnUnderstanding:
    return TurnUnderstanding(
        intent="provide_information",
        fact_patches=_patches(
            product_name=["OrbitClean V3"],
            product_type=["로봇청소기"],
            category=["테크·가전"],
            key_strengths=["얇은 본체", "자동 먼지 비움"],
            target_supporters=["가구 아래 청소가 필요한 사용자"],
        ),
    )


def _all_fields_understanding() -> TurnUnderstanding:
    return TurnUnderstanding(
        intent="provide_information",
        fact_patches=_patches(**{field: [f"{field} 값"] for field in FACT_FIELDS}),
    )


class _Model:
    def understand_turn(self, *, message, messages, facts, image_path):
        content = message["content"]
        if content == "필수 제품 입력":
            return _required_understanding()
        if content == "15개 전체 입력":
            return _all_fields_understanding()
        if content == "선택 정보 전체 생략":
            return TurnUnderstanding(
                intent="request_generation",
                collection_directive=CollectionDirective(action="skip_all_optional"),
            )
        if content == "정책과 프로젝트 설명부터":
            return TurnUnderstanding(
                intent="provide_information",
                collection_directive=CollectionDirective(
                    action="select_groups", groups=["policy", "project_explanation"]
                ),
            )
        if content == "타깃 변경":
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
        purpose,
        candidate_fields,
        requested_group,
        requested_detail,
        question_history,
        turn_understanding,
    ):
        requested = candidate_fields[:3]
        return QuestionPlan(
            purpose=purpose,
            requested_fields=requested,
            requested_group=requested_group,
            requested_detail=requested_detail,
            question=(
                requested_detail
                if purpose in {"clarify", "confirm-skip"}
                else "부족한 제품 정보를 알려주세요."
            ),
            rationale="현재 수집 단계에 필요한 정보입니다.",
        )

    def repair_question_plan(
        self,
        *,
        invalid_plan,
        validation_error,
        messages,
        facts,
        purpose,
        candidate_fields,
        requested_group,
        requested_detail,
    ):
        return QuestionPlan(
            purpose=purpose,
            requested_fields=candidate_fields[:3],
            requested_group=requested_group,
            requested_detail=requested_detail,
            question=requested_detail,
            rationale="질문 계획을 교정합니다.",
        )

    def build_summary(self, *, messages, facts, optional_collection):
        confirmed = provided_facts(facts)
        return StorySummary(
            headline="입력 요약",
            confirmed_facts=confirmed,
            explicitly_absent_fields=explicitly_absent_fields(facts),
            skipped_fields=skipped_optional_fields(optional_collection),
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


class _ToggleFailureModel(_Model):
    def __init__(self) -> None:
        self.fail = False

    def understand_turn(self, *, message, messages, facts, image_path):
        if self.fail:
            raise TimeoutError("temporary Gemini timeout")
        return super().understand_turn(
            message=message,
            messages=messages,
            facts=facts,
            image_path=image_path,
        )


class _LongRevisionModel(_Model):
    def understand_turn(self, *, message, messages, facts, image_path):
        if message["content"].startswith("타깃 수정 "):
            return TurnUnderstanding(
                intent="revise_information",
                fact_patches=_patches(target_supporters=[message["content"]]),
            )
        return super().understand_turn(
            message=message,
            messages=messages,
            facts=facts,
            image_path=image_path,
        )


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


class _StructuredAdapter:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = iter(outputs)
        self.calls = 0

    def generate_multimodal_json(self, *, prompt, images, response_schema):
        self.calls += 1
        return GenerationResult(model="test-model", data=next(self.outputs))

    def generate_json(self, *, prompt, response_schema):
        self.calls += 1
        return GenerationResult(model="test-model", data=next(self.outputs))


def _worker(*, checkpointer=None, connection=None) -> StoryMakerWorker:
    return StoryMakerWorker(
        conversation_model=_Model(),
        checkpointer=checkpointer or InMemorySaver(),
        checkpoint_connection=connection,
    )


def _request(thread_id: str, message: str, index: int = 1) -> WorkerRequest:
    return WorkerRequest(
        thread_id=thread_id,
        input_id=thread_id,
        message=message,
        message_id=f"message-{index}",
    )


def _ready_worker_state(worker: StoryMakerWorker, thread_id: str) -> dict[str, Any]:
    asyncio.run(worker.handle(_request(thread_id, "필수 제품 입력", 1)))
    asyncio.run(worker.handle(_request(thread_id, "선택 정보 전체 생략", 2)))
    outcome = asyncio.run(worker.handle(_request(thread_id, "이 내용으로 생성해줘", 3)))
    assert outcome.status == "generation-ready"
    return worker.get_state(thread_id)


def test_worker_stops_at_generation_ready_without_mcp_dispatch() -> None:
    worker = _worker()
    offer = asyncio.run(worker.handle(_request("worker-flow", "필수 제품 입력")))
    assert offer.status == "awaiting-input"
    assert offer.collection_phase == "optional-offer"
    assert len(offer.remaining_optional_fields) == 10

    summary = asyncio.run(
        worker.handle(_request("worker-flow", "선택 정보 전체 생략", 2))
    )
    assert summary.status == "awaiting-approval"
    assert summary.summary_version == 1

    ambiguous = asyncio.run(worker.handle(_request("worker-flow", "네", 3)))
    assert ambiguous.status == "awaiting-approval"

    ready = asyncio.run(
        worker.handle(_request("worker-flow", "이 내용으로 생성해줘", 4))
    )
    assert ready.status == "generation-ready"
    assert ready.stage == "generation-ready"
    assert ready.generation_start_trigger == "explicit-confirmation"
    assert ready.approved_summary_version == ready.summary_version
    assert ready.to_dict()["optional_collection"]["offered"] is True


def test_mcp_dispatch_is_a_separate_explicit_operation() -> None:
    repository = DataRepository()
    worker = _worker()
    state = _ready_worker_state(worker, "worker-dispatch")
    builder = _BriefBuilder(repository)
    tool = _Tool()
    dispatcher = StoryGenerationDispatcher(
        repository=repository,
        brief_builder=builder,
        generation_tool=tool,
    )
    result = asyncio.run(
        dispatcher.submit(
            request=_request("worker-dispatch", "이 내용으로 생성해줘", 3),
            state=state,
        )
    )
    assert result["status"] == "accepted"
    assert builder.calls == 1
    assert len(tool.calls) == 1
    assert tool.calls[0]["idempotency_key"] == "worker-worker-dispatch-summary-1"
    package = tool.calls[0]["generation_package"]
    assert package["approval"]["status"] == "approved"
    assert package["approval"]["summary_version"] == state["summary_version"]
    assert package["worker_projection"]["facts_revision"] == state["facts_revision"]


def test_mcp_dispatch_rejects_state_before_generation_ready() -> None:
    repository = DataRepository()
    worker = _worker()
    asyncio.run(worker.handle(_request("worker-not-ready", "필수 제품 입력")))
    dispatcher = StoryGenerationDispatcher(
        repository=repository,
        brief_builder=_BriefBuilder(repository),
        generation_tool=_Tool(),
    )
    with pytest.raises(ValueError, match="generation-ready"):
        asyncio.run(
            dispatcher.submit(
                request=_request("worker-not-ready", "제출", 2),
                state=worker.get_state("worker-not-ready"),
            )
        )


def test_initial_generation_request_never_skips_required_or_optional_information() -> None:
    worker = _worker()
    outcome = asyncio.run(worker.handle(_request("worker-missing", "바로 생성해줘")))
    assert outcome.status == "awaiting-input"
    assert 1 <= len(outcome.requested_fields) <= 3
    assert outcome.collection_phase == "required"


def test_required_question_stops_after_one_no_progress_rephrase() -> None:
    worker = _worker()
    first = asyncio.run(worker.handle(_request("worker-no-progress", "바로 생성해줘", 1)))
    second = asyncio.run(worker.handle(_request("worker-no-progress", "잘 모르겠어요", 2)))
    stopped = asyncio.run(worker.handle(_request("worker-no-progress", "아직 모르겠어요", 3)))

    assert first.requested_fields
    assert second.requested_fields == first.requested_fields
    assert stopped.requested_fields == ()
    assert "같은 질문은 여기서 중단" in stopped.reply
    assert stopped.status == "awaiting-input"
    assert stopped.stage == "collecting"
    assert stopped.generation_start_trigger is None


def test_fact_progress_allows_a_previously_requested_field_to_be_asked_again() -> None:
    worker = _worker()
    asyncio.run(worker.handle(_request("worker-progress-reset", "바로 생성해줘", 1)))
    asyncio.run(worker.handle(_request("worker-progress-reset", "잘 모르겠어요", 2)))

    progressed = asyncio.run(
        worker.handle(_request("worker-progress-reset", "필수 제품 입력", 3))
    )

    assert progressed.collection_phase == "optional-offer"
    assert "필수 정보는 모두 확인했습니다" in progressed.reply


def test_optional_choice_offer_does_not_repeat_without_progress() -> None:
    worker = _worker()
    asyncio.run(worker.handle(_request("worker-optional-no-progress", "필수 제품 입력", 1)))
    first_choice = asyncio.run(
        worker.handle(_request("worker-optional-no-progress", "아직 결정 못했어요", 2))
    )
    stopped = asyncio.run(
        worker.handle(_request("worker-optional-no-progress", "조금 더 생각할게요", 3))
    )

    assert "남은 선택 정보는 다음과 같습니다" in first_choice.reply
    assert "같은 안내는 여기서 중단" in stopped.reply
    assert stopped.requested_fields == ()
    assert stopped.status == "awaiting-input"


def test_temporary_model_failure_preserves_session_and_never_approves() -> None:
    model = _ToggleFailureModel()
    worker = StoryMakerWorker(conversation_model=model, checkpointer=InMemorySaver())
    asyncio.run(worker.handle(_request("worker-temporary-failure", "필수 제품 입력", 1)))
    before = worker.get_state("worker-temporary-failure")
    model.fail = True

    failed = asyncio.run(
        worker.handle(_request("worker-temporary-failure", "선택 정보 전체 생략", 2))
    )
    after = worker.get_state("worker-temporary-failure")

    assert failed.status == "awaiting-input"
    assert failed.generation_start_trigger is None
    assert failed.temporary_error is True
    assert "현재 입력 상태는 유지했습니다" in failed.reply
    assert after["facts"] == before["facts"]
    assert after["facts_revision"] == before["facts_revision"]
    assert after["collection_revision"] == before["collection_revision"]
    assert after.get("summary_version", 0) == before.get("summary_version", 0)
    assert after["workflow_stage"] != "generation-ready"


def test_understanding_repairs_invalid_structured_output_once() -> None:
    adapter = _StructuredAdapter(
        [
            {"intent": "provide_information", "requires_clarification": True},
            {"intent": "provide_information", "fact_patches": []},
        ]
    )
    model = GeminiConversationModel(adapter)  # type: ignore[arg-type]

    result = model.understand_turn(
        message={"id": "message", "role": "user", "content": "제품 정보"},
        messages=[{"id": "message", "role": "user", "content": "제품 정보"}],
        facts={},
        image_path=None,
    )

    assert result.intent == "provide_information"
    assert adapter.calls == 2
    assert model.last_call_count == 2


def test_understanding_repairs_missing_explicit_required_facts_once() -> None:
    message = (
        "제품명은 펫세이프 6이고 반려동물 카테고리의 반려동물 급식기야. "
        "핵심 강점은 정량 급여, 주요 서포터는 직장인 반려인이야."
    )
    patches = [
        {
            "field": field,
            "operation": "replace",
            "values": [value],
        }
        for field, value in {
            "product_name": "펫세이프 6",
            "product_type": "반려동물 급식기",
            "category": "반려동물",
            "key_strengths": "정량 급여",
            "target_supporters": "직장인 반려인",
        }.items()
    ]
    adapter = _StructuredAdapter(
        [
            {"intent": "provide_information", "fact_patches": []},
            {"intent": "provide_information", "fact_patches": patches},
        ]
    )
    model = GeminiConversationModel(adapter)  # type: ignore[arg-type]

    result = model.understand_turn(
        message={"id": "message", "role": "user", "content": message},
        messages=[{"id": "message", "role": "user", "content": message}],
        facts=initial_facts(),
        image_path=None,
    )

    assert {patch.field for patch in result.fact_patches} == set(REQUIRED_FACT_FIELDS)
    assert adapter.calls == 2
    assert model.last_call_count == 2


def test_summary_repairs_a_grounding_violation_once() -> None:
    facts = initial_facts()
    values = {
        "product_name": "데일리결 5",
        "product_type": "기초 화장품",
        "category": "뷰티",
        "key_strengths": "무향 보습",
        "target_supporters": "민감성 피부 사용자",
    }
    for field, value in values.items():
        facts[field] = {
            "status": "provided",
            "values": [value],
            "source_message_ids": ["message-1"],
            "updated_at_turn": 1,
        }
    collection = OptionalCollection.model_validate(initial_optional_collection(facts))
    collection = collection.model_copy(
        update={
            "offered": True,
            "field_states": {field: "skipped" for field in OPTIONAL_FACT_FIELDS},
        }
    )
    confirmed = provided_facts(facts)
    common = {
        "headline": "스토리 생성 정보 요약",
        "confirmed_facts": confirmed,
        "explicitly_absent_fields": [],
        "skipped_fields": list(OPTIONAL_FACT_FIELDS),
        "confirmation_question": "이 내용으로 스토리를 생성할까요?",
    }
    adapter = _StructuredAdapter(
        [
            {**common, "summary_text": "선택 정보 11개를 생략했습니다."},
            {**common, "summary_text": "선택 정보는 이번 생성에서 생략했습니다."},
        ]
    )
    model = GeminiConversationModel(adapter)  # type: ignore[arg-type]

    result = model.build_summary(
        messages=[],
        facts=facts,
        optional_collection=collection.model_dump(mode="json"),
    )

    assert "11" not in result.summary_text
    assert adapter.calls == 2
    assert model.last_call_count == 2


def test_repeated_structured_output_failure_returns_safe_worker_response() -> None:
    invalid = {"intent": "provide_information", "requires_clarification": True}
    adapter = _StructuredAdapter([invalid, invalid])
    worker = StoryMakerWorker(
        conversation_model=GeminiConversationModel(adapter),  # type: ignore[arg-type]
        checkpointer=InMemorySaver(),
    )

    outcome = asyncio.run(worker.handle(_request("worker-structured-failure", "제품 정보", 1)))

    assert adapter.calls == 2
    assert outcome.status == "awaiting-input"
    assert outcome.generation_start_trigger is None
    assert outcome.temporary_error is True
    assert "현재 입력 상태는 유지했습니다" in outcome.reply


def test_long_conversation_with_continuous_progress_has_no_arbitrary_turn_cap() -> None:
    worker = StoryMakerWorker(
        conversation_model=_LongRevisionModel(), checkpointer=InMemorySaver()
    )
    first = asyncio.run(worker.handle(_request("worker-long-progress", "15개 전체 입력", 1)))
    assert first.status == "awaiting-approval"

    outcome = first
    for index in range(2, 22):
        outcome = asyncio.run(
            worker.handle(
                _request("worker-long-progress", f"타깃 수정 {index}", index)
            )
        )

    assert outcome.status == "awaiting-approval"
    assert outcome.stage == "awaiting-approval"
    assert outcome.summary_version == 21
    assert outcome.generation_start_trigger is None


def test_sqlite_checkpointer_restores_no_progress_question_history(tmp_path) -> None:
    path = tmp_path / "no-progress.sqlite3"
    connection_one = sqlite3.connect(path, check_same_thread=False)
    saver_one = SqliteSaver(connection_one)
    saver_one.setup()
    first_worker = _worker(checkpointer=saver_one, connection=connection_one)
    asyncio.run(first_worker.handle(_request("worker-no-progress-persist", "바로 생성해줘", 1)))
    asyncio.run(first_worker.handle(_request("worker-no-progress-persist", "모르겠어요", 2)))
    first_worker.close()

    connection_two = sqlite3.connect(path, check_same_thread=False)
    saver_two = SqliteSaver(connection_two)
    saver_two.setup()
    second_worker = _worker(checkpointer=saver_two, connection=connection_two)
    stopped = asyncio.run(
        second_worker.handle(_request("worker-no-progress-persist", "아직 모르겠어요", 3))
    )

    assert stopped.requested_fields == ()
    assert "같은 질문은 여기서 중단" in stopped.reply
    second_worker.close()


def test_revision_after_summary_requires_new_approval() -> None:
    worker = _worker()
    first = asyncio.run(worker.handle(_request("worker-revise", "15개 전체 입력")))
    assert first.status == "awaiting-approval"
    assert first.summary_version == 1
    revised = asyncio.run(worker.handle(_request("worker-revise", "타깃 변경", 2)))
    assert revised.status == "awaiting-approval"
    assert revised.summary_version == 2
    assert revised.approved_summary_version is None
    assert revised.facts["target_supporters"]["values"] == ["반려동물 가구"]


def test_sqlite_checkpointer_restores_collection_and_approval_state(tmp_path) -> None:
    path = tmp_path / "conversation.sqlite3"
    connection_one = sqlite3.connect(path, check_same_thread=False)
    saver_one = SqliteSaver(connection_one)
    saver_one.setup()
    first_worker = _worker(checkpointer=saver_one, connection=connection_one)
    asyncio.run(first_worker.handle(_request("worker-persist", "필수 제품 입력")))
    first = asyncio.run(
        first_worker.handle(_request("worker-persist", "선택 정보 전체 생략", 2))
    )
    assert first.status == "awaiting-approval"
    first_worker.close()

    connection_two = sqlite3.connect(path, check_same_thread=False)
    saver_two = SqliteSaver(connection_two)
    saver_two.setup()
    second_worker = _worker(checkpointer=saver_two, connection=connection_two)
    ready = asyncio.run(
        second_worker.handle(_request("worker-persist", "이 내용으로 생성해줘", 3))
    )
    assert ready.status == "generation-ready"
    assert ready.remaining_optional_fields == ()
    second_worker.close()


def test_sqlite_checkpointer_preserves_selected_optional_group_order(tmp_path) -> None:
    path = tmp_path / "selected-groups.sqlite3"
    connection_one = sqlite3.connect(path, check_same_thread=False)
    saver_one = SqliteSaver(connection_one)
    saver_one.setup()
    first_worker = _worker(checkpointer=saver_one, connection=connection_one)
    asyncio.run(first_worker.handle(_request("worker-groups", "필수 제품 입력")))
    selected = asyncio.run(
        first_worker.handle(_request("worker-groups", "정책과 프로젝트 설명부터", 2))
    )
    assert selected.active_optional_group == "policy"
    assert selected.optional_collection["selected_groups"] == [
        "policy",
        "project_explanation",
    ]
    first_worker.close()

    connection_two = sqlite3.connect(path, check_same_thread=False)
    saver_two = SqliteSaver(connection_two)
    saver_two.setup()
    second_worker = _worker(checkpointer=saver_two, connection=connection_two)
    restored = second_worker.get_state("worker-groups")["optional_collection"]
    assert restored["selected_groups"] == ["policy", "project_explanation"]
    assert restored["active_group"] == "policy"
    second_worker.close()


def test_delete_thread_starts_a_fresh_session() -> None:
    worker = _worker()
    first = asyncio.run(worker.handle(_request("worker-reset", "필수 제품 입력")))
    assert first.facts["product_name"]["status"] == "provided"
    worker.delete_thread("worker-reset")
    restarted = asyncio.run(worker.handle(_request("worker-reset", "바로 생성해줘", 2)))
    assert restarted.facts["product_name"]["status"] == "unknown"
    assert restarted.summary_version == 0


def test_graph_state_adapter_preserves_current_values_and_collection_gate() -> None:
    worker = _worker()
    state = _ready_worker_state(worker, "worker-adapter")
    semantic = graph_state_to_semantic_state(input_id="worker-adapter", state=state)
    assert semantic["slots"]["target_supporters"]["values"] == [
        "가구 아래 청소가 필요한 사용자"
    ]
    assert semantic["decision"]["ready_to_confirm"] is True
    DataRepository().validate_intake_semantic_state(semantic)


def test_grounded_brief_preserves_optional_input_without_domain_expansion() -> None:
    repository = DataRepository()
    worker = _worker()
    asyncio.run(worker.handle(_request("robot-grounded", "15개 전체 입력")))
    state = deepcopy(worker.get_state("robot-grounded"))
    state["facts"]["key_strengths"]["values"] = [
        "최대 6,800Pa",
        "먼지 비움과 충전만 지원하는 도크",
    ]
    state["facts"]["rewards"]["values"] = ["본품 얼리버드 1개"]
    state["facts"]["funding_end"]["values"] = ["2026-09-30"]
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
