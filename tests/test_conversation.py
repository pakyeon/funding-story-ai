from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from funding_story_ai.conversation import (
    ApprovalDecision,
    FactPatch,
    QuestionPlan,
    StorySummary,
    TurnUnderstanding,
    apply_fact_patches,
    build_conversation_graph,
    initial_facts,
    missing_required_fields,
    provided_facts,
)


def _patches(**values: list[str]) -> list[FactPatch]:
    return [
        FactPatch(field=field, operation="replace", values=value)
        for field, value in values.items()
    ]


class _ConversationModel:
    def __init__(self, turns: dict[str, TurnUnderstanding] | None = None) -> None:
        self.turns = turns or {}
        self.calls: list[tuple[str, Any]] = []

    def understand_turn(self, *, message, messages, facts, image_path):
        self.calls.append(("understand_turn", message["content"]))
        return self.turns.get(
            message["content"],
            TurnUnderstanding(intent="unclear"),
        )

    def plan_questions(
        self,
        *,
        messages,
        facts,
        missing_required_fields,
        asked_topics,
        turn_understanding,
    ):
        self.calls.append(("plan_questions", tuple(missing_required_fields)))
        requested = missing_required_fields[:2]
        if not requested and turn_understanding.requires_clarification:
            return QuestionPlan(
                requested_fields=[],
                question=turn_understanding.clarification_question,
                rationale="불명확한 지시 대상을 확인합니다.",
            )
        return QuestionPlan(
            requested_fields=requested,
            question=f"{', '.join(requested)} 정보를 알려주세요.",
            rationale="필수 정보가 부족합니다.",
        )

    def build_summary(self, *, messages, facts):
        self.calls.append(("build_summary", None))
        confirmed = provided_facts(facts)
        unconfirmed = [field for field, value in facts.items() if value["status"] != "provided"]
        text = " / ".join(value for values in confirmed.values() for value in values)
        return StorySummary(
            headline="스토리 생성 정보 요약",
            confirmed_facts=confirmed,
            unconfirmed_fields=unconfirmed,
            summary_text=text,
            confirmation_question="이 내용으로 스토리를 생성할까요?",
        )

    def classify_approval(self, *, message, summary, messages):
        self.calls.append(("classify_approval", message["content"]))
        content = message["content"]
        if content == "이 내용으로 생성해줘":
            return ApprovalDecision(decision="approve", reason="명시적 생성 요청")
        if content == "네":
            return ApprovalDecision(decision="ambiguous", reason="대상이 명확하지 않음")
        if content == "생성하지 않을게":
            return ApprovalDecision(decision="reject", reason="명시적 거절")
        return ApprovalDecision(decision="revise", reason="정보 수정 포함")


def _invoke(graph, thread_id: str, message: str, index: int) -> dict[str, Any]:
    user_message = {"id": f"message-{index}", "role": "user", "content": message}
    return graph.invoke(
        {
            "incoming_message": user_message,
            "messages": [user_message],
            "input_id": thread_id,
        },
        {"configurable": {"thread_id": thread_id}},
    )


def _complete_turn() -> TurnUnderstanding:
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


def test_fact_patch_contract_rejects_invalid_operations() -> None:
    with pytest.raises(ValidationError):
        FactPatch(field="product_name", operation="replace", values=[])
    with pytest.raises(ValidationError):
        FactPatch(field="product_name", operation="mark_absent", values=["없음"])
    with pytest.raises(ValidationError):
        FactPatch(field="not-a-field", operation="replace", values=["value"])


def test_fact_patch_reducer_preserves_superseded_value_and_invalidates_approval() -> None:
    facts = initial_facts()
    facts["target_supporters"] = {
        "status": "provided",
        "values": ["30대 맞벌이"],
        "source_message_ids": ["message-1"],
        "updated_at_turn": 1,
    }
    state = {
        "incoming_message": {"id": "message-2", "role": "user", "content": "타깃 수정"},
        "messages": [
            {"id": "message-1", "role": "user", "content": "기존"},
            {"id": "message-2", "role": "user", "content": "타깃 수정"},
        ],
        "facts": facts,
        "fact_history": [],
        "facts_revision": 1,
        "current_summary": {"old": True},
        "approval_pending": True,
        "approved_summary_version": 1,
        "turn_understanding": TurnUnderstanding(
            intent="revise_information",
            fact_patches=_patches(target_supporters=["20~40대 반려동물 가구"]),
        ).model_dump(mode="json"),
    }
    updated = apply_fact_patches(state)
    assert updated["facts"]["target_supporters"]["values"] == [
        "20~40대 반려동물 가구"
    ]
    assert updated["fact_history"][0]["previous"]["values"] == ["30대 맞벌이"]
    assert updated["facts_revision"] == 2
    assert updated["current_summary"] is None
    assert updated["approval_pending"] is False
    assert updated["approved_summary_version"] is None


def test_required_field_check_is_deterministic() -> None:
    facts = initial_facts()
    assert missing_required_fields(facts) == [
        "product_name",
        "product_type",
        "category",
        "key_strengths",
        "target_supporters",
    ]
    for field in ("product_name", "product_type", "category", "key_strengths"):
        facts[field] = {
            "status": "provided",
            "values": [field],
            "source_message_ids": ["message-1"],
            "updated_at_turn": 1,
        }
    facts["target_supporters"] = {
        "status": "explicitly-absent",
        "values": [],
        "source_message_ids": ["message-1"],
        "updated_at_turn": 1,
    }
    assert missing_required_fields(facts) == ["target_supporters"]


def test_graph_runs_question_summary_ambiguous_and_approval_flow() -> None:
    model = _ConversationModel(
        {
            "바로 만들어줘": TurnUnderstanding(
                intent="request_generation",
                fact_patches=_patches(
                    product_name=["OrbitClean V3"],
                    product_type=["로봇청소기"],
                    category=["테크·가전"],
                ),
            ),
            "강점과 타깃": TurnUnderstanding(
                intent="provide_information",
                fact_patches=_patches(
                    key_strengths=["얇은 본체"],
                    target_supporters=["가구 아래 청소가 필요한 사용자"],
                ),
            ),
        }
    )
    graph = build_conversation_graph(model, checkpointer=InMemorySaver())

    first = _invoke(graph, "thread-flow", "바로 만들어줘", 1)
    assert first["workflow_stage"] == "collecting"
    assert set(first["requested_fields"]) <= {"key_strengths", "target_supporters"}

    second = _invoke(graph, "thread-flow", "강점과 타깃", 2)
    assert second["workflow_stage"] == "awaiting-approval"
    assert second["summary_version"] == 1
    assert second["approval_pending"] is True

    ambiguous = _invoke(graph, "thread-flow", "네", 3)
    assert ambiguous["workflow_stage"] == "awaiting-approval"
    assert ambiguous["approved_summary_version"] is None

    approved = _invoke(graph, "thread-flow", "이 내용으로 생성해줘", 4)
    assert approved["workflow_stage"] == "approved"
    assert approved["approved_summary_version"] == approved["summary_version"]
    assert "approval_guard" in approved["visited_nodes"]


def test_revision_after_summary_rebuilds_summary_and_invalidates_old_version() -> None:
    model = _ConversationModel(
        {
            "완전한 입력": _complete_turn(),
            "타깃을 수정할게": TurnUnderstanding(
                intent="revise_information",
                fact_patches=_patches(target_supporters=["20~40대 반려동물 가구"]),
            ),
        }
    )
    graph = build_conversation_graph(model, checkpointer=InMemorySaver())
    first = _invoke(graph, "thread-revision", "완전한 입력", 1)
    assert first["summary_version"] == 1

    revised = _invoke(graph, "thread-revision", "타깃을 수정할게", 2)
    assert revised["workflow_stage"] == "awaiting-approval"
    assert revised["summary_version"] == 2
    assert revised["approved_summary_version"] is None
    assert revised["facts"]["target_supporters"]["values"] == [
        "20~40대 반려동물 가구"
    ]


def test_threads_do_not_share_conversation_state() -> None:
    model = _ConversationModel({"완전한 입력": _complete_turn()})
    graph = build_conversation_graph(model, checkpointer=InMemorySaver())
    _invoke(graph, "thread-a", "완전한 입력", 1)
    other = _invoke(graph, "thread-b", "알 수 없는 입력", 1)
    assert other["facts"]["product_name"]["status"] == "unknown"
    assert other["workflow_stage"] == "collecting"


def test_question_plan_cannot_repeat_an_already_provided_field() -> None:
    model = _ConversationModel(
        {
            "부분 입력": TurnUnderstanding(
                intent="provide_information",
                fact_patches=_patches(product_name=["OrbitClean V3"]),
            )
        }
    )

    def repeated_question(**kwargs):
        return QuestionPlan(
            requested_fields=["product_name"],
            question="제품명을 다시 알려주세요.",
            rationale="잘못된 반복 질문",
        )

    model.plan_questions = repeated_question  # type: ignore[method-assign]
    graph = build_conversation_graph(model, checkpointer=InMemorySaver())
    with pytest.raises(ValueError, match="already provided"):
        _invoke(graph, "thread-repeat", "부분 입력", 1)


def test_summary_grounding_rejects_changed_fact_values() -> None:
    model = _ConversationModel({"완전한 입력": _complete_turn()})
    original = model.build_summary

    def hallucinated_summary(*, messages, facts):
        summary = original(messages=messages, facts=facts).model_copy(deep=True)
        changed = deepcopy(summary.confirmed_facts)
        changed["key_strengths"] = ["존재하지 않는 강점"]
        return summary.model_copy(update={"confirmed_facts": changed})

    model.build_summary = hallucinated_summary  # type: ignore[method-assign]
    graph = build_conversation_graph(model, checkpointer=InMemorySaver())
    with pytest.raises(ValueError, match="exactly match"):
        _invoke(graph, "thread-hallucination", "완전한 입력", 1)
