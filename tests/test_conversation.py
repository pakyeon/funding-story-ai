from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from funding_story_ai.conversation import (
    FACT_FIELDS,
    OPTIONAL_FACT_FIELDS,
    OPTIONAL_FIELD_GROUPS,
    ApprovalDecision,
    CollectionDirective,
    ConversationNodes,
    FactPatch,
    OptionalCollection,
    QuestionPlan,
    QuestionRecord,
    StorySummary,
    TurnUnderstanding,
    apply_collection_directive,
    apply_fact_patches,
    build_conversation_graph,
    explicitly_absent_fields,
    initial_facts,
    initial_optional_collection,
    missing_required_fields,
    no_progress_repeat_count,
    provided_facts,
    question_plan_error,
    question_signature,
    reconcile_turn_understanding,
    skipped_optional_fields,
)


def test_progress_revision_allows_same_question_after_user_correction() -> None:
    plan = QuestionPlan(
        purpose="required",
        requested_fields=["product_name"],
        requested_detail="부족한 필수 제품 정보를 확인합니다.",
        question="제품명을 알려주세요.",
        rationale="필수 정보가 없습니다.",
    )
    state = {
        "facts_revision": 1,
        "collection_revision": 0,
        "summary_version": 0,
        "workflow_stage": "collecting",
        "question_history": [
            QuestionRecord(
                fields=["product_name"],
                purpose="required",
                requested_detail=plan.requested_detail,
                facts_revision=0,
                collection_revision=0,
                progress_fingerprint="0:0:0:collecting",
                signature=question_signature(plan),
            ).model_dump(mode="json")
        ],
    }

    assert no_progress_repeat_count(state, plan) == 0


def test_turn_understanding_rejects_duplicate_fact_patches() -> None:
    with pytest.raises(ValidationError, match="at most one patch"):
        TurnUnderstanding(
            intent="provide_information",
            fact_patches=[
                FactPatch(field="product_name", operation="replace", values=["A"]),
                FactPatch(field="product_name", operation="replace", values=["B"]),
            ],
        )


def test_reconcile_blocks_ambiguous_reference_patch_but_allows_named_revision() -> None:
    proposed = TurnUnderstanding(
        intent="revise_information",
        fact_patches=[
            FactPatch(
                field="target_supporters",
                operation="replace",
                values=["반려동물 가구"],
            )
        ],
    )

    blocked = reconcile_turn_understanding("그건 반려동물 가구로 바꿔줘.", proposed)
    allowed = reconcile_turn_understanding("타깃은 반려동물 가구로 바꿔줘.", proposed)

    assert blocked.fact_patches == []
    assert blocked.requires_clarification is True
    assert blocked.unresolved_references == ["그건"]
    assert allowed.fact_patches == proposed.fact_patches


def test_reconcile_does_not_store_a_competitor_as_the_user_product_type() -> None:
    proposed = TurnUnderstanding(
        intent="provide_information",
        fact_patches=[
            FactPatch(field="product_type", operation="replace", values=["로봇청소기"]),
            FactPatch(
                field="problem_context",
                operation="replace",
                values=["낮은 가구 밑에 들어가지 못함"],
            ),
        ],
    )

    reconciled = reconcile_turn_understanding(
        "기존 로봇청소기는 낮은 가구 밑에 들어가지 못해.", proposed
    )

    assert [patch.field for patch in reconciled.fact_patches] == ["problem_context"]


@pytest.mark.parametrize("message", ["알아서 해줘.", "응, 좋아."])
def test_reconcile_requires_clarification_for_bare_collection_replies(
    message: str,
) -> None:
    proposed = TurnUnderstanding(intent="unclear")

    reconciled = reconcile_turn_understanding(message, proposed)

    assert reconciled.collection_directive.action == "none"
    assert reconciled.collection_directive.requires_clarification is True
    assert reconciled.collection_directive.clarification_question


def test_reconcile_routes_a_pure_generation_request_to_missing_required_fields() -> None:
    proposed = TurnUnderstanding(
        intent="request_generation",
        clarification_fields=["product_name", "product_type"],
        requires_clarification=True,
        clarification_question="어떤 제품인지 알려주세요.",
    )

    reconciled = reconcile_turn_understanding("바로 스토리를 생성해줘.", proposed)

    assert reconciled.requires_clarification is False
    assert reconciled.clarification_fields == []
    assert reconciled.clarification_question is None


def test_reconcile_preserves_ambiguous_date_clarification_on_generation_request() -> None:
    proposed = TurnUnderstanding(
        intent="request_generation",
        clarification_fields=["funding_end", "shipping_start"],
        requires_clarification=True,
        clarification_question="11월 30일이 종료일인지 발송일인지 알려주세요.",
    )

    reconciled = reconcile_turn_understanding(
        "11월 30일 일정으로 스토리를 생성해줘.", proposed
    )

    assert reconciled.requires_clarification is True
    assert reconciled.clarification_fields == ["funding_end", "shipping_start"]


@pytest.mark.parametrize(
    "message",
    [
        "선택 정보 전체를 생략할게.",
        "선택 정보는 모두 건너뛰자.",
        "남은 선택 정보 전부를 생략해줘.",
    ],
)
def test_reconcile_preserves_an_explicit_skip_all_command(message: str) -> None:
    proposed = TurnUnderstanding(intent="request_generation")

    reconciled = reconcile_turn_understanding(message, proposed)

    assert reconciled.collection_directive.action == "skip_all_optional"


@pytest.mark.parametrize(
    "message",
    ["선택 정보 전체를 생략하면 안 돼.", "선택 정보 전체를 생략하지 않을게."],
)
def test_reconcile_does_not_turn_negative_skip_phrases_into_commands(message: str) -> None:
    proposed = TurnUnderstanding(intent="provide_information")

    reconciled = reconcile_turn_understanding(message, proposed)

    assert reconciled.collection_directive.action == "none"


def test_no_progress_clarification_preserves_required_collection_phase() -> None:
    state = {
        "question_purpose": "clarify",
        "question_candidate_fields": [],
        "collection_phase": "required",
        "messages": [],
    }

    result = ConversationNodes.handle_no_progress(state)

    assert result["collection_phase"] == "required"
    assert result["requested_fields"] == []


def test_clarification_question_preserves_the_current_collection_phase() -> None:
    plan = QuestionPlan(
        purpose="clarify",
        requested_fields=["product_name"],
        requested_detail="제품명 표기를 확인합니다.",
        question="제품명 표기를 확인해 주세요.",
        rationale="입력 의미 확인",
    )
    result = ConversationNodes.emit_question(
        {
            "current_question_plan": plan.model_dump(mode="json"),
            "collection_phase": "required",
            "facts_revision": 0,
            "collection_revision": 0,
            "summary_version": 0,
            "workflow_stage": "collecting",
            "messages": [],
            "question_history": [],
        }
    )

    assert result["collection_phase"] == "required"


def _patches(**values: list[str]) -> list[FactPatch]:
    return [
        FactPatch(field=field, operation="replace", values=value)
        for field, value in values.items()
    ]


def _required_turn() -> TurnUnderstanding:
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


def _all_fields_turn() -> TurnUnderstanding:
    values = {field: [f"{field} 값"] for field in FACT_FIELDS}
    return TurnUnderstanding(intent="provide_information", fact_patches=_patches(**values))


class _ConversationModel:
    def __init__(self, understandings: dict[str, TurnUnderstanding]) -> None:
        self.understandings = understandings
        self.invalid_plan = False
        self.invalid_repair = False

    def understand_turn(self, *, message, messages, facts, image_path):
        return self.understandings.get(message["content"], TurnUnderstanding(intent="unclear"))

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
        if self.invalid_plan:
            return QuestionPlan(
                purpose=purpose,
                requested_fields=["product_name"],
                requested_group=requested_group,
                requested_detail=requested_detail,
                question="잘못된 반복 질문",
                rationale="repair 동작을 검증합니다.",
            )
        requested = candidate_fields[:3]
        return QuestionPlan(
            purpose=purpose,
            requested_fields=requested,
            requested_group=requested_group,
            requested_detail=requested_detail,
            question=(
                requested_detail
                if purpose in {"clarify", "confirm-skip"}
                else f"{', '.join(requested)} 정보를 알려주세요."
            ),
            rationale="현재 목적과 후보에 맞는 질문입니다.",
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
        requested = ["product_name"] if self.invalid_repair else candidate_fields[:3]
        return QuestionPlan(
            purpose=purpose,
            requested_fields=requested,
            requested_group=requested_group,
            requested_detail=requested_detail,
            question="교정된 질문",
            rationale="질문 계획 오류를 교정합니다.",
        )

    def build_summary(self, *, messages, facts, optional_collection):
        confirmed = provided_facts(facts)
        return StorySummary(
            headline="스토리 생성 정보 요약",
            confirmed_facts=confirmed,
            explicitly_absent_fields=explicitly_absent_fields(facts),
            skipped_fields=skipped_optional_fields(optional_collection),
            summary_text=" / ".join(
                value for values in confirmed.values() for value in values
            ),
            confirmation_question="이 내용으로 스토리를 생성할까요?",
        )

    def classify_approval(self, *, message, summary, messages):
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


def test_optional_groups_cover_each_optional_field_once() -> None:
    flattened = [field for fields in OPTIONAL_FIELD_GROUPS.values() for field in fields]
    assert tuple(flattened) == OPTIONAL_FACT_FIELDS
    assert len(flattened) == len(set(flattened)) == 10
    assert set(flattened).isdisjoint(set(FACT_FIELDS[:5]))


def test_contracts_reject_invalid_fact_and_collection_operations() -> None:
    with pytest.raises(ValidationError):
        FactPatch(field="product_name", operation="replace", values=[])
    with pytest.raises(ValidationError):
        FactPatch(field="product_name", operation="mark_absent", values=["없음"])
    with pytest.raises(ValidationError):
        CollectionDirective(action="select_fields", fields=["product_name"])
    with pytest.raises(ValidationError):
        CollectionDirective(action="select_groups")


def test_fact_and_collection_states_are_independent_and_skip_can_be_resolved() -> None:
    facts = initial_facts()
    collection = OptionalCollection.model_validate(initial_optional_collection(facts))
    states = dict(collection.field_states)
    states["trust_elements"] = "skipped"
    state = {
        "incoming_message": {
            "id": "message-1",
            "role": "user",
            "content": "시험 성적서를 추가할게",
        },
        "messages": [
            {"id": "message-1", "role": "user", "content": "시험 성적서를 추가할게"}
        ],
        "facts": facts,
        "optional_collection": collection.model_copy(
            update={"offered": True, "field_states": states}
        ).model_dump(mode="json"),
        "turn_understanding": TurnUnderstanding(
            intent="provide_information",
            fact_patches=_patches(trust_elements=["KTL 시험 성적서"]),
        ).model_dump(mode="json"),
    }
    updated = apply_fact_patches(state)
    assert updated["facts"]["trust_elements"]["status"] == "provided"
    assert updated["optional_collection"]["field_states"]["trust_elements"] == "resolved"

    facts = deepcopy(updated["facts"])
    facts["maker_team_intro"] = {
        "status": "explicitly-absent",
        "values": [],
        "source_message_ids": ["message-2"],
        "updated_at_turn": 2,
    }
    assert facts["maker_team_intro"]["status"] == "explicitly-absent"
    assert updated["facts"]["problem_context"]["status"] == "unknown"


@pytest.mark.parametrize(
    ("directive", "expected_pending", "expected_skipped"),
    [
        (
            CollectionDirective(action="continue_recommended"),
            list(OPTIONAL_FACT_FIELDS),
            [],
        ),
        (
            CollectionDirective(
                action="select_groups", groups=["policy", "project_explanation"]
            ),
            [
                *OPTIONAL_FIELD_GROUPS["policy"],
                *OPTIONAL_FIELD_GROUPS["project_explanation"],
            ],
            [],
        ),
        (
            CollectionDirective(
                action="select_fields", fields=["rewards", "risk_response"]
            ),
            ["rewards", "risk_response"],
            [],
        ),
        (
            CollectionDirective(
                action="skip_fields", fields=["refund_policy", "as_policy"]
            ),
            [],
            ["refund_policy", "as_policy"],
        ),
    ],
)
def test_collection_directives_route_groups_fields_and_skips(
    directive: CollectionDirective,
    expected_pending: list[str],
    expected_skipped: list[str],
) -> None:
    facts = initial_facts()
    state = {
        "facts": facts,
        "optional_collection": OptionalCollection.model_validate(
            initial_optional_collection(facts)
        ).model_copy(update={"offered": True}).model_dump(mode="json"),
        "turn_understanding": TurnUnderstanding(
            intent="provide_information",
            collection_directive=directive,
        ).model_dump(mode="json"),
    }
    updated = apply_collection_directive(state)
    assert updated["optional_collection"]["pending_fields"] == expected_pending
    assert [
        field
        for field, status in updated["optional_collection"]["field_states"].items()
        if status == "skipped"
    ] == expected_skipped


def test_return_to_optional_reopens_skipped_fields() -> None:
    facts = initial_facts()
    collection = OptionalCollection.model_validate(initial_optional_collection(facts))
    states = dict(collection.field_states)
    states["refund_policy"] = "skipped"
    states["as_policy"] = "skipped"
    state = {
        "facts": facts,
        "optional_collection": collection.model_copy(
            update={"offered": True, "field_states": states}
        ).model_dump(mode="json"),
        "turn_understanding": TurnUnderstanding(
            intent="revise_information",
            collection_directive=CollectionDirective(
                action="return_to_optional", fields=["refund_policy"]
            ),
        ).model_dump(mode="json"),
    }
    updated = apply_collection_directive(state)
    assert updated["optional_collection"]["pending_fields"] == ["refund_policy"]
    assert updated["optional_collection"]["field_states"]["refund_policy"] == "requested"
    assert updated["optional_collection"]["field_states"]["as_policy"] == "skipped"


def test_question_repetition_uses_purpose_detail_and_fact_revision() -> None:
    base_state = {
        "question_purpose": "clarify",
        "question_candidate_fields": ["product_name"],
        "question_group": None,
        "question_requested_detail": "제품명 표기를 확인합니다.",
        "facts_revision": 2,
        "question_history": [
            {
                "fields": ["product_name"],
                "purpose": "clarify",
                "requested_group": None,
                "requested_detail": "제품명 표기를 확인합니다.",
                "facts_revision": 2,
                "resolved": True,
            }
        ],
    }
    plan = QuestionPlan(
        purpose="clarify",
        requested_fields=["product_name"],
        requested_group=None,
        requested_detail="제품명 표기를 확인합니다.",
        question="제품명 표기를 다시 확인해 주세요.",
        rationale="표기 충돌 확인",
    )
    assert question_plan_error(base_state, plan) is not None
    assert question_plan_error({**base_state, "facts_revision": 3}, plan) is None
    changed_detail = plan.model_copy(
        update={"requested_detail": "제품명에 영문 대소문자를 확인합니다."}
    )
    changed_state = {
        **base_state,
        "question_requested_detail": "제품명에 영문 대소문자를 확인합니다.",
    }
    assert question_plan_error(changed_state, changed_detail) is None


def test_question_plan_requires_all_candidates_and_hides_internal_field_names() -> None:
    state = {
        "question_purpose": "optional-collect",
        "question_candidate_fields": [
            "funding_plan",
            "risk_response",
        ],
        "question_group": "project_explanation",
        "question_requested_detail": "프로젝트 설명 정보를 확인합니다.",
        "facts_revision": 1,
        "collection_revision": 1,
        "question_history": [],
    }
    omitted = QuestionPlan(
        purpose="optional-collect",
        requested_fields=["funding_plan"],
        requested_group="project_explanation",
        requested_detail="프로젝트 설명 정보를 확인합니다.",
        question="펀딩금 사용 계획과 위험 대응 계획을 알려주세요.",
        rationale="선택 정보 수집",
    )
    assert "every planned candidate" in str(question_plan_error(state, omitted))

    exposed = omitted.model_copy(
        update={
            "requested_fields": [
                "funding_plan",
                "risk_response",
            ],
            "question": "funding_plan, risk_response를 알려주세요.",
        }
    )
    assert "internal fact field" in str(question_plan_error(state, exposed))


def test_required_optional_skip_summary_and_generation_ready_flow() -> None:
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
            "스토리 설득 정보부터": TurnUnderstanding(
                intent="provide_information",
                collection_directive=CollectionDirective(
                    action="select_groups", groups=["story_persuasion"]
                ),
            ),
            "설득 정보 답변": TurnUnderstanding(
                intent="provide_information",
                fact_patches=[
                    *_patches(
                        problem_context=["가구 아래 반복 청소"],
                        trust_elements=["KTL 시험 성적서"],
                    ),
                    FactPatch(field="maker_team_intro", operation="mark_absent"),
                ],
            ),
            "나머지는 모두 생략": TurnUnderstanding(
                intent="request_generation",
                collection_directive=CollectionDirective(action="skip_all_optional"),
            ),
        }
    )
    graph = build_conversation_graph(model, checkpointer=InMemorySaver())

    required_question = _invoke(graph, "thread-flow", "바로 만들어줘", 1)
    assert required_question["collection_phase"] == "required"
    assert len(required_question["requested_fields"]) <= 3

    offer = _invoke(graph, "thread-flow", "강점과 타깃", 2)
    assert offer["collection_phase"] == "optional-offer"
    assert offer["optional_collection"]["offered"] is True
    assert "build_summary" not in offer["visited_nodes"]

    group_question = _invoke(graph, "thread-flow", "스토리 설득 정보부터", 3)
    assert group_question["collection_phase"] == "optional-collect"
    assert group_question["optional_collection"]["active_group"] == "story_persuasion"
    assert set(group_question["requested_fields"]) == set(
        OPTIONAL_FIELD_GROUPS["story_persuasion"]
    )

    next_choice = _invoke(graph, "thread-flow", "설득 정보 답변", 4)
    assert next_choice["collection_phase"] == "optional-offer"
    assert next_choice["facts"]["maker_team_intro"]["status"] == "explicitly-absent"

    summary = _invoke(graph, "thread-flow", "나머지는 모두 생략", 5)
    assert summary["workflow_stage"] == "awaiting-approval"
    assert summary["summary_version"] == 1
    assert summary["current_summary"]["explicitly_absent_fields"] == [
        "maker_team_intro"
    ]
    assert len(summary["current_summary"]["skipped_fields"]) == 7
    assert summary["workflow_stage"] != "generation-ready"

    ambiguous = _invoke(graph, "thread-flow", "네", 6)
    assert ambiguous["workflow_stage"] == "awaiting-approval"
    assert ambiguous["approved_summary_version"] is None

    ready = _invoke(graph, "thread-flow", "이 내용으로 생성해줘", 7)
    assert ready["workflow_stage"] == "generation-ready"
    assert ready["approved_summary_version"] == ready["summary_version"]


def test_active_group_answer_applies_cross_group_facts_without_reasking_them() -> None:
    model = _ConversationModel(
        {
            "필수 입력": _required_turn(),
            "권장 순서": TurnUnderstanding(
                intent="provide_information",
                collection_directive=CollectionDirective(action="continue_recommended"),
            ),
            "설득과 리워드 답변": TurnUnderstanding(
                intent="provide_information",
                fact_patches=_patches(
                    problem_context=["반복 청소"],
                    trust_elements=["시험 성적서"],
                    maker_team_intro=["가전 개발팀"],
                    rewards=["본품 1개"],
                ),
            ),
        }
    )
    graph = build_conversation_graph(model, checkpointer=InMemorySaver())
    _invoke(graph, "thread-cross-group", "필수 입력", 1)
    first_group = _invoke(graph, "thread-cross-group", "권장 순서", 2)
    assert first_group["optional_collection"]["active_group"] == "story_persuasion"
    second_group = _invoke(graph, "thread-cross-group", "설득과 리워드 답변", 3)
    assert second_group["facts"]["rewards"]["status"] == "provided"
    assert "rewards" not in second_group["optional_collection"]["pending_fields"]
    assert "rewards" not in second_group["requested_fields"]
    assert set(second_group["requested_fields"]) == {"funding_end", "shipping_start"}


def test_all_fifteen_fields_skip_optional_offer_and_go_to_summary() -> None:
    model = _ConversationModel({"15개 전체 입력": _all_fields_turn()})
    graph = build_conversation_graph(model, checkpointer=InMemorySaver())
    state = _invoke(graph, "thread-all", "15개 전체 입력", 1)
    assert state["workflow_stage"] == "awaiting-approval"
    assert "offer_optional_information" not in state["visited_nodes"]
    assert state["current_summary"]["skipped_fields"] == []
    assert len(state["current_summary"]["confirmed_facts"]) == 15


def test_ambiguous_skip_is_confirmed_without_changing_collection_state() -> None:
    model = _ConversationModel(
        {
            "필수 입력": _required_turn(),
            "그냥 해줘": TurnUnderstanding(
                intent="request_generation",
                collection_directive=CollectionDirective(
                    action="none",
                    fields=["rewards"],
                    requires_clarification=True,
                    clarification_question="리워드를 생략한다는 뜻인지 알려주세요.",
                ),
            ),
        }
    )
    graph = build_conversation_graph(model, checkpointer=InMemorySaver())
    _invoke(graph, "thread-ambiguous-skip", "필수 입력", 1)
    state = _invoke(graph, "thread-ambiguous-skip", "그냥 해줘", 2)
    assert state["current_question_plan"]["purpose"] == "confirm-skip"
    assert state["optional_collection"]["field_states"]["rewards"] != "skipped"
    assert state["workflow_stage"] == "collecting"


def test_invalid_question_plan_is_repaired_then_uses_deterministic_fallback() -> None:
    model = _ConversationModel(
        {
            "부분 입력": TurnUnderstanding(
                intent="provide_information",
                fact_patches=_patches(product_name=["OrbitClean V3"]),
            )
        }
    )
    model.invalid_plan = True
    model.invalid_repair = True
    graph = build_conversation_graph(model, checkpointer=InMemorySaver())
    state = _invoke(graph, "thread-repair", "부분 입력", 1)
    assert "repair_question_plan" in state["visited_nodes"]
    assert set(state["requested_fields"]).issubset(set(missing_required_fields(state["facts"])))
    assert 1 <= len(state["requested_fields"]) <= 3
    assert "을(를)" not in state["reply"]


def test_revision_after_summary_invalidates_old_version_and_rebuilds() -> None:
    all_fields = _all_fields_turn()
    model = _ConversationModel(
        {
            "15개 전체 입력": all_fields,
            "타깃 수정": TurnUnderstanding(
                intent="revise_information",
                fact_patches=_patches(target_supporters=["20~40대 반려동물 가구"]),
            ),
        }
    )
    graph = build_conversation_graph(model, checkpointer=InMemorySaver())
    first = _invoke(graph, "thread-revision", "15개 전체 입력", 1)
    assert first["summary_version"] == 1
    revised = _invoke(graph, "thread-revision", "타깃 수정", 2)
    assert revised["workflow_stage"] == "awaiting-approval"
    assert revised["summary_version"] == 2
    assert revised["approved_summary_version"] is None
    assert revised["facts"]["target_supporters"]["values"] == [
        "20~40대 반려동물 가구"
    ]


def test_threads_do_not_share_conversation_or_question_history() -> None:
    model = _ConversationModel({"필수 입력": _required_turn()})
    graph = build_conversation_graph(model, checkpointer=InMemorySaver())
    _invoke(graph, "thread-a", "필수 입력", 1)
    other = _invoke(graph, "thread-b", "알 수 없는 입력", 1)
    assert other["facts"]["product_name"]["status"] == "unknown"
    assert other["optional_collection"]["offered"] is False


def test_summary_grounding_rejects_changed_fact_values() -> None:
    model = _ConversationModel({"15개 전체 입력": _all_fields_turn()})
    original = model.build_summary

    def hallucinated_summary(*, messages, facts, optional_collection):
        summary = original(
            messages=messages,
            facts=facts,
            optional_collection=optional_collection,
        ).model_copy(deep=True)
        changed = deepcopy(summary.confirmed_facts)
        changed["key_strengths"] = ["존재하지 않는 강점"]
        return summary.model_copy(update={"confirmed_facts": changed})

    model.build_summary = hallucinated_summary  # type: ignore[method-assign]
    graph = build_conversation_graph(model, checkpointer=InMemorySaver())
    with pytest.raises(ValueError, match="exactly match"):
        _invoke(graph, "thread-hallucination", "15개 전체 입력", 1)
