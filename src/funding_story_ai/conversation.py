from __future__ import annotations

import operator
import re
from copy import deepcopy
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FACT_FIELDS = (
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
    "platform_choice",
    "risk_response",
)
FactField = Literal[
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
    "platform_choice",
    "risk_response",
]
REQUIRED_FACT_FIELDS: tuple[FactField, ...] = (
    "product_name",
    "product_type",
    "category",
    "key_strengths",
    "target_supporters",
)

FactStatus = Literal["provided", "explicitly-absent", "unknown"]
FactOperation = Literal["replace", "append", "mark_absent", "clear"]
TurnIntent = Literal[
    "provide_information",
    "revise_information",
    "request_generation",
    "ask_question",
    "cancel",
    "unclear",
]
ApprovalKind = Literal["approve", "revise", "reject", "ambiguous"]
WorkflowStage = Literal[
    "collecting",
    "awaiting-approval",
    "approved",
    "submitted",
    "cancelled",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FactPatch(StrictModel):
    field: FactField
    operation: FactOperation
    values: list[str] = Field(default_factory=list)
    reason: str | None = None

    @field_validator("values")
    @classmethod
    def normalize_values(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values if value.strip()]
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def validate_operation_values(self) -> FactPatch:
        if self.operation in {"replace", "append"} and not self.values:
            raise ValueError(f"{self.operation} requires at least one value")
        if self.operation in {"mark_absent", "clear"} and self.values:
            raise ValueError(f"{self.operation} does not accept values")
        return self


class TurnUnderstanding(StrictModel):
    intent: TurnIntent
    fact_patches: list[FactPatch] = Field(default_factory=list)
    unresolved_references: list[str] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_question: str | None = None

    @model_validator(mode="after")
    def validate_clarification(self) -> TurnUnderstanding:
        if self.requires_clarification and not (
            self.clarification_question and self.clarification_question.strip()
        ):
            raise ValueError("clarification_question is required when clarification is needed")
        return self


class QuestionPlan(StrictModel):
    requested_fields: list[FactField]
    question: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @field_validator("requested_fields")
    @classmethod
    def unique_fields(cls, fields: list[FactField]) -> list[FactField]:
        return list(dict.fromkeys(fields))


class StorySummary(StrictModel):
    headline: str = Field(min_length=1)
    confirmed_facts: dict[FactField, list[str]]
    unconfirmed_fields: list[FactField]
    summary_text: str = Field(min_length=1)
    confirmation_question: str = Field(min_length=1)


class ApprovalDecision(StrictModel):
    decision: ApprovalKind
    reason: str = Field(min_length=1)


class FactValue(StrictModel):
    status: FactStatus = "unknown"
    values: list[str] = Field(default_factory=list)
    source_message_ids: list[str] = Field(default_factory=list)
    updated_at_turn: int = 0

    @model_validator(mode="after")
    def validate_status_values(self) -> FactValue:
        if self.status == "provided" and not self.values:
            raise ValueError("provided facts require values")
        if self.status != "provided" and self.values:
            raise ValueError(f"{self.status} facts cannot contain values")
        return self


class FactChange(StrictModel):
    field: FactField
    previous: FactValue
    current: FactValue
    source_message_id: str
    turn: int


class StoryWorkerState(TypedDict, total=False):
    messages: Annotated[list[dict[str, str]], operator.add]
    visited_nodes: Annotated[list[str], operator.add]
    incoming_message: dict[str, str]
    image_path: str | None
    image_attached: bool
    input_id: str
    caller_id: str
    idempotency_key: str
    facts: dict[str, dict[str, Any]]
    fact_history: list[dict[str, Any]]
    facts_revision: int
    missing_required_fields: list[str]
    asked_topics: list[str]
    current_question_plan: dict[str, Any] | None
    turn_understanding: dict[str, Any] | None
    current_summary: dict[str, Any] | None
    summary_version: int
    summary_facts_revision: int | None
    approval_pending: bool
    approved_summary_version: int | None
    approval_decision: dict[str, Any] | None
    workflow_stage: WorkflowStage
    reply: str
    requested_fields: list[str]
    tool_result: dict[str, Any] | None


class ConversationModel(Protocol):
    def understand_turn(
        self,
        *,
        message: dict[str, str],
        messages: list[dict[str, str]],
        facts: dict[str, dict[str, Any]],
        image_path: str | None,
    ) -> TurnUnderstanding: ...

    def plan_questions(
        self,
        *,
        messages: list[dict[str, str]],
        facts: dict[str, dict[str, Any]],
        missing_required_fields: list[str],
        asked_topics: list[str],
        turn_understanding: TurnUnderstanding,
    ) -> QuestionPlan: ...

    def build_summary(
        self,
        *,
        messages: list[dict[str, str]],
        facts: dict[str, dict[str, Any]],
    ) -> StorySummary: ...

    def classify_approval(
        self,
        *,
        message: dict[str, str],
        summary: StorySummary,
        messages: list[dict[str, str]],
    ) -> ApprovalDecision: ...


def initial_facts() -> dict[str, dict[str, Any]]:
    return {field: FactValue().model_dump(mode="json") for field in FACT_FIELDS}


def provided_facts(facts: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    return {
        field: list(FactValue.model_validate(value).values)
        for field, value in facts.items()
        if FactValue.model_validate(value).status == "provided"
    }


def unconfirmed_fields(facts: dict[str, dict[str, Any]]) -> list[str]:
    return [
        field
        for field in FACT_FIELDS
        if FactValue.model_validate(facts.get(field, FactValue().model_dump())).status
        != "provided"
    ]


def missing_required_fields(facts: dict[str, dict[str, Any]]) -> list[str]:
    return [
        field
        for field in REQUIRED_FACT_FIELDS
        if FactValue.model_validate(facts.get(field, FactValue().model_dump())).status
        != "provided"
    ]


def apply_fact_patches(state: StoryWorkerState) -> StoryWorkerState:
    understanding = TurnUnderstanding.model_validate(state.get("turn_understanding") or {})
    message = state["incoming_message"]
    message_id = message["id"]
    turn = sum(1 for item in state.get("messages", []) if item["role"] == "user")
    facts = deepcopy(state.get("facts") or initial_facts())
    history = list(state.get("fact_history", []))
    changed = False

    for patch in understanding.fact_patches:
        previous = FactValue.model_validate(facts[patch.field])
        values = list(previous.values)
        status: FactStatus = previous.status
        if patch.operation == "replace":
            values = list(patch.values)
            status = "provided"
        elif patch.operation == "append":
            values = list(dict.fromkeys([*values, *patch.values]))
            status = "provided"
        elif patch.operation == "mark_absent":
            values = []
            status = "explicitly-absent"
        elif patch.operation == "clear":
            values = []
            status = "unknown"

        materially_changed = previous.status != status or previous.values != values
        source_ids = list(previous.source_message_ids)
        if materially_changed and message_id not in source_ids:
            source_ids.append(message_id)
        current = FactValue(
            status=status,
            values=values,
            source_message_ids=source_ids,
            updated_at_turn=turn if materially_changed else previous.updated_at_turn,
        )
        if materially_changed:
            history.append(
                FactChange(
                    field=patch.field,
                    previous=previous,
                    current=current,
                    source_message_id=message_id,
                    turn=turn,
                ).model_dump(mode="json")
            )
            facts[patch.field] = current.model_dump(mode="json")
            changed = True

    update: StoryWorkerState = {
        "facts": facts,
        "fact_history": history,
        "facts_revision": int(state.get("facts_revision", 0)) + int(changed),
        "visited_nodes": ["apply_fact_patch"],
    }
    if changed:
        update.update(
            {
                "current_summary": None,
                "summary_facts_revision": None,
                "approval_pending": False,
                "approved_summary_version": None,
            }
        )
    return update


def validate_summary_grounding(
    summary: StorySummary, facts: dict[str, dict[str, Any]]
) -> None:
    expected_facts = provided_facts(facts)
    actual_facts = {str(field): values for field, values in summary.confirmed_facts.items()}
    if actual_facts != expected_facts:
        raise ValueError("Summary confirmed_facts must exactly match current provided facts")
    if list(summary.unconfirmed_fields) != unconfirmed_fields(facts):
        raise ValueError("Summary unconfirmed_fields must match current unprovided fields")

    allowed_text = " ".join(
        value for values in expected_facts.values() for value in values
    )
    unsupported_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", summary.summary_text)) - set(
        re.findall(r"\d+(?:[.,]\d+)?", allowed_text)
    )
    if unsupported_numbers:
        raise ValueError(
            f"Summary introduced unsupported numeric claims: {sorted(unsupported_numbers)}"
        )


class ConversationNodes:
    def __init__(self, model: ConversationModel) -> None:
        self.model = model

    @staticmethod
    def route_start(state: StoryWorkerState) -> str:
        if state.get("workflow_stage") in {"submitted", "cancelled"}:
            return "closed_session"
        if state.get("approval_pending", False):
            return "classify_approval"
        return "understand_turn"

    def understand_turn(self, state: StoryWorkerState) -> StoryWorkerState:
        facts = state.get("facts") or initial_facts()
        understanding = self.model.understand_turn(
            message=state["incoming_message"],
            messages=state.get("messages", []),
            facts=facts,
            image_path=state.get("image_path"),
        )
        return {
            "facts": facts,
            "turn_understanding": understanding.model_dump(mode="json"),
            "approval_decision": None,
            "visited_nodes": ["understand_turn"],
        }

    @staticmethod
    def route_understanding(state: StoryWorkerState) -> str:
        understanding = TurnUnderstanding.model_validate(state["turn_understanding"])
        return "cancel_session" if understanding.intent == "cancel" else "apply_fact_patch"

    @staticmethod
    def apply_fact_patch(state: StoryWorkerState) -> StoryWorkerState:
        return apply_fact_patches(state)

    @staticmethod
    def check_required_fields(state: StoryWorkerState) -> StoryWorkerState:
        missing = missing_required_fields(state["facts"])
        return {
            "missing_required_fields": missing,
            "visited_nodes": ["check_required_fields"],
        }

    @staticmethod
    def route_after_required_check(state: StoryWorkerState) -> str:
        understanding = TurnUnderstanding.model_validate(state["turn_understanding"])
        if state.get("missing_required_fields") or understanding.requires_clarification:
            return "plan_next_questions"
        return "build_summary"

    def plan_next_questions(self, state: StoryWorkerState) -> StoryWorkerState:
        understanding = TurnUnderstanding.model_validate(state["turn_understanding"])
        plan = self.model.plan_questions(
            messages=state.get("messages", []),
            facts=state["facts"],
            missing_required_fields=list(state.get("missing_required_fields", [])),
            asked_topics=list(state.get("asked_topics", [])),
            turn_understanding=understanding,
        )
        missing = set(state.get("missing_required_fields", []))
        already_provided = set(provided_facts(state["facts"]))
        repeated = already_provided.intersection(plan.requested_fields)
        if repeated:
            raise ValueError(
                f"Question plan repeated already provided fields: {sorted(repeated)}"
            )
        if missing and not missing.intersection(plan.requested_fields):
            raise ValueError("Question plan must address at least one missing required field")
        if not plan.requested_fields and not understanding.requires_clarification:
            raise ValueError("Question plan must request at least one field")
        return {
            "current_question_plan": plan.model_dump(mode="json"),
            "visited_nodes": ["plan_next_questions"],
        }

    @staticmethod
    def emit_question(state: StoryWorkerState) -> StoryWorkerState:
        plan = QuestionPlan.model_validate(state["current_question_plan"])
        asked = list(dict.fromkeys([*state.get("asked_topics", []), *plan.requested_fields]))
        assistant_message = {
            "id": f"assistant-question-{len(state.get('messages', [])) + 1}",
            "role": "assistant",
            "content": plan.question,
        }
        return {
            "messages": [assistant_message],
            "asked_topics": asked,
            "workflow_stage": "collecting",
            "reply": plan.question,
            "requested_fields": list(plan.requested_fields),
            "approval_pending": False,
            "visited_nodes": ["emit_question"],
        }

    def build_summary(self, state: StoryWorkerState) -> StoryWorkerState:
        summary = self.model.build_summary(
            messages=state.get("messages", []),
            facts=state["facts"],
        )
        validate_summary_grounding(summary, state["facts"])
        return {
            "current_summary": summary.model_dump(mode="json"),
            "summary_version": int(state.get("summary_version", 0)) + 1,
            "summary_facts_revision": int(state.get("facts_revision", 0)),
            "visited_nodes": ["build_summary"],
        }

    @staticmethod
    def request_approval(state: StoryWorkerState) -> StoryWorkerState:
        summary = StorySummary.model_validate(state["current_summary"])
        reply = f"{summary.summary_text}\n\n{summary.confirmation_question}"
        assistant_message = {
            "id": f"assistant-summary-{state['summary_version']}",
            "role": "assistant",
            "content": reply,
        }
        return {
            "messages": [assistant_message],
            "workflow_stage": "awaiting-approval",
            "approval_pending": True,
            "approved_summary_version": None,
            "reply": reply,
            "requested_fields": ["approval"],
            "visited_nodes": ["request_approval"],
        }

    def classify_approval(self, state: StoryWorkerState) -> StoryWorkerState:
        summary = StorySummary.model_validate(state.get("current_summary") or {})
        decision = self.model.classify_approval(
            message=state["incoming_message"],
            summary=summary,
            messages=state.get("messages", []),
        )
        return {
            "approval_decision": decision.model_dump(mode="json"),
            "visited_nodes": ["classify_approval"],
        }

    @staticmethod
    def route_approval(state: StoryWorkerState) -> str:
        decision = ApprovalDecision.model_validate(state["approval_decision"])
        return {
            "approve": "approval_guard",
            "revise": "understand_turn",
            "reject": "reject_generation",
            "ambiguous": "clarify_approval",
        }[decision.decision]

    @staticmethod
    def approval_guard(state: StoryWorkerState) -> StoryWorkerState:
        decision = ApprovalDecision.model_validate(state["approval_decision"])
        current_version = int(state.get("summary_version", 0))
        if decision.decision != "approve":
            raise ValueError("Only an explicit approval may pass the approval guard")
        if not state.get("approval_pending") or not state.get("current_summary"):
            raise ValueError("Approval requires a pending summary")
        if state.get("summary_facts_revision") != state.get("facts_revision"):
            raise ValueError("Approval summary is stale after fact changes")
        if current_version <= 0:
            raise ValueError("Approval requires a versioned summary")
        return {
            "workflow_stage": "approved",
            "approval_pending": False,
            "approved_summary_version": current_version,
            "reply": "승인된 정보로 스토리 생성을 시작합니다.",
            "requested_fields": [],
            "visited_nodes": ["approval_guard"],
        }

    @staticmethod
    def clarify_approval(state: StoryWorkerState) -> StoryWorkerState:
        reply = "현재 요약 내용으로 스토리를 생성할지 명확히 알려주세요."
        return {
            "messages": [{
                "id": f"assistant-approval-clarification-{len(state.get('messages', [])) + 1}",
                "role": "assistant",
                "content": reply,
            }],
            "workflow_stage": "awaiting-approval",
            "approval_pending": True,
            "reply": reply,
            "requested_fields": ["approval"],
            "visited_nodes": ["clarify_approval"],
        }

    @staticmethod
    def reject_generation(state: StoryWorkerState) -> StoryWorkerState:
        reply = "생성을 시작하지 않았습니다. 수정하거나 보완할 내용을 알려주세요."
        return {
            "messages": [{
                "id": f"assistant-rejection-{len(state.get('messages', [])) + 1}",
                "role": "assistant",
                "content": reply,
            }],
            "workflow_stage": "collecting",
            "approval_pending": False,
            "approved_summary_version": None,
            "reply": reply,
            "requested_fields": [],
            "visited_nodes": ["reject_generation"],
        }

    @staticmethod
    def cancel_session(state: StoryWorkerState) -> StoryWorkerState:
        reply = "현재 스토리 작성 세션을 종료했습니다."
        return {
            "messages": [{
                "id": f"assistant-cancel-{len(state.get('messages', [])) + 1}",
                "role": "assistant",
                "content": reply,
            }],
            "workflow_stage": "cancelled",
            "approval_pending": False,
            "reply": reply,
            "requested_fields": [],
            "visited_nodes": ["cancel_session"],
        }

    @staticmethod
    def closed_session(state: StoryWorkerState) -> StoryWorkerState:
        reply = "이미 종료된 세션입니다. 새 thread_id로 시작해 주세요."
        return {
            "messages": [{
                "id": f"assistant-closed-{len(state.get('messages', [])) + 1}",
                "role": "assistant",
                "content": reply,
            }],
            "reply": reply,
            "requested_fields": [],
            "visited_nodes": ["closed_session"],
        }


def build_conversation_graph(model: ConversationModel, *, checkpointer: Any):
    nodes = ConversationNodes(model)
    builder = StateGraph(StoryWorkerState)
    builder.add_node("understand_turn", nodes.understand_turn)
    builder.add_node("apply_fact_patch", nodes.apply_fact_patch)
    builder.add_node("check_required_fields", nodes.check_required_fields)
    builder.add_node("plan_next_questions", nodes.plan_next_questions)
    builder.add_node("emit_question", nodes.emit_question)
    builder.add_node("build_summary", nodes.build_summary)
    builder.add_node("request_approval", nodes.request_approval)
    builder.add_node("classify_approval", nodes.classify_approval)
    builder.add_node("approval_guard", nodes.approval_guard)
    builder.add_node("clarify_approval", nodes.clarify_approval)
    builder.add_node("reject_generation", nodes.reject_generation)
    builder.add_node("cancel_session", nodes.cancel_session)
    builder.add_node("closed_session", nodes.closed_session)

    builder.add_conditional_edges(
        START,
        nodes.route_start,
        {
            "understand_turn": "understand_turn",
            "classify_approval": "classify_approval",
            "closed_session": "closed_session",
        },
    )
    builder.add_conditional_edges(
        "understand_turn",
        nodes.route_understanding,
        {
            "apply_fact_patch": "apply_fact_patch",
            "cancel_session": "cancel_session",
        },
    )
    builder.add_edge("apply_fact_patch", "check_required_fields")
    builder.add_conditional_edges(
        "check_required_fields",
        nodes.route_after_required_check,
        {
            "plan_next_questions": "plan_next_questions",
            "build_summary": "build_summary",
        },
    )
    builder.add_edge("plan_next_questions", "emit_question")
    builder.add_edge("build_summary", "request_approval")
    builder.add_conditional_edges(
        "classify_approval",
        nodes.route_approval,
        {
            "approval_guard": "approval_guard",
            "understand_turn": "understand_turn",
            "reject_generation": "reject_generation",
            "clarify_approval": "clarify_approval",
        },
    )
    for node in (
        "emit_question",
        "request_approval",
        "approval_guard",
        "clarify_approval",
        "reject_generation",
        "cancel_session",
        "closed_session",
    ):
        builder.add_edge(node, END)
    return builder.compile(checkpointer=checkpointer)
