from __future__ import annotations

import json
import operator
import re
from copy import deepcopy
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    "risk_response",
]
FACT_FIELDS: tuple[FactField, ...] = (
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
)
REQUIRED_FACT_FIELDS: tuple[FactField, ...] = (
    "product_name",
    "product_type",
    "category",
    "key_strengths",
    "target_supporters",
)
REQUIRED_QUESTION_GROUPS: tuple[tuple[FactField, ...], ...] = (
    ("product_name", "product_type", "category"),
    ("key_strengths", "target_supporters"),
)

OptionalGroup = Literal[
    "story_persuasion",
    "funding_configuration",
    "policy",
    "project_explanation",
]
OPTIONAL_FIELD_GROUPS: dict[OptionalGroup, tuple[FactField, ...]] = {
    "story_persuasion": (
        "problem_context",
        "trust_elements",
        "maker_team_intro",
    ),
    "funding_configuration": ("rewards", "funding_end", "shipping_start"),
    "policy": ("refund_policy", "as_policy"),
    "project_explanation": ("funding_plan", "risk_response"),
}
OPTIONAL_FACT_FIELDS: tuple[FactField, ...] = tuple(
    field for fields in OPTIONAL_FIELD_GROUPS.values() for field in fields
)
OPTIONAL_GROUP_LABELS: dict[OptionalGroup, str] = {
    "story_persuasion": "스토리 설득 정보",
    "funding_configuration": "펀딩 구성",
    "policy": "정책",
    "project_explanation": "프로젝트 설명",
}
FACT_LABELS: dict[FactField, str] = {
    "product_name": "제품명",
    "product_type": "제품 유형",
    "category": "펀딩 카테고리",
    "key_strengths": "핵심 강점",
    "target_supporters": "핵심 서포터",
    "problem_context": "해결하려는 문제",
    "trust_elements": "신뢰 근거",
    "maker_team_intro": "메이커·팀 소개",
    "rewards": "리워드 구성",
    "funding_end": "펀딩 종료 일정",
    "shipping_start": "발송 시작 일정",
    "refund_policy": "교환·환불 정책",
    "as_policy": "보증·사후지원 정책",
    "funding_plan": "펀딩금 사용 계획",
    "risk_response": "위험과 대응 계획",
}

FactStatus = Literal["provided", "explicitly-absent", "unknown"]
FactOperation = Literal["replace", "append", "mark_absent", "clear"]
CollectionStatus = Literal["not-offered", "offered", "requested", "resolved", "skipped"]
CollectionAction = Literal[
    "continue_recommended",
    "select_groups",
    "select_fields",
    "skip_fields",
    "skip_all_optional",
    "return_to_optional",
    "none",
]
QuestionPurpose = Literal["required", "optional-collect", "clarify", "confirm-skip"]
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
    "generation-ready",
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


class CollectionDirective(StrictModel):
    action: CollectionAction = "none"
    groups: list[OptionalGroup] = Field(default_factory=list)
    fields: list[FactField] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_question: str | None = None

    @field_validator("groups", "fields")
    @classmethod
    def unique_values(cls, values: list[Any]) -> list[Any]:
        return list(dict.fromkeys(values))

    @model_validator(mode="after")
    def validate_directive(self) -> CollectionDirective:
        invalid_fields = set(self.fields) - set(OPTIONAL_FACT_FIELDS)
        if invalid_fields:
            raise ValueError(f"Collection directive accepts only optional fields: {invalid_fields}")
        if self.action == "select_groups" and not self.groups:
            raise ValueError("select_groups requires at least one group")
        if self.action in {"select_fields", "skip_fields"} and not self.fields:
            raise ValueError(f"{self.action} requires at least one field")
        if self.action == "select_groups" and self.fields:
            raise ValueError("select_groups does not accept fields")
        if self.action in {"select_fields", "skip_fields"} and self.groups:
            raise ValueError(f"{self.action} does not accept groups")
        if self.action in {
            "continue_recommended",
            "skip_all_optional",
        } and (self.groups or self.fields):
            raise ValueError(f"{self.action} does not accept groups or fields")
        if (
            self.action == "none"
            and not self.requires_clarification
            and (self.groups or self.fields)
        ):
            raise ValueError("none accepts groups or fields only for clarification targets")
        if self.requires_clarification and not (
            self.clarification_question and self.clarification_question.strip()
        ):
            raise ValueError("clarification_question is required when clarification is needed")
        return self


class TurnUnderstanding(StrictModel):
    intent: TurnIntent
    fact_patches: list[FactPatch] = Field(default_factory=list)
    collection_directive: CollectionDirective = Field(default_factory=CollectionDirective)
    unresolved_references: list[str] = Field(default_factory=list)
    clarification_fields: list[FactField] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_question: str | None = None

    @model_validator(mode="after")
    def validate_clarification(self) -> TurnUnderstanding:
        patch_fields = [patch.field for patch in self.fact_patches]
        if len(patch_fields) != len(set(patch_fields)):
            raise ValueError("fact_patches must contain at most one patch per field")
        if self.requires_clarification and not (
            self.clarification_question and self.clarification_question.strip()
        ):
            raise ValueError("clarification_question is required when clarification is needed")
        return self


_AMBIGUOUS_REFERENCE = re.compile(r"그건|그걸|그거|앞의 값|이전 값")
_AMBIGUOUS_COLLECTION_REPLY = re.compile(
    r"(?:알아서\s*(?:해|진행해)\s*줘|응[,\s]*좋아|그냥\s*진행해\s*줘|다음으로\s*가자)"
)
_EXPLICIT_SKIP_ALL_OPTIONAL = re.compile(
    r"(?:남은\s*)?선택\s*정보(?:는|를)?\s*(?:전체|전부|모두)(?:를)?\s*"
    r"(?:생략|건너뛰)(?:할게|할래|하자|해줘|해주세요|하겠습니다|하겠어|자)?"
)
_DATE_LIKE_VALUE = re.compile(r"(?:\d{4}\s*년|\d{1,2}\s*월|\d{1,2}\s*일|\d{4}[-./]\d{1,2})")
_EXPLICIT_FIELD_CUES = (
    "제품명",
    "제품 이름",
    "제품 종류",
    "제품 유형",
    "카테고리",
    "강점",
    "타깃",
    "서포터",
    "문제",
    "신뢰",
    "인증",
    "메이커",
    "팀 소개",
    "리워드",
    "종료일",
    "종료 일정",
    "발송",
    "환불",
    "A/S",
    "수리",
    "펀딩금",
    "위험",
    "대응",
)


def _only_competitor_type_mention(message: str, value: str) -> bool:
    matches = list(re.finditer(re.escape(value), message, flags=re.IGNORECASE))
    if not matches:
        return False
    return all(
        re.search(
            r"(?:기존|경쟁사|경쟁 제품|다른)\s*$",
            message[max(0, match.start() - 12) : match.start()],
        )
        is not None
        for match in matches
    )


def reconcile_turn_understanding(
    message: str, understanding: TurnUnderstanding
) -> TurnUnderstanding:
    """Apply deterministic grounding guards without inventing new facts."""
    patches = list(understanding.fact_patches)
    directive = understanding.collection_directive
    normalized_message = message.strip().rstrip(".!?").strip()
    if (
        directive.action == "none"
        and not directive.requires_clarification
        and _AMBIGUOUS_COLLECTION_REPLY.fullmatch(normalized_message)
    ):
        directive = directive.model_copy(
            update={
                "requires_clarification": True,
                "clarification_question": (
                    "선택 정보를 계속 입력할지, 특정 항목을 생략할지 명확히 알려주세요."
                ),
            }
        )
    if _EXPLICIT_SKIP_ALL_OPTIONAL.fullmatch(normalized_message):
        directive = CollectionDirective(action="skip_all_optional")
    if directive.action == "skip_fields":
        skipped = set(directive.fields)
        patches = [patch for patch in patches if patch.field not in skipped]

    patches = [
        patch
        for patch in patches
        if not (
            patch.field == "product_type"
            and patch.values
            and all(_only_competitor_type_mention(message, value) for value in patch.values)
        )
    ]

    reference = _AMBIGUOUS_REFERENCE.search(message)
    explicit_field = any(cue.casefold() in message.casefold() for cue in _EXPLICIT_FIELD_CUES)
    if (
        understanding.intent == "request_generation"
        and not patches
        and not understanding.unresolved_references
        and directive.action == "none"
        and not directive.requires_clarification
        and not explicit_field
        and _DATE_LIKE_VALUE.search(message) is None
    ):
        understanding = understanding.model_copy(
            update={
                "clarification_fields": [],
                "requires_clarification": False,
                "clarification_question": None,
            }
        )
    if reference is not None and patches and not explicit_field:
        patch_fields = list(dict.fromkeys(patch.field for patch in patches))
        unresolved = list(dict.fromkeys([*understanding.unresolved_references, reference.group(0)]))
        return understanding.model_copy(
            update={
                "fact_patches": [],
                "collection_directive": directive,
                "unresolved_references": unresolved,
                "clarification_fields": patch_fields,
                "requires_clarification": True,
                "clarification_question": (
                    "어떤 정보 항목을 수정하려는지 항목 이름과 값을 함께 알려주세요."
                ),
            }
        )
    return understanding.model_copy(
        update={"fact_patches": patches, "collection_directive": directive}
    )


class QuestionPlan(StrictModel):
    purpose: QuestionPurpose
    requested_fields: list[FactField]
    requested_group: OptionalGroup | None = None
    requested_detail: str = Field(min_length=1)
    question: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @field_validator("requested_fields")
    @classmethod
    def unique_fields(cls, fields: list[FactField]) -> list[FactField]:
        return list(dict.fromkeys(fields))


class StorySummary(StrictModel):
    headline: str = Field(min_length=1)
    confirmed_facts: dict[FactField, list[str]]
    explicitly_absent_fields: list[FactField]
    skipped_fields: list[FactField]
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


class OptionalCollection(StrictModel):
    offered: bool = False
    selected_groups: list[OptionalGroup] = Field(default_factory=list)
    active_group: OptionalGroup | None = None
    pending_fields: list[FactField] = Field(default_factory=list)
    field_states: dict[FactField, CollectionStatus]
    skip_confirmation_pending: list[FactField] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_optional_fields(self) -> OptionalCollection:
        if set(self.field_states) != set(OPTIONAL_FACT_FIELDS):
            raise ValueError("field_states must contain every optional field exactly once")
        if set(self.pending_fields) - set(OPTIONAL_FACT_FIELDS):
            raise ValueError("pending_fields accepts only optional fields")
        if set(self.skip_confirmation_pending) - set(OPTIONAL_FACT_FIELDS):
            raise ValueError("skip_confirmation_pending accepts only optional fields")
        return self


class QuestionRecord(StrictModel):
    fields: list[FactField]
    purpose: QuestionPurpose
    requested_group: OptionalGroup | None = None
    requested_detail: str
    facts_revision: int
    collection_revision: int = 0
    progress_fingerprint: str = ""
    signature: str = ""
    resolved: bool = False


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
    collection_revision: int
    missing_required_fields: list[FactField]
    optional_collection: dict[str, Any]
    question_history: list[dict[str, Any]]
    question_purpose: QuestionPurpose
    question_candidate_fields: list[FactField]
    question_group: OptionalGroup | None
    question_requested_detail: str
    current_question_plan: dict[str, Any] | None
    question_plan_error: str | None
    question_no_progress: bool
    turn_understanding: dict[str, Any] | None
    current_summary: dict[str, Any] | None
    summary_version: int
    summary_facts_revision: int | None
    summary_collection_revision: int | None
    approval_pending: bool
    approved_summary_version: int | None
    approval_decision: dict[str, Any] | None
    workflow_stage: WorkflowStage
    collection_phase: str
    reply: str
    requested_fields: list[str]
    optional_choice_fingerprint: str
    optional_choice_offer_count: int


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
        purpose: QuestionPurpose,
        candidate_fields: list[FactField],
        requested_group: OptionalGroup | None,
        requested_detail: str,
        question_history: list[dict[str, Any]],
        turn_understanding: TurnUnderstanding,
    ) -> QuestionPlan: ...

    def repair_question_plan(
        self,
        *,
        invalid_plan: QuestionPlan | None,
        validation_error: str,
        messages: list[dict[str, str]],
        facts: dict[str, dict[str, Any]],
        purpose: QuestionPurpose,
        candidate_fields: list[FactField],
        requested_group: OptionalGroup | None,
        requested_detail: str,
    ) -> QuestionPlan: ...

    def build_summary(
        self,
        *,
        messages: list[dict[str, str]],
        facts: dict[str, dict[str, Any]],
        optional_collection: dict[str, Any],
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


def explicitly_absent_fields(facts: dict[str, dict[str, Any]]) -> list[FactField]:
    return [
        field
        for field in FACT_FIELDS
        if FactValue.model_validate(facts.get(field, FactValue().model_dump())).status
        == "explicitly-absent"
    ]


def missing_required_fields(facts: dict[str, dict[str, Any]]) -> list[FactField]:
    return [
        field
        for field in REQUIRED_FACT_FIELDS
        if FactValue.model_validate(facts.get(field, FactValue().model_dump())).status != "provided"
    ]


def optional_group_for(field: FactField) -> OptionalGroup | None:
    for group, fields in OPTIONAL_FIELD_GROUPS.items():
        if field in fields:
            return group
    return None


def initial_optional_collection(
    facts: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    current_facts = facts or initial_facts()
    field_states: dict[FactField, CollectionStatus] = {}
    for field in OPTIONAL_FACT_FIELDS:
        fact = FactValue.model_validate(current_facts[field])
        field_states[field] = "resolved" if fact.status != "unknown" else "not-offered"
    return OptionalCollection(field_states=field_states).model_dump(mode="json")


def sync_optional_collection(
    collection_data: dict[str, Any] | None,
    facts: dict[str, dict[str, Any]],
) -> OptionalCollection:
    collection = OptionalCollection.model_validate(
        collection_data or initial_optional_collection(facts)
    )
    states = dict(collection.field_states)
    for field in OPTIONAL_FACT_FIELDS:
        fact = FactValue.model_validate(facts[field])
        if fact.status != "unknown":
            states[field] = "resolved"
        elif states[field] == "resolved":
            states[field] = "offered" if collection.offered else "not-offered"
    pending = [
        field for field in collection.pending_fields if states[field] not in {"resolved", "skipped"}
    ]
    active_group = optional_group_for(pending[0]) if pending else None
    return collection.model_copy(
        update={
            "field_states": states,
            "pending_fields": pending,
            "active_group": active_group,
        }
    )


def unresolved_optional_fields(collection_data: dict[str, Any]) -> list[FactField]:
    collection = OptionalCollection.model_validate(collection_data)
    return [
        field
        for field in OPTIONAL_FACT_FIELDS
        if collection.field_states[field] not in {"resolved", "skipped"}
    ]


def skipped_optional_fields(collection_data: dict[str, Any]) -> list[FactField]:
    collection = OptionalCollection.model_validate(collection_data)
    return [field for field in OPTIONAL_FACT_FIELDS if collection.field_states[field] == "skipped"]


def collection_ready(collection_data: dict[str, Any]) -> bool:
    collection = OptionalCollection.model_validate(collection_data)
    all_resolved_without_offer = all(
        collection.field_states[field] == "resolved" for field in OPTIONAL_FACT_FIELDS
    )
    return (collection.offered or all_resolved_without_offer) and not unresolved_optional_fields(
        collection_data
    )


def _resolved_question_history(
    history_data: list[dict[str, Any]],
    facts: dict[str, dict[str, Any]],
    collection: OptionalCollection,
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for item in history_data:
        record = QuestionRecord.model_validate(item)
        resolved = bool(record.fields) and all(
            FactValue.model_validate(facts[field]).status != "unknown"
            or (field in OPTIONAL_FACT_FIELDS and collection.field_states[field] == "skipped")
            for field in record.fields
        )
        updated.append(record.model_copy(update={"resolved": resolved}).model_dump(mode="json"))
    return updated


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

    previous_collection = state.get("optional_collection")
    collection = sync_optional_collection(previous_collection, facts)
    collection_changed = previous_collection is not None and (
        collection.model_dump(mode="json") != previous_collection
    )
    question_history = _resolved_question_history(
        list(state.get("question_history", [])), facts, collection
    )
    update: StoryWorkerState = {
        "facts": facts,
        "fact_history": history,
        "facts_revision": int(state.get("facts_revision", 0)) + int(changed),
        "collection_revision": int(state.get("collection_revision", 0)) + int(collection_changed),
        "optional_collection": collection.model_dump(mode="json"),
        "question_history": question_history,
        "visited_nodes": ["apply_fact_patch"],
    }
    if changed or collection_changed:
        update.update(
            {
                "current_summary": None,
                "summary_facts_revision": None,
                "summary_collection_revision": None,
                "approval_pending": False,
                "approved_summary_version": None,
            }
        )
    return update


def apply_collection_directive(state: StoryWorkerState) -> StoryWorkerState:
    understanding = TurnUnderstanding.model_validate(state["turn_understanding"])
    directive = understanding.collection_directive
    collection = sync_optional_collection(state.get("optional_collection"), state["facts"])
    states = dict(collection.field_states)
    selected_groups = list(collection.selected_groups)
    pending = list(collection.pending_fields)
    skip_pending: list[FactField] = []
    offered = collection.offered

    if directive.requires_clarification:
        skip_pending = list(directive.fields)
    elif directive.action != "none":
        offered = True
        if directive.action == "continue_recommended":
            selected_groups = list(OPTIONAL_FIELD_GROUPS)
            pending = list(OPTIONAL_FACT_FIELDS)
            for field in pending:
                if states[field] == "skipped":
                    states[field] = "offered"
        elif directive.action == "select_groups":
            selected_groups = list(directive.groups)
            pending = [field for group in selected_groups for field in OPTIONAL_FIELD_GROUPS[group]]
            for field in pending:
                if states[field] == "skipped":
                    states[field] = "offered"
        elif directive.action == "select_fields":
            selected_groups = list(
                dict.fromkeys(
                    group
                    for field in directive.fields
                    if (group := optional_group_for(field)) is not None
                )
            )
            pending = list(directive.fields)
            for field in pending:
                if states[field] == "skipped":
                    states[field] = "offered"
        elif directive.action == "skip_fields":
            for field in directive.fields:
                if states[field] != "resolved":
                    states[field] = "skipped"
            pending = [field for field in pending if field not in directive.fields]
        elif directive.action == "skip_all_optional":
            for field in OPTIONAL_FACT_FIELDS:
                if states[field] != "resolved":
                    states[field] = "skipped"
            pending = []
            selected_groups = []
        elif directive.action == "return_to_optional":
            pending = list(directive.fields) or [
                field for field in OPTIONAL_FACT_FIELDS if states[field] == "skipped"
            ]
            for field in pending:
                if states[field] == "skipped":
                    states[field] = "offered"
            selected_groups = list(
                dict.fromkeys(
                    group for field in pending if (group := optional_group_for(field)) is not None
                )
            )

    pending = [
        field for field in dict.fromkeys(pending) if states[field] not in {"resolved", "skipped"}
    ]
    for field in pending:
        states[field] = "requested"
    active_group = optional_group_for(pending[0]) if pending else None
    updated = OptionalCollection(
        offered=offered,
        selected_groups=selected_groups,
        active_group=active_group,
        pending_fields=pending,
        field_states=states,
        skip_confirmation_pending=skip_pending,
    )
    previous_dump = collection.model_dump(mode="json")
    updated_dump = updated.model_dump(mode="json")
    changed = updated_dump != previous_dump
    question_history = _resolved_question_history(
        list(state.get("question_history", [])), state["facts"], updated
    )
    result: StoryWorkerState = {
        "optional_collection": updated_dump,
        "collection_revision": int(state.get("collection_revision", 0)) + int(changed),
        "question_history": question_history,
        "visited_nodes": ["apply_collection_directive"],
    }
    if changed:
        result.update(
            {
                "current_summary": None,
                "summary_collection_revision": None,
                "approval_pending": False,
                "approved_summary_version": None,
            }
        )
    return result


def validate_summary_grounding(
    summary: StorySummary,
    facts: dict[str, dict[str, Any]],
    collection_data: dict[str, Any],
) -> None:
    expected_facts = provided_facts(facts)
    actual_facts = {str(field): values for field, values in summary.confirmed_facts.items()}
    if actual_facts != expected_facts:
        raise ValueError("Summary confirmed_facts must exactly match current provided facts")
    if list(summary.explicitly_absent_fields) != explicitly_absent_fields(facts):
        raise ValueError("Summary explicitly_absent_fields must match current facts")
    if list(summary.skipped_fields) != skipped_optional_fields(collection_data):
        raise ValueError("Summary skipped_fields must match current collection state")

    allowed_text = " ".join(value for values in expected_facts.values() for value in values)
    unsupported_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", summary.summary_text)) - set(
        re.findall(r"\d+(?:[.,]\d+)?", allowed_text)
    )
    if unsupported_numbers:
        raise ValueError(
            f"Summary introduced unsupported numeric claims: {sorted(unsupported_numbers)}"
        )


def _remaining_group_lines(collection: OptionalCollection) -> list[str]:
    lines: list[str] = []
    for group, fields in OPTIONAL_FIELD_GROUPS.items():
        remaining = [
            field
            for field in fields
            if collection.field_states[field] not in {"resolved", "skipped"}
        ]
        if remaining:
            labels = ", ".join(FACT_LABELS[field] for field in remaining)
            lines.append(f"- {OPTIONAL_GROUP_LABELS[group]}: {labels}")
    return lines


def _fallback_question(
    *,
    purpose: QuestionPurpose,
    candidate_fields: list[FactField],
    requested_group: OptionalGroup | None,
    requested_detail: str,
) -> QuestionPlan:
    fields = [field for field in candidate_fields if field in FACT_FIELDS][:3]
    if purpose in {"clarify", "confirm-skip"} and not fields:
        question = requested_detail
    else:
        labels = " · ".join(FACT_LABELS[field] for field in fields)
        question = f"다음 정보를 각각 알려주세요: {labels}."
    return QuestionPlan(
        purpose=purpose,
        requested_fields=fields,
        requested_group=requested_group,
        requested_detail=requested_detail,
        question=question,
        rationale="유효한 질문 계획을 보장하는 결정적 대체 질문입니다.",
    )


def progress_fingerprint(state: StoryWorkerState) -> str:
    return ":".join(
        [
            str(int(state.get("facts_revision", 0))),
            str(int(state.get("collection_revision", 0))),
            str(int(state.get("summary_version", 0))),
            str(state.get("workflow_stage", "collecting")),
        ]
    )


def question_signature(plan: QuestionPlan) -> str:
    return json.dumps(
        {
            "purpose": plan.purpose,
            "fields": sorted(plan.requested_fields),
            "group": plan.requested_group,
            "detail": " ".join(plan.requested_detail.casefold().split()),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def no_progress_repeat_count(state: StoryWorkerState, plan: QuestionPlan) -> int:
    signature = question_signature(plan)
    fingerprint = progress_fingerprint(state)
    count = 0
    for item in state.get("question_history", []):
        record = QuestionRecord.model_validate(item)
        record_signature = record.signature or json.dumps(
            {
                "purpose": record.purpose,
                "fields": sorted(record.fields),
                "group": record.requested_group,
                "detail": " ".join(record.requested_detail.casefold().split()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        record_fingerprint = record.progress_fingerprint or ":".join(
            [
                str(record.facts_revision),
                str(record.collection_revision),
                "0",
                "collecting",
            ]
        )
        if record_signature == signature and record_fingerprint == fingerprint:
            count += 1
    return count


def question_plan_error(state: StoryWorkerState, plan: QuestionPlan) -> str | None:
    purpose = state["question_purpose"]
    candidates = list(state.get("question_candidate_fields", []))
    candidate_set = set(candidates)
    if plan.purpose != purpose:
        return "question purpose does not match the planned purpose"
    if plan.requested_detail != state.get("question_requested_detail"):
        return "question detail does not match the planned detail"
    if plan.requested_group != state.get("question_group"):
        return "question group does not match the planned group"
    if len(plan.requested_fields) > 3:
        return "a question may request at most three fields"
    if purpose in {"required", "optional-collect"} and not plan.requested_fields:
        return "a collection question must request at least one field"
    if not set(plan.requested_fields).issubset(candidate_set):
        return "question requested fields outside the candidate set"
    expected_fields = candidates[:3]
    if set(plan.requested_fields) != set(expected_fields):
        return "question must cover every planned candidate field up to three"
    if any(field in plan.question for field in FACT_FIELDS):
        return "question exposed an internal fact field name"
    if purpose == "optional-collect":
        group = state.get("question_group")
        if group and any(optional_group_for(field) != group for field in plan.requested_fields):
            return "optional question crossed a group boundary"
    current_revision = int(state.get("facts_revision", 0))
    current_collection_revision = int(state.get("collection_revision", 0))
    for item in state.get("question_history", []):
        record = QuestionRecord.model_validate(item)
        if (
            record.resolved
            and record.purpose == plan.purpose
            and set(record.fields) == set(plan.requested_fields)
            and record.requested_detail == plan.requested_detail
            and record.facts_revision == current_revision
            and record.collection_revision == current_collection_revision
        ):
            return "question semantically repeats a resolved request in the same revision"
    return None


class ConversationNodes:
    def __init__(self, model: ConversationModel) -> None:
        self.model = model

    @staticmethod
    def route_start(state: StoryWorkerState) -> str:
        if state.get("workflow_stage") in {"generation-ready", "cancelled"}:
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
    def apply_collection_directive(state: StoryWorkerState) -> StoryWorkerState:
        return apply_collection_directive(state)

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
        directive = understanding.collection_directive
        if understanding.requires_clarification:
            return "prepare_clarification_question"
        if directive.requires_clarification:
            return "prepare_skip_confirmation"
        if state.get("missing_required_fields"):
            return "prepare_required_question"
        collection = OptionalCollection.model_validate(state["optional_collection"])
        if collection_ready(collection.model_dump(mode="json")):
            return "build_summary"
        if not collection.offered:
            return "offer_optional_information"
        if collection.pending_fields:
            return "prepare_optional_question"
        return "offer_optional_choices"

    @staticmethod
    def prepare_required_question(state: StoryWorkerState) -> StoryWorkerState:
        missing = set(state.get("missing_required_fields", []))
        candidates = next(
            (
                fields
                for group in REQUIRED_QUESTION_GROUPS
                if (fields := [field for field in group if field in missing])
            ),
            [],
        )
        return {
            "question_purpose": "required",
            "question_candidate_fields": candidates,
            "question_group": None,
            "question_requested_detail": "부족한 필수 제품 정보를 확인합니다.",
            "visited_nodes": ["prepare_required_question"],
        }

    @staticmethod
    def prepare_clarification_question(state: StoryWorkerState) -> StoryWorkerState:
        understanding = TurnUnderstanding.model_validate(state["turn_understanding"])
        return {
            "question_purpose": "clarify",
            "question_candidate_fields": list(understanding.clarification_fields),
            "question_group": None,
            "question_requested_detail": understanding.clarification_question
            or "어떤 정보를 뜻하는지 구체적으로 알려주세요.",
            "visited_nodes": ["prepare_clarification_question"],
        }

    @staticmethod
    def prepare_skip_confirmation(state: StoryWorkerState) -> StoryWorkerState:
        understanding = TurnUnderstanding.model_validate(state["turn_understanding"])
        directive = understanding.collection_directive
        fields = list(directive.fields)
        labels = ", ".join(FACT_LABELS[field] for field in fields)
        detail = directive.clarification_question or (
            f"{labels or '남은 선택 정보'}를 이번 생성에서 생략할지 명확히 알려주세요."
        )
        return {
            "question_purpose": "confirm-skip",
            "question_candidate_fields": fields,
            "question_group": None,
            "question_requested_detail": detail,
            "visited_nodes": ["prepare_skip_confirmation"],
        }

    @staticmethod
    def prepare_optional_question(state: StoryWorkerState) -> StoryWorkerState:
        collection = OptionalCollection.model_validate(state["optional_collection"])
        group = optional_group_for(collection.pending_fields[0])
        if group is None:
            raise ValueError("pending optional field does not belong to a collection group")
        candidates = [
            field for field in collection.pending_fields if optional_group_for(field) == group
        ][:3]
        return {
            "question_purpose": "optional-collect",
            "question_candidate_fields": candidates,
            "question_group": group,
            "question_requested_detail": f"{OPTIONAL_GROUP_LABELS[group]}를 보완합니다.",
            "visited_nodes": ["prepare_optional_question"],
        }

    def plan_next_questions(self, state: StoryWorkerState) -> StoryWorkerState:
        understanding = TurnUnderstanding.model_validate(state["turn_understanding"])
        try:
            plan = self.model.plan_questions(
                messages=state.get("messages", []),
                facts=state["facts"],
                purpose=state["question_purpose"],
                candidate_fields=list(state.get("question_candidate_fields", [])),
                requested_group=state.get("question_group"),
                requested_detail=state["question_requested_detail"],
                question_history=list(state.get("question_history", [])),
                turn_understanding=understanding,
            )
            error = question_plan_error(state, plan)
            no_progress = error is None and no_progress_repeat_count(state, plan) >= 2
        except Exception as exc:
            plan = None
            error = f"question planner failed: {exc}"
            no_progress = False
        return {
            "current_question_plan": (plan.model_dump(mode="json") if plan is not None else None),
            "question_plan_error": error,
            "question_no_progress": no_progress,
            "visited_nodes": ["plan_next_questions"],
        }

    @staticmethod
    def route_question_plan(state: StoryWorkerState) -> str:
        if state.get("question_no_progress"):
            return "handle_no_progress"
        return "repair_question_plan" if state.get("question_plan_error") else "emit_question"

    def repair_question_plan(self, state: StoryWorkerState) -> StoryWorkerState:
        invalid = (
            QuestionPlan.model_validate(state["current_question_plan"])
            if state.get("current_question_plan")
            else None
        )
        try:
            repaired = self.model.repair_question_plan(
                invalid_plan=invalid,
                validation_error=str(state.get("question_plan_error")),
                messages=state.get("messages", []),
                facts=state["facts"],
                purpose=state["question_purpose"],
                candidate_fields=list(state.get("question_candidate_fields", [])),
                requested_group=state.get("question_group"),
                requested_detail=state["question_requested_detail"],
            )
            if question_plan_error(state, repaired):
                raise ValueError("repaired question plan is still invalid")
        except Exception:
            repaired = _fallback_question(
                purpose=state["question_purpose"],
                candidate_fields=list(state.get("question_candidate_fields", [])),
                requested_group=state.get("question_group"),
                requested_detail=state["question_requested_detail"],
            )
        return {
            "current_question_plan": repaired.model_dump(mode="json"),
            "question_plan_error": None,
            "question_no_progress": False,
            "visited_nodes": ["repair_question_plan"],
        }

    @staticmethod
    def handle_no_progress(state: StoryWorkerState) -> StoryWorkerState:
        purpose = state["question_purpose"]
        candidates = list(state.get("question_candidate_fields", []))
        labels = ", ".join(FACT_LABELS[field] for field in candidates)
        if purpose == "required":
            reply = (
                f"현재 확인되지 않은 필수 정보는 {labels}입니다. 같은 질문은 여기서 중단합니다. "
                "이 정보가 준비되면 항목과 값을 함께 입력해 주세요."
            )
            phase = "required"
        elif purpose == "optional-collect":
            reply = (
                f"{labels or '선택 정보'} 단계에서 진행이 없어 같은 질문은 여기서 중단합니다. "
                "계속 입력할 항목과 값을 알려주거나, 남은 선택 정보를 생략한다고 명확히 "
                "입력해 주세요."
            )
            phase = "optional-offer"
        else:
            reply = (
                "지시 대상을 확정할 수 없어 현재 상태를 유지합니다. 같은 확인 질문은 여기서 "
                "중단하며, 수정하거나 생략할 항목과 의사를 구체적으로 입력해 주세요."
            )
            phase = str(state.get("collection_phase", "required"))
        return {
            "messages": [
                {
                    "id": f"assistant-no-progress-{len(state.get('messages', [])) + 1}",
                    "role": "assistant",
                    "content": reply,
                }
            ],
            "workflow_stage": "collecting",
            "collection_phase": phase,
            "reply": reply,
            "requested_fields": [],
            "approval_pending": False,
            "visited_nodes": ["handle_no_progress"],
        }

    @staticmethod
    def emit_question(state: StoryWorkerState) -> StoryWorkerState:
        plan = QuestionPlan.model_validate(state["current_question_plan"])
        history = list(state.get("question_history", []))
        history.append(
            QuestionRecord(
                fields=list(plan.requested_fields),
                purpose=plan.purpose,
                requested_group=plan.requested_group,
                requested_detail=plan.requested_detail,
                facts_revision=int(state.get("facts_revision", 0)),
                collection_revision=int(state.get("collection_revision", 0)),
                progress_fingerprint=progress_fingerprint(state),
                signature=question_signature(plan),
            ).model_dump(mode="json")
        )
        assistant_message = {
            "id": f"assistant-question-{len(state.get('messages', [])) + 1}",
            "role": "assistant",
            "content": plan.question,
        }
        if plan.purpose == "required":
            collection_phase = "required"
        elif plan.purpose == "optional-collect":
            collection_phase = "optional-collect"
        else:
            collection_phase = str(state.get("collection_phase", "required"))
        return {
            "messages": [assistant_message],
            "question_history": history,
            "workflow_stage": "collecting",
            "collection_phase": collection_phase,
            "reply": plan.question,
            "requested_fields": list(plan.requested_fields),
            "approval_pending": False,
            "visited_nodes": ["emit_question"],
        }

    @staticmethod
    def offer_optional_information(state: StoryWorkerState) -> StoryWorkerState:
        collection = OptionalCollection.model_validate(state["optional_collection"])
        states = dict(collection.field_states)
        for field in OPTIONAL_FACT_FIELDS:
            if states[field] == "not-offered":
                states[field] = "offered"
        collection = collection.model_copy(update={"offered": True, "field_states": states})
        lines = _remaining_group_lines(collection)
        reply = (
            "필수 정보는 모두 확인했습니다. 더 충실한 스토리를 위해 아래 선택 정보를 "
            "추가할 수 있습니다.\n\n"
            + "\n".join(lines)
            + "\n\n모두 입력, 특정 그룹·항목 선택, 권장 순서로 진행, "
            "전체 생략 중 하나를 알려주세요."
        )
        return {
            "optional_collection": collection.model_dump(mode="json"),
            "collection_revision": int(state.get("collection_revision", 0)) + 1,
            "messages": [
                {
                    "id": f"assistant-optional-offer-{len(state.get('messages', [])) + 1}",
                    "role": "assistant",
                    "content": reply,
                }
            ],
            "workflow_stage": "collecting",
            "collection_phase": "optional-offer",
            "reply": reply,
            "requested_fields": [],
            "visited_nodes": ["offer_optional_information"],
        }

    @staticmethod
    def offer_optional_choices(state: StoryWorkerState) -> StoryWorkerState:
        collection = OptionalCollection.model_validate(state["optional_collection"])
        lines = _remaining_group_lines(collection)
        fingerprint = progress_fingerprint(state)
        same_progress = state.get("optional_choice_fingerprint") == fingerprint
        offer_count = int(state.get("optional_choice_offer_count", 0)) + 1 if same_progress else 1
        if offer_count >= 2:
            reply = (
                "선택 정보 단계에서 진행이 없어 같은 안내는 여기서 중단합니다. 계속 입력할 "
                "그룹·항목과 값을 알려주거나, 남은 선택 정보를 모두 생략한다고 명확히 입력해 "
                "주세요."
            )
        else:
            reply = (
                "남은 선택 정보는 다음과 같습니다.\n\n"
                + "\n".join(lines)
                + "\n\n계속할 그룹·항목을 선택하거나, 남은 정보를 모두 생략한다고 명확히 "
                "알려주세요."
            )
        return {
            "messages": [
                {
                    "id": f"assistant-optional-choice-{len(state.get('messages', [])) + 1}",
                    "role": "assistant",
                    "content": reply,
                }
            ],
            "workflow_stage": "collecting",
            "collection_phase": "optional-offer",
            "reply": reply,
            "requested_fields": [],
            "optional_choice_fingerprint": fingerprint,
            "optional_choice_offer_count": offer_count,
            "visited_nodes": ["offer_optional_choices"],
        }

    def build_summary(self, state: StoryWorkerState) -> StoryWorkerState:
        if missing_required_fields(state["facts"]):
            raise ValueError("Summary requires all required facts")
        if not collection_ready(state["optional_collection"]):
            raise ValueError("Summary requires every optional field to be resolved or skipped")
        summary = self.model.build_summary(
            messages=state.get("messages", []),
            facts=state["facts"],
            optional_collection=state["optional_collection"],
        )
        validate_summary_grounding(summary, state["facts"], state["optional_collection"])
        return {
            "current_summary": summary.model_dump(mode="json"),
            "summary_version": int(state.get("summary_version", 0)) + 1,
            "summary_facts_revision": int(state.get("facts_revision", 0)),
            "summary_collection_revision": int(state.get("collection_revision", 0)),
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
            "collection_phase": "approval",
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
        if state.get("summary_collection_revision") != state.get("collection_revision"):
            raise ValueError("Approval summary is stale after collection changes")
        if not collection_ready(state["optional_collection"]):
            raise ValueError("Approval requires completed optional collection")
        if current_version <= 0:
            raise ValueError("Approval requires a versioned summary")
        return {
            "workflow_stage": "generation-ready",
            "collection_phase": "generation-ready",
            "approval_pending": False,
            "approved_summary_version": current_version,
            "reply": "승인된 정보로 스토리를 생성할 준비가 완료됐습니다.",
            "requested_fields": [],
            "visited_nodes": ["approval_guard"],
        }

    @staticmethod
    def clarify_approval(state: StoryWorkerState) -> StoryWorkerState:
        reply = "현재 요약 내용으로 스토리를 생성할지 명확히 알려주세요."
        return {
            "messages": [
                {
                    "id": f"assistant-approval-clarification-{len(state.get('messages', [])) + 1}",
                    "role": "assistant",
                    "content": reply,
                }
            ],
            "workflow_stage": "awaiting-approval",
            "collection_phase": "approval",
            "approval_pending": True,
            "reply": reply,
            "requested_fields": ["approval"],
            "visited_nodes": ["clarify_approval"],
        }

    @staticmethod
    def reject_generation(state: StoryWorkerState) -> StoryWorkerState:
        reply = "생성을 시작하지 않았습니다. 수정하거나 보완할 내용을 알려주세요."
        return {
            "messages": [
                {
                    "id": f"assistant-rejection-{len(state.get('messages', [])) + 1}",
                    "role": "assistant",
                    "content": reply,
                }
            ],
            "workflow_stage": "collecting",
            "collection_phase": "optional-offer",
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
            "messages": [
                {
                    "id": f"assistant-cancel-{len(state.get('messages', [])) + 1}",
                    "role": "assistant",
                    "content": reply,
                }
            ],
            "workflow_stage": "cancelled",
            "collection_phase": "closed",
            "approval_pending": False,
            "reply": reply,
            "requested_fields": [],
            "visited_nodes": ["cancel_session"],
        }

    @staticmethod
    def closed_session(state: StoryWorkerState) -> StoryWorkerState:
        reply = "이미 종료되었거나 생성 준비가 완료된 세션입니다. 새 thread_id로 시작해 주세요."
        return {
            "messages": [
                {
                    "id": f"assistant-closed-{len(state.get('messages', [])) + 1}",
                    "role": "assistant",
                    "content": reply,
                }
            ],
            "reply": reply,
            "requested_fields": [],
            "visited_nodes": ["closed_session"],
        }


def build_conversation_graph(model: ConversationModel, *, checkpointer: Any):
    nodes = ConversationNodes(model)
    builder = StateGraph(StoryWorkerState)
    builder.add_node("understand_turn", nodes.understand_turn)
    builder.add_node("apply_fact_patch", nodes.apply_fact_patch)
    builder.add_node("apply_collection_directive", nodes.apply_collection_directive)
    builder.add_node("check_required_fields", nodes.check_required_fields)
    builder.add_node("prepare_required_question", nodes.prepare_required_question)
    builder.add_node("prepare_clarification_question", nodes.prepare_clarification_question)
    builder.add_node("prepare_skip_confirmation", nodes.prepare_skip_confirmation)
    builder.add_node("prepare_optional_question", nodes.prepare_optional_question)
    builder.add_node("plan_next_questions", nodes.plan_next_questions)
    builder.add_node("repair_question_plan", nodes.repair_question_plan)
    builder.add_node("handle_no_progress", nodes.handle_no_progress)
    builder.add_node("emit_question", nodes.emit_question)
    builder.add_node("offer_optional_information", nodes.offer_optional_information)
    builder.add_node("offer_optional_choices", nodes.offer_optional_choices)
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
    builder.add_edge("apply_fact_patch", "apply_collection_directive")
    builder.add_edge("apply_collection_directive", "check_required_fields")
    builder.add_conditional_edges(
        "check_required_fields",
        nodes.route_after_required_check,
        {
            "prepare_clarification_question": "prepare_clarification_question",
            "prepare_skip_confirmation": "prepare_skip_confirmation",
            "prepare_required_question": "prepare_required_question",
            "prepare_optional_question": "prepare_optional_question",
            "offer_optional_information": "offer_optional_information",
            "offer_optional_choices": "offer_optional_choices",
            "build_summary": "build_summary",
        },
    )
    for node in (
        "prepare_clarification_question",
        "prepare_skip_confirmation",
        "prepare_required_question",
        "prepare_optional_question",
    ):
        builder.add_edge(node, "plan_next_questions")
    builder.add_conditional_edges(
        "plan_next_questions",
        nodes.route_question_plan,
        {
            "repair_question_plan": "repair_question_plan",
            "handle_no_progress": "handle_no_progress",
            "emit_question": "emit_question",
        },
    )
    builder.add_edge("repair_question_plan", "emit_question")
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
        "handle_no_progress",
        "offer_optional_information",
        "offer_optional_choices",
        "request_approval",
        "approval_guard",
        "clarify_approval",
        "reject_generation",
        "cancel_session",
        "closed_session",
    ):
        builder.add_edge(node, END)
    return builder.compile(checkpointer=checkpointer)
