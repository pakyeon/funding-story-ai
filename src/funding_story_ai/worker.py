from __future__ import annotations

import json
import re
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from dotenv import load_dotenv
from fastmcp import Client
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, ValidationError

from .adapter import GeminiAdapter, is_model_access_error
from .conversation import (
    OPTIONAL_FACT_FIELDS,
    OPTIONAL_FIELD_GROUPS,
    ApprovalDecision,
    ConversationModel,
    QuestionPlan,
    QuestionPurpose,
    StorySummary,
    TurnUnderstanding,
    build_conversation_graph,
    collection_ready,
    explicitly_absent_fields,
    initial_facts,
    initial_optional_collection,
    missing_required_fields,
    provided_facts,
    reconcile_turn_understanding,
    skipped_optional_fields,
    unresolved_optional_fields,
    validate_summary_grounding,
)
from .data_repository import DataRepository
from .media_projection import build_approved_generation_package
from .smoke import build_runtime

WorkerStatus = Literal[
    "awaiting-input",
    "awaiting-approval",
    "generation-ready",
    "closed",
]

MAX_MESSAGE_CHARS = 1_000
MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)

_EXPLICIT_REQUIRED_FACT_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "product_name": (re.compile(r"(?:제품명|제품\s*이름)\s*(?:은|는|:)"),),
    "product_type": (
        re.compile(r"제품(?:\s*종류|\s*유형)?\s*(?:은|는|:)"),
        re.compile(r"카테고리의\s+\S+"),
    ),
    "category": (re.compile(r"(?:카테고리|분류)\s*(?:은|는|:|의)"),),
    "key_strengths": (re.compile(r"(?:핵심\s*)?(?:강점|장점)\s*(?:은|는|:)"),),
    "target_supporters": (
        re.compile(r"(?:주요\s*)?(?:서포터|타깃|대상)\s*(?:은|는|:)"),
    ),
}


def _validate_explicit_required_fact_coverage(
    message: str, understanding: TurnUnderstanding
) -> None:
    explicit = {
        field
        for field, patterns in _EXPLICIT_REQUIRED_FACT_PATTERNS.items()
        if any(pattern.search(message) for pattern in patterns)
    }
    if len(explicit) < 2:
        return
    patched = {patch.field for patch in understanding.fact_patches}
    missing = explicit - patched
    if missing:
        raise ValueError(
            "Latest user message explicitly states required facts that were not patched: "
            f"{sorted(missing)}"
        )


class StructuredModelOutputError(RuntimeError):
    """Raised after the single allowed structured-output correction also fails."""


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    thread_id: str
    input_id: str
    message: str
    message_id: str = ""
    image_path: Path | None = None
    caller_id: str = "local-story-worker"
    idempotency_key: str = ""


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    thread_id: str
    input_id: str
    status: WorkerStatus
    stage: str
    reply: str
    requested_fields: tuple[str, ...]
    questions: tuple[str, ...]
    facts: dict[str, dict[str, Any]]
    collection_phase: str
    optional_collection: dict[str, Any]
    active_optional_group: str | None
    remaining_optional_fields: tuple[str, ...]
    current_summary: dict[str, Any] | None
    summary_version: int
    approved_summary_version: int | None
    generation_start_trigger: str | None = None
    temporary_error: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "input_id": self.input_id,
            "status": self.status,
            "stage": self.stage,
            "reply": self.reply,
            "requested_fields": list(self.requested_fields),
            "questions": list(self.questions),
            "facts": self.facts,
            "collection_phase": self.collection_phase,
            "optional_collection": self.optional_collection,
            "active_optional_group": self.active_optional_group,
            "remaining_optional_fields": list(self.remaining_optional_fields),
            "current_summary": self.current_summary,
            "summary_version": self.summary_version,
            "approved_summary_version": self.approved_summary_version,
            "generation_start_trigger": self.generation_start_trigger,
            "temporary_error": self.temporary_error,
        }


class BriefBuilder(Protocol):
    def build(self, request: WorkerRequest, semantic_state: dict[str, Any]) -> dict[str, Any]: ...


class GenerationTool(Protocol):
    async def create(
        self,
        *,
        generation_package: dict[str, Any],
        caller_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]: ...


def validate_worker_request(request: WorkerRequest) -> None:
    if not request.thread_id.strip():
        raise ValueError("thread_id must not be empty")
    if not request.input_id.strip():
        raise ValueError("input_id must not be empty")
    if not request.message.strip():
        raise ValueError("message must not be empty")
    if len(request.message) > MAX_MESSAGE_CHARS:
        raise ValueError(f"message must be at most {MAX_MESSAGE_CHARS:,} characters")
    if request.image_path is not None:
        if not request.image_path.is_file():
            raise FileNotFoundError(request.image_path)
        if request.image_path.suffix.lower() not in SUPPORTED_IMAGES:
            raise ValueError("Product image must be JPG, PNG, or WEBP")
        if request.image_path.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError("Product image must be at most 10 MB")


def _image_payload(path: Path | None) -> list[tuple[bytes, str]]:
    if path is None:
        return []
    return [(path.read_bytes(), SUPPORTED_IMAGES[path.suffix.lower()])]


class GeminiConversationModel:
    """Gemini-backed semantic nodes with separate structured contracts."""

    def __init__(self, adapter: GeminiAdapter) -> None:
        self.adapter = adapter
        self.last_model: str | None = None
        self.last_call_count = 0

    def _generate_validated(
        self,
        *,
        prompt: str,
        response_model: type[StructuredModel],
        images: list[tuple[bytes, str]] | None = None,
        validator: Callable[[StructuredModel], None] | None = None,
    ) -> StructuredModel:
        self.last_call_count = 0
        current_prompt = prompt
        last_error: Exception | None = None
        for attempt in range(2):
            self.last_call_count += 1
            try:
                result = (
                    self.adapter.generate_multimodal_json(
                        prompt=current_prompt,
                        images=images,
                        response_schema=response_model.model_json_schema(),
                    )
                    if images is not None
                    else self.adapter.generate_json(
                        prompt=current_prompt,
                        response_schema=response_model.model_json_schema(),
                    )
                )
                self.last_model = result.model
                validated = response_model.model_validate(result.data)
                if validator is not None:
                    validator(validated)
                return validated
            except (ValueError, ValidationError) as exc:
                last_error = exc
                if attempt == 0:
                    current_prompt = (
                        prompt
                        + "\n\n직전 구조화 출력이 계약 검증에 실패했습니다. 아래 오류를 바로잡아 "
                        "스키마에 맞는 JSON만 다시 반환하세요.\n검증 오류: "
                        + str(exc)
                    )
        raise StructuredModelOutputError(
            "structured response validation failed after one correction"
        ) from last_error

    @staticmethod
    def _context(messages: list[dict[str, str]], facts: dict[str, dict[str, Any]]) -> str:
        return json.dumps(
            {"messages": messages, "current_facts": facts},
            ensure_ascii=False,
            indent=2,
        )

    def understand_turn(
        self,
        *,
        message: dict[str, str],
        messages: list[dict[str, str]],
        facts: dict[str, dict[str, Any]],
        image_path: str | None,
    ) -> TurnUnderstanding:
        prompt = f"""당신은 크라우드펀딩 스토리 작성 대화의 입력 이해 노드입니다.
최신 사용자 발화의 의도와 현재 사실을 어떻게 추가·수정·삭제하는지 구조화하세요.

출력 전 점검 순서:
1. 최신 사용자 발화를 문장 단위로 끝까지 읽습니다.
2. 아래 16개 필드별로 사용자가 직접 말한 사실이 있는지 각각 확인합니다.
3. 한 문장에 여러 사실이 있으면 fact_patches를 필드별로 모두 분리합니다.
4. 사실 추출을 마친 뒤 발화 의도와 선택 정보 수집 지시를 별도로 판정합니다.
5. 생성·질문·취소 의도와 사실 제공이 함께 있어도 서로 하나를 버리지 않습니다.

사실 필드 정의:
- product_name: 사용자가 정한 제품 고유명
- product_type: 로봇청소기와 같은 제품 종류
- category: 테크·가전과 같은 펀딩 카테고리
- key_strengths: 제품이 제공하는 핵심 기능·특징·차별점
- target_supporters: 핵심 서포터 집단이나 사용자 유형
- problem_context: 사용자가 제품으로 해결하려는 불편·문제 상황
- trust_elements: 인증·시험·검증·수치 근거
- maker_team_intro: 메이커·브랜드·팀의 구성과 경력
- rewards: 리워드 구성·수량·가격
- funding_end: 펀딩 종료일
- shipping_start: 리워드 발송 시작일이나 시작 일정
- refund_policy: 교환·환불 조건과 기간
- as_policy: 보증·수리·사후지원 정책
- funding_plan: 펀딩금 사용 계획
- platform_choice: 와디즈를 선택한 이유. 플랫폼 이름만 언급한 것은 해당하지 않음
- risk_response: 생산·공급·배송 위험과 대응 조치

규칙:
- 사용자가 직접 제공한 사실만 fact_patches로 반환합니다.
- values는 사용자가 말한 핵심 표현을 그대로 보존합니다. 조사·어미를 정리하는 최소한의
  분리 외에는 요약, 재서술, 의미 확장, 날짜 형식 변경을 하지 않습니다.
- 최신 발화에 명시된 사실을 빠짐없이 각각 알맞은 필드로 분리합니다.
- 긴 문장이나 여러 문장에서도 제품명·제품 종류·카테고리·강점·타깃과 선택 정보가 각각
  명시됐다면 일부만 고르지 말고 모두 반환합니다.
- "이 정보로 생성해줘" 같은 생성 요청이 문장 끝에 있어도 앞에서 제공한 모든 사실을 먼저
  fact_patches에 보존하고 intent만 request_generation으로 분류합니다.
- 타깃 집단과 그 집단의 문제 상황이 함께 나오면 target_supporters와 problem_context로
  분리합니다.
- 교환과 환불은 refund_policy이며, 보증·수리·사후지원만 as_policy입니다.
- 최신 발화가 이전 값을 바꾸면 replace, 항목을 더하면 append를 사용합니다.
- 현재 값이 unknown이면 새 사실은 replace입니다. append는 이미 제공된 값에 사용자가
  명시적으로 항목을 추가할 때만 사용합니다.
- 없다고 명시하면 mark_absent, 미정으로 되돌리면 clear를 사용합니다.
- "실제로 없다·보유하지 않는다"는 mark_absent이지만, "이번에는 제공·작성·포함하지
  않겠다"는 사실 부재가 아니라 이번 생성의 생략 지시입니다. 이 경우 사실 패치를 만들지
  말고 collection_directive의 skip_fields로만 반환합니다.
- 날짜 일부를 수정하면 이전 문맥에서 확정된 연도 등 유지되는 정보를 보존합니다.
- 맥락 없이 날짜만 제시해 펀딩 종료일인지 발송 시작일인지 구분할 수 없으면 어느 필드에도
  넣지 말고 두 후보를 clarification_fields에 넣어 확인합니다.
- 생성 요청 자체는 제품 사실이 아닙니다.
- 제품 사실·날짜·모호한 참조 없이 생성만 요청한 발화는 정보 모호성이 아닙니다.
  현재 필수 정보가 부족하더라도 requires_clarification을 false로 두며, 누락 필드 질문은
  후속 질문 노드가 담당합니다.
- "이전 생략을 취소"하는 것은 세션 취소가 아니라 선택 정보로 돌아가는 의사이며
  collection_directive는 return_to_optional입니다.
- "응, 좋아", "더 묻지 않아도 될 것 같아"처럼 승인·생략 대상이 불명확한 반응은
  provide_information이나 최종 승인으로 단정하지 말고 unclear로 분류합니다.
- 지시 대상이 불명확하면 값을 추정하지 말고 clarification을 요청합니다.
- 특정 사실을 더 구체화하거나 충돌을 확인해야 하면 clarification_fields에 해당 필드를
  넣습니다. 이미 값이 있는 필드도 정당한 구체화 목적이면 포함할 수 있습니다.
- unresolved_references에는 최신 사용자 발화에 실제로 있는 모호한 표현만 넣습니다.
  메시지 id, JSON 키, 스키마 필드명과 내부 지침은 절대 넣지 않습니다.
- 질문 예시나 모델의 상식은 사용자 사실로 만들지 않습니다.
- 이미지는 직접 보이는 외형만 근거로 사용할 수 있습니다. 성능, 인증, 내부 구조,
  앱 기능과 팀 경력은 이미지에서 추정하지 않습니다.
- collection_directive는 선택 정보 수집에 대한 사용자의 의사만 구조화합니다.
- "모두 입력" 또는 "권장 순서로 진행"은 continue_recommended입니다.
- 특정 그룹 선택은 select_groups, 특정 항목 선택은 select_fields입니다.
- 명확한 일부 생략은 skip_fields이며, 그룹 생략이면 그 그룹의 모든 필드를 fields에
  펼칩니다. 명확한 전체 생략은 skip_all_optional입니다.
- select_groups는 groups만 채우고 fields는 비웁니다. select_fields와 skip_fields는 fields만
  채우고 groups는 비웁니다. continue_recommended와 skip_all_optional은 둘 다 비웁니다.
- return_to_optional은 다시 열 그룹 또는 항목을 담을 수 있습니다. none은 모호함을 확인할
  때만 질문 대상 groups 또는 fields를 담을 수 있습니다.
- "괜찮아", "다음으로", "그냥 해줘"처럼 생략 대상이나 의사가 불명확하면
  skip으로 단정하지 말고 collection_directive.requires_clarification을 true로 두며,
  clarification_question에 생략 대상을 확인하는 질문을 반드시 작성합니다.
- 생성 요청은 선택 정보 생략도, 최종 생성 승인도 아닙니다.

예시 1:
- 최신 발화: "슬림봇 R1이라는 로봇청소기야. 얇은 본체가 강점이고 원룸 거주자를
  타깃으로 이 정보로 생성해줘."
- intent는 request_generation이지만 product_name, product_type, key_strengths,
  target_supporters 네 필드의 패치를 모두 반환합니다.

예시 2:
- 최신 발화: "추가 일정은 2026년 11월 30일이야."
- 종료일인지 발송일인지 확인할 수 없으므로 날짜를 추정해 채우지 않고 명확화를 요청합니다.

선택 정보 그룹:
{json.dumps(OPTIONAL_FIELD_GROUPS, ensure_ascii=False)}

현재 문맥:
{self._context(messages, facts)}

최신 사용자 발화:
{json.dumps(message, ensure_ascii=False)}
"""
        images = _image_payload(Path(image_path)) if image_path else []
        understanding = self._generate_validated(
            prompt=prompt,
            images=images,
            response_model=TurnUnderstanding,
            validator=lambda candidate: _validate_explicit_required_fact_coverage(
                message["content"], candidate  # type: ignore[arg-type]
            ),
        )
        return reconcile_turn_understanding(message["content"], understanding)

    def plan_questions(
        self,
        *,
        messages: list[dict[str, str]],
        facts: dict[str, dict[str, Any]],
        purpose: QuestionPurpose,
        candidate_fields: list[str],
        requested_group: str | None,
        requested_detail: str,
        question_history: list[dict[str, Any]],
        turn_understanding: TurnUnderstanding,
    ) -> QuestionPlan:
        prompt = f"""당신은 크라우드펀딩 스토리 작성 대화의 다음 질문 계획 노드입니다.
현재 질문 목적과 허용 후보를 바탕으로 다음 한 번의 자연스러운 질문을 작성하세요.

규칙:
- purpose, requested_group, requested_detail은 아래 값을 그대로 복사합니다.
- requested_fields는 candidate_fields 안에서만 고르고 최대 3개입니다.
- required와 optional-collect는 requested_fields에 하나 이상을 포함합니다.
- clarify와 confirm-skip은 필드가 없어도 됩니다.
- 서로 밀접한 항목만 한 질문으로 묶습니다.
- optional-collect는 하나의 requested_group에 속한 항목만 묶습니다.
- 제공·실제 없음·생략으로 해소된 정보를 같은 목적으로 반복 질문하지 않습니다.
- 사용자가 곧바로 생성을 요청해도 필수 정보는 생략하지 않습니다.
- 질문에 사실 예시를 넣더라도 그 예시는 확정 정보가 아닙니다.

질문 목적: {purpose}
허용 후보: {json.dumps(candidate_fields, ensure_ascii=False)}
요청 그룹: {json.dumps(requested_group, ensure_ascii=False)}
요청 세부 내용: {json.dumps(requested_detail, ensure_ascii=False)}
질문 이력: {json.dumps(question_history, ensure_ascii=False)}
이번 입력 분석: {turn_understanding.model_dump_json()}
현재 문맥:
{self._context(messages, facts)}
"""
        result = self.adapter.generate_json(
            prompt=prompt,
            response_schema=QuestionPlan.model_json_schema(),
        )
        self.last_model = result.model
        return QuestionPlan.model_validate(result.data)

    def repair_question_plan(
        self,
        *,
        invalid_plan: QuestionPlan | None,
        validation_error: str,
        messages: list[dict[str, str]],
        facts: dict[str, dict[str, Any]],
        purpose: QuestionPurpose,
        candidate_fields: list[str],
        requested_group: str | None,
        requested_detail: str,
    ) -> QuestionPlan:
        prompt = f"""당신은 잘못된 후속 질문 계획을 한 번만 교정하는 노드입니다.
아래 제약을 모두 지키는 QuestionPlan을 반환하세요.

- purpose: {purpose}
- candidate_fields 안에서만 최대 3개 선택
- requested_group: {json.dumps(requested_group, ensure_ascii=False)}
- requested_detail: {json.dumps(requested_detail, ensure_ascii=False)}
- optional-collect이면 같은 그룹 항목만 선택
- required와 optional-collect이면 최소 1개 선택
- clarify와 confirm-skip이면 필드가 없어도 됨

검증 오류: {validation_error}
잘못된 계획: {invalid_plan.model_dump_json() if invalid_plan else "null"}
현재 사실: {json.dumps(facts, ensure_ascii=False)}
최근 대화: {json.dumps(messages[-6:], ensure_ascii=False)}
허용 후보: {json.dumps(candidate_fields, ensure_ascii=False)}
"""
        result = self.adapter.generate_json(
            prompt=prompt,
            response_schema=QuestionPlan.model_json_schema(),
        )
        self.last_model = result.model
        return QuestionPlan.model_validate(result.data)

    def build_summary(
        self,
        *,
        messages: list[dict[str, str]],
        facts: dict[str, dict[str, Any]],
        optional_collection: dict[str, Any],
    ) -> StorySummary:
        provided = provided_facts(facts)
        absent = explicitly_absent_fields(facts)
        skipped = skipped_optional_fields(optional_collection)
        prompt = f"""당신은 크라우드펀딩 스토리 생성 직전의 사용자 확인 요약 노드입니다.
제공된 사실을 읽기 쉬운 한국어로 정리하고 생성 여부를 명시적으로 물으세요.

규칙:
- confirmed_facts는 아래 확정 사실을 키와 값까지 그대로 복사합니다.
- explicitly_absent_fields와 skipped_fields는 아래 목록을 순서까지 그대로 복사합니다.
- summary_text에는 확정 사실만 사용하고 새 수치·성능·인증·효과를 추가하지 않습니다.
- summary_text에서 실제로 없음과 이번 생성에서 생략함을 서로 구분합니다.
- 이 출력은 스토리 본문이 아니라 사용자가 결정 내용을 확인하기 위한 요약입니다.
- confirmation_question은 이 내용으로 스토리를 생성할지 명시적으로 묻습니다.

확정 사실: {json.dumps(provided, ensure_ascii=False)}
실제로 없음: {json.dumps(absent, ensure_ascii=False)}
이번 생성에서 생략: {json.dumps(skipped, ensure_ascii=False)}
대화 문맥: {json.dumps(messages, ensure_ascii=False)}
"""
        return self._generate_validated(
            prompt=prompt,
            response_model=StorySummary,
            validator=lambda summary: validate_summary_grounding(
                summary, facts, optional_collection  # type: ignore[arg-type]
            ),
        )

    def classify_approval(
        self,
        *,
        message: dict[str, str],
        summary: StorySummary,
        messages: list[dict[str, str]],
    ) -> ApprovalDecision:
        prompt = f"""당신은 스토리 생성 승인 의도 분류 노드입니다.
사용자의 최신 발화를 approve, revise, reject, ambiguous 중 하나로 분류하세요.

규칙:
- 현재 요약 그대로 생성을 명시적으로 요청한 경우만 approve입니다.
- 사실 추가·수정·삭제가 포함되면 revise입니다.
- 생성하지 않겠다는 의사는 reject입니다.
- 질문, 제안, 조건부 표현과 확신할 수 없는 동의는 ambiguous입니다.
- 질문을 건너뛰겠다는 말은 approve가 아닙니다.

승인 대상 요약:
{summary.model_dump_json()}

최신 사용자 발화:
{json.dumps(message, ensure_ascii=False)}

최근 대화:
{json.dumps(messages[-6:], ensure_ascii=False)}
"""
        return self._generate_validated(prompt=prompt, response_model=ApprovalDecision)


def _brief_id(input_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "-", input_id.lower()).strip("-_")
    return normalized if len(normalized) >= 2 else f"input-{normalized or 'unknown'}"


def _values(slots: dict[str, Any], name: str) -> list[str]:
    return list(slots[name]["values"])


def _provided(slots: dict[str, Any], name: str) -> bool:
    return slots[name]["status"] == "provided" and bool(slots[name]["values"])


def _first(slots: dict[str, Any], name: str) -> str | None:
    values = _values(slots, name)
    return values[0] if values else None


def _iso_date(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


class GroundedBriefBuilder:
    """Map extracted slots to a brief without adding product knowledge."""

    _UNKNOWN_SPECS = {
        "rewards": ("확정된 리워드 구성과 가격이 무엇인가요?", ["rewards", "cta"]),
        "schedule_policy": (
            "확정된 펀딩 종료일, 발송 일정과 정책이 무엇인가요?",
            ["timeline", "risks", "cta"],
        ),
        "funding_plan": ("펀딩금 사용 계획이 무엇인가요?", ["funding_plan"]),
        "platform_choice": ("플랫폼 선택 이유가 무엇인가요?", ["platform_choice"]),
        "risk_response": ("생산·공급 리스크와 대응 계획이 무엇인가요?", ["risks"]),
    }

    def __init__(self, *, repository: DataRepository) -> None:
        self.repository = repository

    def build(self, request: WorkerRequest, semantic_state: dict[str, Any]) -> dict[str, Any]:
        slots = semantic_state["slots"]
        required = ("product_name", "product_type", "category")
        missing = [name for name in required if not _provided(slots, name)]
        if missing:
            raise ValueError(f"Cannot build a brief without: {', '.join(missing)}")

        source_id = "source_maker_dialogue"
        refs = [
            {
                "source_id": source_id,
                "source_type": "maker-input",
                "location": f"worker:{request.input_id}",
                "captured_at": None,
            }
        ]
        if request.image_path is not None:
            refs.append(
                {
                    "source_id": "source_product_image",
                    "source_type": "image",
                    "location": f"worker-image:{request.input_id}",
                    "captured_at": None,
                }
            )

        strengths = _values(slots, "key_strengths")
        authoritative = list(semantic_state["fact_conflict"]["authoritative_values"])
        fact_values = list(dict.fromkeys([*strengths, *authoritative]))
        facts = [
            {
                "id": f"fact_user_value_{index:02d}",
                "name": f"사용자 제공 정보 {index}",
                "value": value,
                "unit": None,
                "source_refs": [source_id],
            }
            for index, value in enumerate(fact_values, start=1)
        ]
        fact_ids = {fact["value"]: fact["id"] for fact in facts}
        features = [
            {
                "id": f"feature_strength_{index:02d}",
                "name": value,
                "description": value,
                "fact_ids": [fact_ids[value]],
                "evidence_ids": [],
                "source_refs": [source_id],
            }
            for index, value in enumerate(strengths, start=1)
        ]
        audiences = [
            {
                "id": f"aud_target_{index:02d}",
                "description": value,
                "source_refs": [source_id],
            }
            for index, value in enumerate(_values(slots, "target_supporters"), start=1)
        ]
        problems = [
            {
                "id": f"problem_context_{index:02d}",
                "description": value,
                "source_refs": [source_id],
            }
            for index, value in enumerate(_values(slots, "problem_context"), start=1)
        ]

        raw_funding_end = _first(slots, "funding_end")
        raw_shipping_start = _first(slots, "shipping_start")
        non_iso_schedule = [
            f"{label}: {value}"
            for label, value in (
                ("펀딩 종료 일정", raw_funding_end),
                ("발송 시작 일정", raw_shipping_start),
            )
            if value is not None and _iso_date(value) is None
        ]
        claim_values = list(
            dict.fromkeys(
                [
                    *_values(slots, "trust_elements"),
                    *_values(slots, "maker_team_intro"),
                    *_values(slots, "funding_plan"),
                    *_values(slots, "platform_choice"),
                    *_values(slots, "risk_response"),
                    *non_iso_schedule,
                ]
            )
        )
        claims = [
            {
                "id": f"claim_maker_statement_{index:02d}",
                "statement": value,
                "status": "maker-stated-unverified",
                "evidence_ids": [],
                "source_refs": [source_id],
            }
            for index, value in enumerate(claim_values, start=1)
        ]

        rewards = [
            {
                "id": f"reward_user_value_{index:02d}",
                "name": value,
                "price_krw": None,
                "quantity": None,
                "components": [value],
                "source_refs": [source_id],
            }
            for index, value in enumerate(_values(slots, "rewards"), start=1)
        ]
        schedule_values = {
            "funding_end": _iso_date(raw_funding_end),
            "shipping_start": _iso_date(raw_shipping_start),
            "refund_policy": _first(slots, "refund_policy"),
            "as_policy": _first(slots, "as_policy"),
        }
        assets: list[dict[str, Any]] = []
        if request.image_path is not None:
            superseded = semantic_state["fact_conflict"]["status"] == "superseded-resolved"
            assets.append(
                {
                    "id": "asset_product_reference",
                    "asset_type": "product",
                    "description": (
                        "사용자가 대체 대상으로 명시한 참조 이미지"
                        if superseded
                        else "사용자가 첨부한 제품 참조 이미지"
                    ),
                    "allowed_sections": [] if superseded else ["hero", "solution", "features"],
                    "source_refs": ["source_product_image"],
                }
            )

        unknowns = []
        if not rewards and slots["rewards"]["status"] == "unknown":
            question, sections = self._UNKNOWN_SPECS["rewards"]
            unknowns.append({"field": "rewards", "question": question, "blocks_sections": sections})
        schedule_known = any(schedule_values.values()) or bool(non_iso_schedule)
        if not schedule_known and all(
            slots[name]["status"] == "unknown"
            for name in ("funding_end", "shipping_start", "refund_policy", "as_policy")
        ):
            question, sections = self._UNKNOWN_SPECS["schedule_policy"]
            unknowns.append(
                {
                    "field": "schedule_policy",
                    "question": question,
                    "blocks_sections": sections,
                }
            )
        for field in ("funding_plan", "platform_choice", "risk_response"):
            if slots[field]["status"] == "unknown":
                question, sections = self._UNKNOWN_SPECS[field]
                unknowns.append({"field": field, "question": question, "blocks_sections": sections})

        summary_parts = list(
            dict.fromkeys(
                [
                    _first(slots, "product_name") or "",
                    *strengths,
                    *_values(slots, "target_supporters"),
                ]
            )
        )
        brief = {
            "schema_version": "story-brief-v1",
            "brief_id": _brief_id(request.input_id),
            "language": semantic_state["language"],
            "source": {
                "project_id": None,
                "project_url": None,
                "purpose": "maker-brief",
                "snapshot_date": date.today().isoformat(),
                "refs": refs,
            },
            "product": {
                "name": _first(slots, "product_name"),
                "category": _first(slots, "category"),
                "product_type": _first(slots, "product_type"),
                "summary": " / ".join(part for part in summary_parts if part),
                "facts": facts,
            },
            "audiences": audiences,
            "problems": problems,
            "features": features,
            "claims": claims,
            "evidence": [],
            "assets": assets,
            "rewards": rewards,
            "schedule_policy": {
                **schedule_values,
                "source_refs": [source_id] if schedule_known else [],
            },
            "unknowns": unknowns,
        }
        self.repository.validate_story_brief(brief)
        return brief


class FastMcpGenerationTool:
    """The worker-facing MCP surface exposes only story creation."""

    def __init__(self, server: str | Any) -> None:
        self.server = server

    async def create(
        self,
        *,
        generation_package: dict[str, Any],
        caller_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        arguments = {
            "request": {
                "caller_id": caller_id,
                "idempotency_key": idempotency_key,
                "generation_package": generation_package,
                "template_id": None,
            }
        }
        async with Client(self.server) as client:
            tools = await client.list_tools()
            if [tool.name for tool in tools] != ["create_crowdfunding_story"]:
                raise RuntimeError("Worker MCP allowlist must contain only story creation")
            result = await client.call_tool("create_crowdfunding_story", arguments)
            if result.structured_content is None:
                raise RuntimeError("Story generation tool returned no structured content")
            return dict(result.structured_content)


class StoryGenerationDispatcher:
    """Dispatch a separately approved generation-ready state across the MCP boundary."""

    def __init__(
        self,
        *,
        repository: DataRepository,
        brief_builder: BriefBuilder,
        generation_tool: GenerationTool,
    ) -> None:
        self.repository = repository
        self.brief_builder = brief_builder
        self.generation_tool = generation_tool

    async def submit(
        self,
        *,
        request: WorkerRequest,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        if state.get("workflow_stage") != "generation-ready":
            raise ValueError("Generation dispatch requires a generation-ready worker state")
        if state.get("approved_summary_version") != state.get("summary_version"):
            raise ValueError("Approved summary version does not match the current summary")
        semantic_state = graph_state_to_semantic_state(input_id=request.input_id, state=state)
        self.repository.validate_intake_semantic_state(semantic_state)
        brief = self.brief_builder.build(request, semantic_state)
        local_asset_paths = {}
        if state.get("image_path"):
            local_asset_paths["asset_product_reference"] = Path(state["image_path"])
        generation_package = build_approved_generation_package(
            repository=self.repository,
            input_id=request.input_id,
            thread_id=request.thread_id,
            state=state,
            brief=brief,
            local_asset_paths=local_asset_paths,
        )
        summary_version = int(state["approved_summary_version"])
        idempotency_key = request.idempotency_key or (
            f"worker-{request.thread_id}-summary-{summary_version}"
        )
        return await self.generation_tool.create(
            generation_package=generation_package,
            caller_id=request.caller_id,
            idempotency_key=idempotency_key,
        )


def graph_state_to_semantic_state(*, input_id: str, state: dict[str, Any]) -> dict[str, Any]:
    """Adapt approved graph facts to the existing grounded brief boundary."""
    slots: dict[str, dict[str, Any]] = {}
    for field, value in state["facts"].items():
        turn = int(value.get("updated_at_turn", 0))
        slots[field] = {
            "status": value["status"],
            "values": list(value["values"]),
            "source_turn": "none" if turn == 0 else ("initial" if turn == 1 else "followup"),
        }

    return {
        "schema_version": "story-intake-semantic-state-v2",
        "input_id": input_id,
        "language": "ko",
        "image_attached": bool(state.get("image_attached", False)),
        "slots": slots,
        "fact_conflict": {
            "status": "none",
            "authoritative_values": [],
            "superseded_values": [],
        },
        "decision": {
            "ready_to_confirm": not missing_required_fields(state["facts"])
            and collection_ready(state["optional_collection"]),
            "requested_fields": [],
            "follow_up_question": None,
        },
    }


class StoryMakerWorker:
    """Stateful conversation worker ending at the generation-ready boundary."""

    def __init__(
        self,
        *,
        conversation_model: ConversationModel,
        checkpointer: Any | None = None,
        checkpoint_connection: sqlite3.Connection | None = None,
    ) -> None:
        self.conversation_model = conversation_model
        self.checkpointer = checkpointer or InMemorySaver()
        self.checkpoint_connection = checkpoint_connection
        self.conversation_graph = build_conversation_graph(
            conversation_model,
            checkpointer=self.checkpointer,
        )

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def _message(request: WorkerRequest) -> dict[str, str]:
        return {
            "id": request.message_id.strip() or f"user-{uuid.uuid4()}",
            "role": "user",
            "content": request.message.strip(),
        }

    @staticmethod
    def _status(stage: str) -> WorkerStatus:
        if stage == "awaiting-approval":
            return "awaiting-approval"
        if stage == "generation-ready":
            return "generation-ready"
        if stage == "cancelled":
            return "closed"
        return "awaiting-input"

    @staticmethod
    def _outcome(
        request: WorkerRequest,
        state: dict[str, Any],
        *,
        trigger: str | None = None,
        temporary_error: bool = False,
    ) -> WorkerOutcome:
        stage = str(state.get("workflow_stage", "collecting"))
        reply = str(state.get("reply", ""))
        questions = (reply,) if reply and stage in {"collecting", "awaiting-approval"} else ()
        optional_collection = dict(state.get("optional_collection", {}))
        return WorkerOutcome(
            thread_id=request.thread_id,
            input_id=request.input_id,
            status=StoryMakerWorker._status(stage),
            stage=stage,
            reply=reply,
            requested_fields=tuple(state.get("requested_fields", [])),
            questions=questions,
            facts=dict(state.get("facts", {})),
            collection_phase=str(state.get("collection_phase", "required")),
            optional_collection=optional_collection,
            active_optional_group=optional_collection.get("active_group"),
            remaining_optional_fields=tuple(
                unresolved_optional_fields(optional_collection)
                if optional_collection
                else OPTIONAL_FACT_FIELDS
            ),
            current_summary=state.get("current_summary"),
            summary_version=int(state.get("summary_version", 0)),
            approved_summary_version=state.get("approved_summary_version"),
            generation_start_trigger=trigger,
            temporary_error=temporary_error,
        )

    async def handle(self, request: WorkerRequest) -> WorkerOutcome:
        validate_worker_request(request)
        message = self._message(request)
        graph_input: dict[str, Any] = {
            "incoming_message": message,
            "messages": [message],
            "input_id": request.input_id,
            "caller_id": request.caller_id,
            "idempotency_key": request.idempotency_key,
        }
        if request.image_path is not None:
            graph_input.update(
                {
                    "image_path": str(request.image_path),
                    "image_attached": True,
                }
            )
        config = self._config(request.thread_id)
        try:
            state = self.conversation_graph.invoke(graph_input, config)
        except Exception as exc:
            if not (
                is_model_access_error(exc) or isinstance(exc, StructuredModelOutputError)
            ):
                raise
            state = self.get_state(request.thread_id)
            if not state:
                facts = initial_facts()
                state = {
                    "facts": facts,
                    "facts_revision": 0,
                    "collection_revision": 0,
                    "optional_collection": initial_optional_collection(facts),
                    "workflow_stage": "collecting",
                    "collection_phase": "required",
                    "summary_version": 0,
                }
            state = {
                **state,
                "reply": (
                    "AI 응답을 일시적으로 받지 못했습니다. 현재 입력 상태는 유지했습니다. "
                    "잠시 뒤 같은 내용을 다시 입력해 주세요."
                ),
                "requested_fields": [],
            }
            return self._outcome(request, state, temporary_error=True)
        trigger = (
            "explicit-confirmation" if state.get("workflow_stage") == "generation-ready" else None
        )
        return self._outcome(request, state, trigger=trigger)

    def get_state(self, thread_id: str) -> dict[str, Any]:
        return dict(self.conversation_graph.get_state(self._config(thread_id)).values)

    def delete_thread(self, thread_id: str) -> None:
        self.checkpointer.delete_thread(thread_id)

    def close(self) -> None:
        if self.checkpoint_connection is not None:
            self.checkpoint_connection.close()


def build_live_worker(
    *,
    root: Path | None = None,
    checkpoint_path: Path | None = None,
) -> StoryMakerWorker:
    load_dotenv()
    repository = DataRepository(root)
    adapter = GeminiAdapter(build_runtime())
    state_path = checkpoint_path or (
        repository.root / "artifacts" / "state" / "conversations.sqlite3"
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state_path, check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    checkpointer.setup()
    return StoryMakerWorker(
        conversation_model=GeminiConversationModel(adapter),
        checkpointer=checkpointer,
        checkpoint_connection=connection,
    )
