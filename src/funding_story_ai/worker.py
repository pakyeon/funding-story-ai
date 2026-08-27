from __future__ import annotations

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol

from dotenv import load_dotenv
from fastmcp import Client
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from .adapter import GeminiAdapter
from .conversation import (
    ApprovalDecision,
    ConversationModel,
    QuestionPlan,
    StorySummary,
    TurnUnderstanding,
    build_conversation_graph,
    missing_required_fields,
)
from .data_repository import DataRepository
from .smoke import build_runtime

WorkerStatus = Literal["awaiting-input", "awaiting-approval", "submitted", "closed"]

MAX_MESSAGE_CHARS = 1_000
MAX_IMAGE_BYTES = 10 * 1024 * 1024
SUPPORTED_IMAGES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


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
    current_summary: dict[str, Any] | None
    summary_version: int
    approved_summary_version: int | None
    generation_start_trigger: str | None = None
    tool_result: dict[str, Any] | None = None

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
            "current_summary": self.current_summary,
            "summary_version": self.summary_version,
            "approved_summary_version": self.approved_summary_version,
            "generation_start_trigger": self.generation_start_trigger,
            "tool_result": self.tool_result,
        }


class BriefBuilder(Protocol):
    def build(
        self, request: WorkerRequest, semantic_state: dict[str, Any]
    ) -> dict[str, Any]: ...


class GenerationTool(Protocol):
    async def create(
        self,
        *,
        brief: dict[str, Any],
        caller_id: str,
        idempotency_key: str,
        reference_image_path: Path | None,
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

    @staticmethod
    def _context(
        messages: list[dict[str, str]], facts: dict[str, dict[str, Any]]
    ) -> str:
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
최신 사용자 발화가 현재 사실을 어떻게 추가·수정·삭제하는지만 구조화하세요.

규칙:
- 사용자가 직접 제공한 사실만 fact_patches로 반환합니다.
- 최신 발화가 이전 값을 바꾸면 replace, 항목을 더하면 append를 사용합니다.
- 없다고 명시하면 mark_absent, 미정으로 되돌리면 clear를 사용합니다.
- 생성 요청 자체는 제품 사실이 아닙니다.
- 지시 대상이 불명확하면 값을 추정하지 말고 clarification을 요청합니다.
- 질문 예시나 모델의 상식은 사용자 사실로 만들지 않습니다.
- 이미지는 직접 보이는 외형만 근거로 사용할 수 있습니다. 성능, 인증, 내부 구조,
  앱 기능과 팀 경력은 이미지에서 추정하지 않습니다.

현재 문맥:
{self._context(messages, facts)}

최신 사용자 발화:
{json.dumps(message, ensure_ascii=False)}
"""
        images = _image_payload(Path(image_path)) if image_path else []
        result = self.adapter.generate_multimodal_json(
            prompt=prompt,
            images=images,
            response_schema=TurnUnderstanding.model_json_schema(),
        )
        return TurnUnderstanding.model_validate(result.data)

    def plan_questions(
        self,
        *,
        messages: list[dict[str, str]],
        facts: dict[str, dict[str, Any]],
        missing_required_fields: list[str],
        asked_topics: list[str],
        turn_understanding: TurnUnderstanding,
    ) -> QuestionPlan:
        prompt = f"""당신은 크라우드펀딩 스토리 작성 대화의 다음 질문 계획 노드입니다.
현재 사실과 부족한 필수 정보를 바탕으로 다음 한 번의 자연스러운 질문을 작성하세요.

규칙:
- 필수 정보가 부족하면 requested_fields에 그중 하나 이상을 반드시 포함합니다.
- 서로 밀접한 항목만 한 질문으로 묶습니다.
- 이미 답한 정보는 반복 질문하지 않습니다.
- 이미 물었지만 답하지 않은 선택 정보보다 필수 정보를 우선합니다.
- 사용자가 곧바로 생성을 요청해도 필수 정보는 생략하지 않습니다.
- 질문에 사실 예시를 넣더라도 그 예시는 확정 정보가 아닙니다.

부족한 필수 필드: {json.dumps(missing_required_fields, ensure_ascii=False)}
이미 질문한 주제: {json.dumps(asked_topics, ensure_ascii=False)}
이번 입력 분석: {turn_understanding.model_dump_json()}
현재 문맥:
{self._context(messages, facts)}
"""
        result = self.adapter.generate_json(
            prompt=prompt,
            response_schema=QuestionPlan.model_json_schema(),
        )
        return QuestionPlan.model_validate(result.data)

    def build_summary(
        self,
        *,
        messages: list[dict[str, str]],
        facts: dict[str, dict[str, Any]],
    ) -> StorySummary:
        provided = {
            field: value["values"]
            for field, value in facts.items()
            if value["status"] == "provided"
        }
        unconfirmed = [
            field for field, value in facts.items() if value["status"] != "provided"
        ]
        prompt = f"""당신은 크라우드펀딩 스토리 생성 직전의 사용자 확인 요약 노드입니다.
제공된 사실을 읽기 쉬운 한국어로 정리하고 생성 여부를 명시적으로 물으세요.

규칙:
- confirmed_facts는 아래 확정 사실을 키와 값까지 그대로 복사합니다.
- unconfirmed_fields는 아래 목록을 순서까지 그대로 복사합니다.
- summary_text에는 확정 사실만 사용하고 새 수치·성능·인증·효과를 추가하지 않습니다.
- 이 출력은 스토리 본문이 아니라 사용자가 결정 내용을 확인하기 위한 요약입니다.
- confirmation_question은 이 내용으로 스토리를 생성할지 명시적으로 묻습니다.

확정 사실: {json.dumps(provided, ensure_ascii=False)}
미확인 필드: {json.dumps(unconfirmed, ensure_ascii=False)}
대화 문맥: {json.dumps(messages, ensure_ascii=False)}
"""
        result = self.adapter.generate_json(
            prompt=prompt,
            response_schema=StorySummary.model_json_schema(),
        )
        return StorySummary.model_validate(result.data)

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
        result = self.adapter.generate_json(
            prompt=prompt,
            response_schema=ApprovalDecision.model_json_schema(),
        )
        return ApprovalDecision.model_validate(result.data)


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

    def build(
        self, request: WorkerRequest, semantic_state: dict[str, Any]
    ) -> dict[str, Any]:
        slots = semantic_state["slots"]
        required = ("product_name", "product_type", "category")
        missing = [name for name in required if not _provided(slots, name)]
        if missing:
            raise ValueError(f"Cannot build a brief without: {', '.join(missing)}")

        source_id = "source_maker_dialogue"
        refs = [{
            "source_id": source_id,
            "source_type": "maker-input",
            "location": f"worker:{request.input_id}",
            "captured_at": None,
        }]
        if request.image_path is not None:
            refs.append({
                "source_id": "source_product_image",
                "source_type": "image",
                "location": f"worker-image:{request.input_id}",
                "captured_at": None,
            })

        strengths = _values(slots, "key_strengths")
        authoritative = list(semantic_state["fact_conflict"]["authoritative_values"])
        fact_values = list(dict.fromkeys([*strengths, *authoritative]))
        facts = [{
            "id": f"fact_user_value_{index:02d}",
            "name": f"사용자 제공 정보 {index}",
            "value": value,
            "unit": None,
            "source_refs": [source_id],
        } for index, value in enumerate(fact_values, start=1)]
        fact_ids = {fact["value"]: fact["id"] for fact in facts}
        features = [{
            "id": f"feature_strength_{index:02d}",
            "name": value,
            "description": value,
            "fact_ids": [fact_ids[value]],
            "evidence_ids": [],
            "source_refs": [source_id],
        } for index, value in enumerate(strengths, start=1)]
        audiences = [{
            "id": f"aud_target_{index:02d}",
            "description": value,
            "source_refs": [source_id],
        } for index, value in enumerate(_values(slots, "target_supporters"), start=1)]
        problems = [{
            "id": f"problem_context_{index:02d}",
            "description": value,
            "source_refs": [source_id],
        } for index, value in enumerate(_values(slots, "problem_context"), start=1)]

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
        claim_values = list(dict.fromkeys([
            *_values(slots, "trust_elements"),
            *_values(slots, "maker_team_intro"),
            *_values(slots, "funding_plan"),
            *_values(slots, "platform_choice"),
            *_values(slots, "risk_response"),
            *non_iso_schedule,
        ]))
        claims = [{
            "id": f"claim_maker_statement_{index:02d}",
            "statement": value,
            "status": "maker-stated-unverified",
            "evidence_ids": [],
            "source_refs": [source_id],
        } for index, value in enumerate(claim_values, start=1)]

        rewards = [{
            "id": f"reward_user_value_{index:02d}",
            "name": value,
            "price_krw": None,
            "quantity": None,
            "components": [value],
            "source_refs": [source_id],
        } for index, value in enumerate(_values(slots, "rewards"), start=1)]
        schedule_values = {
            "funding_end": _iso_date(raw_funding_end),
            "shipping_start": _iso_date(raw_shipping_start),
            "refund_policy": _first(slots, "refund_policy"),
            "as_policy": _first(slots, "as_policy"),
        }
        assets: list[dict[str, Any]] = []
        if request.image_path is not None:
            superseded = semantic_state["fact_conflict"]["status"] == "superseded-resolved"
            assets.append({
                "id": "asset_product_reference",
                "asset_type": "product",
                "description": (
                    "사용자가 대체 대상으로 명시한 참조 이미지"
                    if superseded else "사용자가 첨부한 제품 참조 이미지"
                ),
                "allowed_sections": [] if superseded else ["hero", "solution", "features"],
                "source_refs": ["source_product_image"],
            })

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

        summary_parts = list(dict.fromkeys([
            _first(slots, "product_name") or "",
            *strengths,
            *_values(slots, "target_supporters"),
        ]))
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
        brief: dict[str, Any],
        caller_id: str,
        idempotency_key: str,
        reference_image_path: Path | None,
    ) -> dict[str, Any]:
        arguments = {"request": {
            "caller_id": caller_id,
            "idempotency_key": idempotency_key,
            "brief": brief,
            "template_id": None,
            "reference_image_path": str(reference_image_path) if reference_image_path else None,
        }}
        async with Client(self.server) as client:
            tools = await client.list_tools()
            if [tool.name for tool in tools] != ["create_crowdfunding_story"]:
                raise RuntimeError("Worker MCP allowlist must contain only story creation")
            result = await client.call_tool("create_crowdfunding_story", arguments)
            if result.structured_content is None:
                raise RuntimeError("Story generation tool returned no structured content")
            return dict(result.structured_content)


def graph_state_to_semantic_state(
    *, input_id: str, state: dict[str, Any]
) -> dict[str, Any]:
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
            "ready_to_confirm": not missing_required_fields(state["facts"]),
            "requested_fields": [],
            "follow_up_question": None,
        },
    }


class StoryMakerWorker:
    """Stateful conversation graph; all generation crosses the MCP boundary."""

    def __init__(
        self,
        *,
        repository: DataRepository,
        conversation_model: ConversationModel,
        brief_builder: BriefBuilder,
        generation_tool: GenerationTool,
        checkpointer: Any | None = None,
        checkpoint_connection: sqlite3.Connection | None = None,
    ) -> None:
        self.repository = repository
        self.conversation_model = conversation_model
        self.brief_builder = brief_builder
        self.generation_tool = generation_tool
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
        if stage == "submitted":
            return "submitted"
        if stage == "cancelled":
            return "closed"
        return "awaiting-input"

    @staticmethod
    def _outcome(
        request: WorkerRequest, state: dict[str, Any], *, trigger: str | None = None
    ) -> WorkerOutcome:
        stage = str(state.get("workflow_stage", "collecting"))
        reply = str(state.get("reply", ""))
        questions = (reply,) if reply and stage in {"collecting", "awaiting-approval"} else ()
        return WorkerOutcome(
            thread_id=request.thread_id,
            input_id=request.input_id,
            status=StoryMakerWorker._status(stage),
            stage=stage,
            reply=reply,
            requested_fields=tuple(state.get("requested_fields", [])),
            questions=questions,
            facts=dict(state.get("facts", {})),
            current_summary=state.get("current_summary"),
            summary_version=int(state.get("summary_version", 0)),
            approved_summary_version=state.get("approved_summary_version"),
            generation_start_trigger=trigger,
            tool_result=state.get("tool_result"),
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
        state = self.conversation_graph.invoke(graph_input, config)
        if state.get("workflow_stage") != "approved":
            return self._outcome(request, state)

        if state.get("approved_summary_version") != state.get("summary_version"):
            raise ValueError("Approved summary version does not match the current summary")
        semantic_state = graph_state_to_semantic_state(input_id=request.input_id, state=state)
        self.repository.validate_intake_semantic_state(semantic_state)
        brief = self.brief_builder.build(request, semantic_state)
        reference = (
            None
            if semantic_state["fact_conflict"]["status"] == "superseded-resolved"
            else Path(state["image_path"]) if state.get("image_path") else None
        )
        summary_version = int(state["approved_summary_version"])
        idempotency_key = request.idempotency_key or (
            f"worker-{request.thread_id}-summary-{summary_version}"
        )
        tool_result = await self.generation_tool.create(
            brief=brief,
            caller_id=request.caller_id,
            idempotency_key=idempotency_key,
            reference_image_path=reference,
        )
        self.conversation_graph.update_state(
            config,
            {
                "workflow_stage": "submitted",
                "tool_result": tool_result,
                "reply": "스토리 생성 요청을 제출했습니다.",
                "requested_fields": [],
            },
            as_node="approval_guard",
        )
        submitted = dict(self.conversation_graph.get_state(config).values)
        return self._outcome(request, submitted, trigger="explicit-confirmation")

    def get_state(self, thread_id: str) -> dict[str, Any]:
        return dict(self.conversation_graph.get_state(self._config(thread_id)).values)

    def delete_thread(self, thread_id: str) -> None:
        self.checkpointer.delete_thread(thread_id)

    def close(self) -> None:
        if self.checkpoint_connection is not None:
            self.checkpoint_connection.close()


def build_live_worker(
    *,
    server_url: str = "http://127.0.0.1:8765/mcp",
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
        repository=repository,
        conversation_model=GeminiConversationModel(adapter),
        brief_builder=GroundedBriefBuilder(repository=repository),
        generation_tool=FastMcpGenerationTool(server_url),
        checkpointer=checkpointer,
        checkpoint_connection=connection,
    )
