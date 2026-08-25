from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol

from dotenv import load_dotenv
from fastmcp import Client

from .adapter import GeminiAdapter
from .data_repository import DataRepository
from .intake import StoryIntakeState, build_intake_graph
from .smoke import build_runtime

WorkerStatus = Literal["awaiting-input", "ready", "submitted"]

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
    input_id: str
    initial_message: str
    followup_messages: tuple[str, ...] = ()
    image_path: Path | None = None
    prior_semantic_state: dict[str, Any] | None = None
    skip_requested: bool = False
    confirmed: bool = False
    caller_id: str = "local-story-worker"
    idempotency_key: str = ""


@dataclass(frozen=True, slots=True)
class WorkerOutcome:
    input_id: str
    status: WorkerStatus
    stage: str
    requested_fields: tuple[str, ...]
    questions: tuple[str, ...]
    semantic_state: dict[str, Any]
    generation_start_trigger: str | None = None
    tool_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_id": self.input_id,
            "status": self.status,
            "stage": self.stage,
            "requested_fields": list(self.requested_fields),
            "questions": list(self.questions),
            "semantic_state": self.semantic_state,
            "generation_start_trigger": self.generation_start_trigger,
            "tool_result": self.tool_result,
        }


class SemanticExtractor(Protocol):
    def extract(self, request: WorkerRequest) -> dict[str, Any]: ...


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


def _conversation_text(request: WorkerRequest) -> str:
    turns = [f"initial: {request.initial_message.strip()}"]
    turns.extend(
        f"followup_{index}: {message.strip()}"
        for index, message in enumerate(request.followup_messages, start=1)
    )
    return "\n".join(turns)


def validate_worker_request(request: WorkerRequest) -> None:
    if not request.input_id.strip():
        raise ValueError("input_id must not be empty")
    messages = (request.initial_message, *request.followup_messages)
    for index, message in enumerate(messages):
        if len(message) > MAX_MESSAGE_CHARS:
            label = "initial_message" if index == 0 else f"followup_messages[{index - 1}]"
            raise ValueError(f"{label} must be at most {MAX_MESSAGE_CHARS:,} characters")
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


class GeminiSemanticExtractor:
    """Use the dialogue model for both semantic extraction and next-question choice."""

    def __init__(self, *, repository: DataRepository, adapter: GeminiAdapter) -> None:
        self.repository = repository
        self.adapter = adapter

    def extract(self, request: WorkerRequest) -> dict[str, Any]:
        prior = request.prior_semantic_state or {}
        prompt = f"""당신은 크라우드펀딩 스토리 작성 대화 에이전트입니다.
사용자 대화에서 사실 슬롯을 추출하고, 지금 확인할 가치가 가장 큰 후속 질문을
직접 결정하세요. 출력은 JSON Schema만 따릅니다.

규칙:
- 사용자가 직접 말한 내용만 values에 넣습니다. 언급되지 않은 값은 unknown입니다.
- 사용자가 없다고 명시한 값만 explicitly-absent입니다.
- 이미지는 색·형태·보이는 구성 등 직접 관찰 가능한 외형만 근거가 됩니다. 성능,
  인증, 내부 구조, 앱 기능, 팀 경력은 이미지에서 추정하지 않습니다.
- 최신 입력이 이전 값을 명시적으로 교체했다면 superseded-resolved로 기록합니다.
- 제품명, 제품 유형, 카테고리, 핵심 강점, 주요 대상이 스토리 방향을 결정할 만큼
  확보되면 ready_to_confirm=true로 둡니다. 나머지 슬롯은 선택 정보이며 무조건
  질문하지 않습니다.
- 정보가 부족하면 템플릿이나 고정 제품 프로필이 아니라 현재 대화 맥락을 보고 한
  번에 하나 또는 서로 밀접한 여러 항목을 자연스러운 한 질문으로 물을 수 있습니다.
- 질문을 할 때 requested_fields와 follow_up_question을 함께 채웁니다.
- 대화 언어에 따라 language를 ko, en, ja, zh 중 하나로 선택합니다.
- funding_end와 shipping_start는 사용자가 정확한 날짜를 제공한 경우 YYYY-MM-DD로
  정규화하고, 상대 일정만 제공했다면 원문 의미를 유지합니다.

input_id: {request.input_id}
이전 의미 상태(참고용이며 최신 대화가 우선):
{json.dumps(prior, ensure_ascii=False)}

전체 대화:
{_conversation_text(request)}
"""
        images = _image_payload(request.image_path)
        result = self.adapter.generate_multimodal_json(
            prompt=prompt,
            images=images,
            response_schema=self.repository.intake_semantic_state_schema(),
        )
        value = result.data
        value.update(
            {
                "schema_version": "story-intake-semantic-state-v2",
                "input_id": request.input_id,
                "image_attached": bool(images),
            }
        )
        self.repository.validate_intake_semantic_state(value)
        return value


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


def semantic_state_to_intake(
    request: WorkerRequest, semantic_state: dict[str, Any]
) -> StoryIntakeState:
    decision = semantic_state["decision"]
    return {
        "initial_message": request.initial_message,
        "agent_ready_to_confirm": bool(decision["ready_to_confirm"]),
        "agent_question": decision["follow_up_question"],
        "agent_requested_fields": list(decision["requested_fields"]),
        "skip_remaining_questions": request.skip_requested,
        "confirmed": request.confirmed,
    }


_CONFIRMATION_QUESTIONS = {
    "ko": "정리된 제품 정보로 스토리를 생성할까요?",
    "en": "Would you like me to generate the story from the information above?",
    "ja": "整理した製品情報からストーリーを生成しますか？",
    "zh": "要根据以上产品信息生成故事吗？",
}


class StoryMakerWorker:
    """Conversation agent; all generation crosses the MCP boundary."""

    def __init__(
        self,
        *,
        repository: DataRepository,
        extractor: SemanticExtractor,
        brief_builder: BriefBuilder,
        generation_tool: GenerationTool,
    ) -> None:
        self.repository = repository
        self.extractor = extractor
        self.brief_builder = brief_builder
        self.generation_tool = generation_tool
        self.intake_graph = build_intake_graph()

    async def handle(self, request: WorkerRequest) -> WorkerOutcome:
        validate_worker_request(request)
        semantic_state = self.extractor.extract(request)
        self.repository.validate_intake_semantic_state(semantic_state)
        decision = semantic_state["decision"]
        if semantic_state["fact_conflict"]["status"] == "unresolved":
            question = decision["follow_up_question"] or (
                "충돌하는 값 중 최종 사실로 사용할 값을 확인해 주세요."
            )
            return WorkerOutcome(
                request.input_id, "awaiting-input", "conflict-resolution",
                ("fact_conflict_resolution",), (question,), semantic_state,
            )

        routed = self.intake_graph.invoke(semantic_state_to_intake(request, semantic_state))
        stage = str(routed["stage"])
        requested = tuple(routed.get("requested_fields", []))
        if stage != "ready-to-generate":
            question = routed.get("question")
            if stage == "confirmation":
                question = _CONFIRMATION_QUESTIONS[semantic_state["language"]]
            if not isinstance(question, str) or not question.strip():
                raise ValueError("The conversation agent did not provide a follow-up question")
            return WorkerOutcome(
                request.input_id, "awaiting-input", stage, requested,
                (question,), semantic_state,
            )

        trigger = str(routed["generation_start_trigger"])
        brief = self.brief_builder.build(request, semantic_state)
        reference = (
            None
            if semantic_state["fact_conflict"]["status"] == "superseded-resolved"
            else request.image_path
        )
        tool_result = await self.generation_tool.create(
            brief=brief,
            caller_id=request.caller_id,
            idempotency_key=request.idempotency_key or f"worker-{request.input_id}-v2",
            reference_image_path=reference,
        )
        return WorkerOutcome(
            request.input_id, "submitted", stage, (), (), semantic_state,
            trigger, tool_result,
        )


def build_live_worker(
    *, server_url: str = "http://127.0.0.1:8765/mcp", root: Path | None = None
) -> StoryMakerWorker:
    load_dotenv()
    repository = DataRepository(root)
    adapter = GeminiAdapter(build_runtime())
    return StoryMakerWorker(
        repository=repository,
        extractor=GeminiSemanticExtractor(repository=repository, adapter=adapter),
        brief_builder=GroundedBriefBuilder(repository=repository),
        generation_tool=FastMcpGenerationTool(server_url),
    )
