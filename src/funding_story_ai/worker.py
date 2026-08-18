from __future__ import annotations

import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from dotenv import load_dotenv
from fastmcp import Client

from .adapter import GeminiAdapter
from .data_repository import DataRepository
from .intake import StoryIntakeState, build_intake_graph, question_prompt
from .smoke import build_runtime

WorkerStatus = Literal["awaiting-input", "ready", "submitted"]


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    input_id: str
    initial_message: str
    followup_messages: tuple[str, ...] = ()
    image_path: Path | None = None
    profile_id: str = "robot-vacuum-ko-v1"
    primary_answered: bool = False
    combined_answered: bool = False
    secondary_answered: bool = False
    skip_requested: bool = False
    confirmed: bool = False
    caller_id: str = "local-story-worker"
    request_id: str = ""
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
    def extract(self, request: WorkerRequest, profile: dict[str, Any]) -> dict[str, Any]: ...


class BriefBuilder(Protocol):
    def build(
        self,
        request: WorkerRequest,
        semantic_state: dict[str, Any],
        profile: dict[str, Any],
    ) -> dict[str, Any]: ...


class GenerationTool(Protocol):
    async def create(
        self,
        *,
        brief: dict[str, Any],
        profile_id: str,
        caller_id: str,
        request_id: str,
        idempotency_key: str,
        reference_image_path: Path | None,
    ) -> dict[str, Any]: ...


def _conversation_text(request: WorkerRequest) -> str:
    turns = [f"initial: {request.initial_message}"]
    turns.extend(
        f"followup_{index}: {message}"
        for index, message in enumerate(request.followup_messages, start=1)
    )
    return "\n".join(turns)


def _image_payload(path: Path | None) -> list[tuple[bytes, str]]:
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if not mime_type.startswith("image/"):
        raise ValueError(f"Worker image must have an image media type: {path}")
    return [(path.read_bytes(), mime_type)]


class GeminiSemanticExtractor:
    """Extract generic slots while separating visible appearance from product facts."""

    def __init__(self, *, repository: DataRepository, adapter: GeminiAdapter) -> None:
        self.repository = repository
        self.adapter = adapter

    def extract(self, request: WorkerRequest, profile: dict[str, Any]) -> dict[str, Any]:
        prompt = f"""당신은 크라우드펀딩 스토리 대화의 의미 슬롯 추출기입니다.
출력은 제공된 JSON Schema만 따릅니다. 사용자가 직접 말한 사실만 values에 넣고,
없다고 명시하면 explicitly-absent, 그 외에는 unknown으로 둡니다. 이미지에서는 색상,
형태, 보이는 구성처럼 직접 관찰 가능한 외형만 추출할 수 있습니다. 성능, 인증,
내부 구조, 앱 기능, 팀 경력은 이미지로 추정하지 마세요. 최신 입력이 이전 값을
명시적으로 교체하면 superseded-resolved로 기록하고 두 값을 분리하세요.

input_id: {request.input_id}
profile_id: {request.profile_id}
카테고리별 추출·질문 힌트(사실이 아님):
{json.dumps(profile['semantic_slot_guidance'], ensure_ascii=False)}

대화:
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
                "schema_version": "story-intake-semantic-state-v1",
                "input_id": request.input_id,
                "profile_id": request.profile_id,
                "image_attached": bool(images),
                "turn_state": {
                    "primary_answered": request.primary_answered,
                    "combined_answered": request.combined_answered,
                    "secondary_answered": request.secondary_answered,
                    "skip_requested": request.skip_requested,
                    "confirmed": request.confirmed,
                },
            }
        )
        self.repository.validate_intake_semantic_state(value)
        return value


def _brief_id(input_id: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "-", input_id.lower()).strip("-_")
    return normalized if len(normalized) >= 2 else f"input-{normalized or 'unknown'}"


class GroundedBriefBuilder:
    """Map extracted slots to a brief without adding domain knowledge."""

    def __init__(self, *, repository: DataRepository) -> None:
        self.repository = repository

    def build(
        self,
        request: WorkerRequest,
        semantic_state: dict[str, Any],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        slots = semantic_state["slots"]
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

        identities = list(slots["product_identity"]["values"])
        strengths = list(slots["key_strengths"]["values"])
        authoritative = list(semantic_state["fact_conflict"]["authoritative_values"])
        fact_values = list(dict.fromkeys([*strengths, *authoritative]))
        facts = [
            {
                "id": f"fact_user_value_{index:02d}",
                "name": f"사용자 확정 정보 {index}",
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
            for index, value in enumerate(slots["target_supporters"]["values"], start=1)
        ]
        problems = [
            {
                "id": f"problem_context_{index:02d}",
                "description": value,
                "source_refs": [source_id],
            }
            for index, value in enumerate(slots["problem_context"]["values"], start=1)
        ]
        claim_values = list(
            dict.fromkeys(
                [
                    *slots["trust_elements"]["values"],
                    *slots["maker_team_intro"]["values"],
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
        summary = list(
            dict.fromkeys([*identities, *strengths, *slots["target_supporters"]["values"]])
        )
        assets = []
        if request.image_path is not None:
            superseded = semantic_state["fact_conflict"]["status"] == "superseded-resolved"
            assets.append(
                {
                    "id": "asset_product_reference",
                    "asset_type": "product",
                    "description": (
                        "사용자가 폐기 또는 대체 대상으로 명시한 참조 이미지"
                        if superseded
                        else "사용자가 첨부한 제품 참조 이미지"
                    ),
                    "allowed_sections": [] if superseded else ["hero", "solution", "features"],
                    "source_refs": ["source_product_image"],
                }
            )
        brief = {
            "schema_version": "story-brief-v1",
            "brief_id": _brief_id(request.input_id),
            "language": "ko",
            "source": {
                "project_id": None,
                "project_url": None,
                "purpose": "maker-brief",
                "snapshot_date": "2026-08-18",
                "refs": refs,
            },
            "product": {
                "name": identities[0],
                "category": profile["category"],
                "product_type": next(
                    (value for value in identities if value in profile["product_types"]),
                    profile["product_types"][0],
                ),
                "summary": " / ".join(summary),
                "facts": facts,
            },
            "audiences": audiences,
            "problems": problems,
            "features": features,
            "claims": claims,
            "evidence": [],
            "assets": assets,
            "rewards": [],
            "schedule_policy": {
                "funding_end": None,
                "shipping_start": None,
                "refund_policy": None,
                "as_policy": None,
                "source_refs": [],
            },
            "unknowns": [
                {
                    "field": field,
                    "question": question,
                    "blocks_sections": sections,
                }
                for field, question, sections in (
                    ("rewards", "확정된 리워드 구성과 가격이 무엇인가요?", ["features", "cta"]),
                    (
                        "schedule_policy",
                        "확정된 발송 일정과 A/S 정책이 무엇인가요?",
                        ["timeline", "risks", "cta"],
                    ),
                    ("funding_plan", "펀딩금 사용 계획이 무엇인가요?", ["funding_plan"]),
                    (
                        "platform_choice",
                        "플랫폼 선택 이유가 무엇인가요?",
                        ["platform_choice"],
                    ),
                    ("risk_response", "생산·공급 리스크와 대응 계획이 무엇인가요?", ["risks"]),
                )
            ],
        }
        self.repository.validate_story_brief(brief)
        return brief


class FastMcpGenerationTool:
    """The worker's only generation capability: one allowlisted MCP tool."""

    def __init__(self, server: str | Any) -> None:
        self.server = server

    async def create(
        self,
        *,
        brief: dict[str, Any],
        profile_id: str,
        caller_id: str,
        request_id: str,
        idempotency_key: str,
        reference_image_path: Path | None,
    ) -> dict[str, Any]:
        arguments = {
            "request": {
                "request_id": request_id,
                "caller_id": caller_id,
                "idempotency_key": idempotency_key,
                "brief": brief,
                "template_id": None,
                "category_profile_id": profile_id,
                "reference_image_path": (
                    str(reference_image_path) if reference_image_path else None
                ),
                "generate_images": True,
            }
        }
        async with Client(self.server) as client:
            tools = await client.list_tools()
            if [tool.name for tool in tools] != ["create_crowdfunding_story"]:
                raise RuntimeError("Worker MCP allowlist must contain only story creation")
            task = await client.call_tool("create_crowdfunding_story", arguments, task=True)
            result = await task.result()
            if result.structured_content is None:
                raise RuntimeError("Story generation tool returned no structured content")
            return dict(result.structured_content)


def semantic_state_to_intake(
    request: WorkerRequest, semantic_state: dict[str, Any]
) -> StoryIntakeState:
    slots = semantic_state["slots"]

    def provided(slot: str) -> bool:
        return slots[slot]["status"] == "provided" and bool(slots[slot]["values"])

    def answered(slot: str) -> bool:
        return slots[slot]["status"] in {"provided", "explicitly-absent"}

    turn = semantic_state["turn_state"]
    return {
        "initial_message": request.initial_message,
        "product_image_attached": semantic_state["image_attached"],
        "key_strengths": list(slots["key_strengths"]["values"]),
        "target_supporters": list(slots["target_supporters"]["values"]),
        "trust_elements": list(slots["trust_elements"]["values"]),
        "maker_team_intro": " / ".join(slots["maker_team_intro"]["values"]) or None,
        "primary_semantic_complete": provided("key_strengths") and provided("target_supporters"),
        "secondary_semantic_complete": answered("trust_elements")
        and answered("maker_team_intro"),
        "primary_answered_explicitly": turn["primary_answered"],
        "combined_answered_explicitly": turn["combined_answered"],
        "secondary_answered_explicitly": turn["secondary_answered"],
        "prefer_combined_question": False,
        "skip_remaining_questions": turn["skip_requested"],
        "confirmed": turn["confirmed"],
    }


class StoryMakerWorker:
    """Conversation agent that can generate only through the MCP boundary."""

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
        profile = self.repository.get_category_profile(request.profile_id)
        semantic_state = self.extractor.extract(request, profile)
        self.repository.validate_intake_semantic_state(semantic_state)
        if semantic_state["fact_conflict"]["status"] == "unresolved":
            return WorkerOutcome(
                request.input_id,
                "awaiting-input",
                "conflict-resolution",
                ("fact_conflict_resolution",),
                ("충돌하는 값 중 최종 사실로 사용할 값을 확인해 주세요.",),
                semantic_state,
            )
        if semantic_state["slots"]["product_identity"]["status"] == "unknown":
            question = profile["semantic_slot_guidance"]["product_identity"][
                "question_examples"
            ][0]
            return WorkerOutcome(
                request.input_id,
                "awaiting-input",
                "identity-required",
                ("product_identity",),
                (question,),
                semantic_state,
            )

        routed = self.intake_graph.invoke(semantic_state_to_intake(request, semantic_state))
        stage = str(routed["stage"])
        requested = tuple(routed.get("requested_fields", []))
        if stage != "ready-to-generate":
            return WorkerOutcome(
                request.input_id,
                "awaiting-input",
                stage,
                requested,
                (question_prompt(stage, profile),),
                semantic_state,
            )

        trigger = str(routed["generation_start_trigger"])
        brief = self.brief_builder.build(request, semantic_state, profile)
        reference = (
            None
            if semantic_state["fact_conflict"]["status"] == "superseded-resolved"
            else request.image_path
        )
        tool_result = await self.generation_tool.create(
            brief=brief,
            profile_id=request.profile_id,
            caller_id=request.caller_id,
            request_id=request.request_id or f"worker-{request.input_id}",
            idempotency_key=request.idempotency_key or f"worker-{request.input_id}-v1",
            reference_image_path=reference,
        )
        return WorkerOutcome(
            request.input_id,
            "submitted",
            stage,
            (),
            (),
            semantic_state,
            trigger,
            tool_result,
        )


def build_live_worker(
    *,
    server_url: str = "http://127.0.0.1:8765/mcp",
    root: Path | None = None,
) -> StoryMakerWorker:
    """Build the conversation worker while keeping generation behind FastMCP."""
    load_dotenv()
    repository = DataRepository(root)
    settings, ledger = build_runtime()
    adapter = GeminiAdapter(settings, ledger)
    return StoryMakerWorker(
        repository=repository,
        extractor=GeminiSemanticExtractor(repository=repository, adapter=adapter),
        brief_builder=GroundedBriefBuilder(repository=repository),
        generation_tool=FastMcpGenerationTool(server_url),
    )
