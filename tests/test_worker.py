import asyncio
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from funding_story_ai.data_repository import DataRepository
from funding_story_ai.worker import (
    GroundedBriefBuilder,
    StoryMakerWorker,
    WorkerRequest,
    semantic_state_to_intake,
    validate_worker_request,
)


def _slot(status: str = "unknown", values: list[str] | None = None) -> dict[str, Any]:
    return {"status": status, "values": values or [], "source_turn": "initial"}


def _state(*, ready: bool = True) -> dict[str, Any]:
    slots = {
        name: _slot()
        for name in (
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
    }
    slots.update(
        {
            "product_name": _slot("provided", ["OrbitClean V3"]),
            "product_type": _slot("provided", ["로봇청소기"]),
            "category": _slot("provided", ["테크·가전"]),
            "key_strengths": _slot("provided", ["얇은 본체", "자동 먼지 비움"]),
            "target_supporters": _slot("provided", ["가구 아래 청소가 필요한 사용자"]),
            "problem_context": _slot("provided", ["가구 아래 반복 청소"]),
            "trust_elements": _slot("explicitly-absent"),
            "maker_team_intro": _slot("explicitly-absent"),
        }
    )
    return {
        "schema_version": "story-intake-semantic-state-v2",
        "input_id": "worker-case",
        "language": "ko",
        "image_attached": False,
        "slots": slots,
        "fact_conflict": {
            "status": "none",
            "authoritative_values": [],
            "superseded_values": [],
        },
        "decision": {
            "ready_to_confirm": ready,
            "requested_fields": [] if ready else ["key_strengths"],
            "follow_up_question": None if ready else "제품의 핵심 강점은 무엇인가요?",
        },
    }


class _Extractor:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value

    def extract(self, request):
        value = deepcopy(self.value)
        value["input_id"] = request.input_id
        return value


class _BriefBuilder:
    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository
        self.calls = 0

    def build(self, request, semantic_state):
        self.calls += 1
        return self.repository.load_brief()


class _Tool:
    def __init__(self) -> None:
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"run_id": "run-test", "status": "accepted"}


def test_llm_decision_routes_to_confirmation() -> None:
    request = WorkerRequest(input_id="worker-case", initial_message="스토리를 만들어 줘")
    intake = semantic_state_to_intake(request, _state())
    assert intake["agent_ready_to_confirm"] is True
    repository = DataRepository()
    tool = _Tool()
    worker = StoryMakerWorker(
        repository=repository,
        extractor=_Extractor(_state()),
        brief_builder=_BriefBuilder(repository),
        generation_tool=tool,
    )
    outcome = asyncio.run(worker.handle(request))
    assert outcome.stage == "confirmation"
    assert outcome.requested_fields == ("confirmed",)
    assert tool.calls == []


def test_worker_calls_only_mcp_tool_after_confirmation() -> None:
    repository = DataRepository()
    tool = _Tool()
    builder = _BriefBuilder(repository)
    worker = StoryMakerWorker(
        repository=repository,
        extractor=_Extractor(_state()),
        brief_builder=builder,
        generation_tool=tool,
    )
    outcome = asyncio.run(
        worker.handle(
            WorkerRequest(
                input_id="worker-case",
                initial_message="스토리를 만들어 줘",
                confirmed=True,
            )
        )
    )
    assert outcome.status == "submitted"
    assert outcome.generation_start_trigger == "explicit-confirmation"
    assert builder.calls == 1
    assert len(tool.calls) == 1


def test_grounded_brief_preserves_optional_input_without_domain_expansion() -> None:
    repository = DataRepository()
    state = _state()
    state["slots"]["key_strengths"]["values"] = [
        "최대 6,800Pa",
        "먼지 비움과 충전만 지원하는 도크",
    ]
    state["slots"]["rewards"] = _slot("provided", ["본품 얼리버드 1개"])
    state["slots"]["funding_end"] = _slot("provided", ["2026-09-30"])
    state["fact_conflict"] = {
        "status": "superseded-resolved",
        "authoritative_values": ["앱 미지원", "음성 제어 미지원"],
        "superseded_values": ["앱·음성 제어"],
    }
    brief = GroundedBriefBuilder(repository=repository).build(
        WorkerRequest(input_id="robot-grounded", initial_message="최신 사양만 사용해줘"),
        state,
    )
    serialized = json.dumps(brief, ensure_ascii=False)
    assert "6,800Pa" in serialized
    assert "앱 미지원" in serialized
    assert "본품 얼리버드 1개" in serialized
    assert brief["schedule_policy"]["funding_end"] == "2026-09-30"
    assert "카펫" not in serialized
    assert "문턱" not in serialized
    repository.validate_story_brief(brief)


def test_worker_enforces_public_text_and_image_input_limits(tmp_path) -> None:
    try:
        validate_worker_request(
            WorkerRequest(input_id="too-long", initial_message="x" * 1_001)
        )
    except ValueError as exc:
        assert "1,000" in str(exc)
    else:
        raise AssertionError("Expected the 1,000 character limit")

    unsupported = tmp_path / "product.gif"
    unsupported.write_bytes(b"gif")
    try:
        validate_worker_request(
            WorkerRequest(
                input_id="unsupported-image",
                initial_message="product",
                image_path=Path(unsupported),
            )
        )
    except ValueError as exc:
        assert "JPG, PNG, or WEBP" in str(exc)
    else:
        raise AssertionError("Expected the image format limit")
