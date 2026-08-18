import asyncio
import json
from copy import deepcopy
from typing import Any

from funding_story_ai.data_repository import DataRepository
from funding_story_ai.worker import (
    GroundedBriefBuilder,
    StoryMakerWorker,
    WorkerRequest,
    semantic_state_to_intake,
)


def _slot(status: str, values: list[str]) -> dict[str, Any]:
    return {"status": status, "values": values, "source_turn": "initial"}


def _state(*, confirmed: bool = False) -> dict[str, Any]:
    return {
        "schema_version": "story-intake-semantic-state-v1",
        "input_id": "worker-case",
        "profile_id": "robot-vacuum-ko-v1",
        "image_attached": True,
        "slots": {
            "product_identity": _slot("provided", ["OrbitClean V3", "로봇청소기"]),
            "key_strengths": _slot("provided", ["얇은 본체", "자동 먼지 비움"]),
            "target_supporters": _slot("provided", ["가구 아래 청소가 필요한 사용자"]),
            "problem_context": _slot("provided", ["가구 아래 반복 청소"]),
            "trust_elements": _slot("explicitly-absent", []),
            "maker_team_intro": _slot("explicitly-absent", []),
        },
        "fact_conflict": {
            "status": "none",
            "authoritative_values": [],
            "superseded_values": [],
        },
        "turn_state": {
            "primary_answered": False,
            "combined_answered": False,
            "secondary_answered": False,
            "skip_requested": False,
            "confirmed": confirmed,
        },
    }


class _Extractor:
    def __init__(self, value: dict[str, Any]) -> None:
        self.value = value

    def extract(self, request, profile):
        value = deepcopy(self.value)
        value["input_id"] = request.input_id
        return value


class _BriefBuilder:
    def __init__(self, repository: DataRepository) -> None:
        self.repository = repository
        self.calls = 0

    def build(self, request, semantic_state, profile):
        self.calls += 1
        return self.repository.load_brief()


class _Tool:
    def __init__(self) -> None:
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"run_id": "run-test", "status": "completed"}


def test_explicit_absence_counts_as_answer_and_routes_to_confirmation() -> None:
    request = WorkerRequest(input_id="worker-case", initial_message="스토리를 만들어 줘")
    intake = semantic_state_to_intake(request, _state())
    assert intake["primary_semantic_complete"] is True
    assert intake["secondary_semantic_complete"] is True
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
        extractor=_Extractor(_state(confirmed=True)),
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


def test_grounded_brief_preserves_input_without_domain_expansion() -> None:
    repository = DataRepository()
    state = _state(confirmed=True)
    state["slots"]["key_strengths"]["values"] = [
        "최대 6,800Pa",
        "먼지 비움과 충전만 지원하는 도크",
    ]
    state["fact_conflict"] = {
        "status": "superseded-resolved",
        "authoritative_values": ["앱 미지원", "음성 제어 미지원"],
        "superseded_values": ["앱·음성 제어"],
    }
    brief = GroundedBriefBuilder(repository=repository).build(
        WorkerRequest(input_id="robot-grounded", initial_message="최신 사양만 사용해줘"),
        state,
        repository.get_category_profile("robot-vacuum-ko-v1"),
    )
    serialized = json.dumps(brief, ensure_ascii=False)
    assert "6,800Pa" in serialized
    assert "앱 미지원" in serialized
    assert "카펫" not in serialized
    assert "문턱" not in serialized
    repository.validate_story_brief(brief)
