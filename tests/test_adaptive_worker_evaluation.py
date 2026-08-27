from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from funding_story_ai.conversation import (
    ApprovalDecision,
    QuestionPlan,
    StorySummary,
    TurnUnderstanding,
    missing_required_fields,
    provided_facts,
)
from funding_story_ai.data_repository import DataRepository
from funding_story_ai.worker import StoryMakerWorker, WorkerRequest

FIXTURE = Path(__file__).parent / "fixtures" / "adaptive-worker-evaluation.json"


class _FixtureModel:
    def __init__(self, turns: dict[str, dict[str, Any]]) -> None:
        self.turns = turns

    def understand_turn(self, *, message, messages, facts, image_path):
        return TurnUnderstanding.model_validate(self.turns[message["content"]]["understanding"])

    def plan_questions(
        self,
        *,
        messages,
        facts,
        missing_required_fields,
        asked_topics,
        turn_understanding,
    ):
        requested = missing_required_fields[:2]
        return QuestionPlan(
            requested_fields=requested,
            question=f"{', '.join(requested)} 정보를 알려주세요.",
            rationale="현재 누락된 필수 정보를 우선합니다.",
        )

    def build_summary(self, *, messages, facts):
        confirmed = provided_facts(facts)
        return StorySummary(
            headline="현재 결정 내용",
            confirmed_facts=confirmed,
            unconfirmed_fields=[
                field for field, value in facts.items() if value["status"] != "provided"
            ],
            summary_text=" / ".join(
                value for values in confirmed.values() for value in values
            ),
            confirmation_question="이 내용으로 스토리를 생성할까요?",
        )

    def classify_approval(self, *, message, summary, messages):
        return ApprovalDecision.model_validate(
            self.turns[message["content"]]["approval_decision"]
        )


class _BriefBuilder:
    def build(self, request, semantic_state):
        return DataRepository().load_brief()


class _Tool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return {"run_id": f"run-{len(self.calls)}", "status": "accepted"}


def test_adaptive_worker_node_and_flow_metrics() -> None:
    dataset = json.loads(FIXTURE.read_text(encoding="utf-8"))
    metrics = {
        "turn_patch_total": 0,
        "turn_patch_correct": 0,
        "required_check_total": 0,
        "required_check_correct": 0,
        "question_total": 0,
        "question_relevant": 0,
        "question_repeated_provided": 0,
        "summary_total": 0,
        "summary_grounded": 0,
        "approval_total": 0,
        "approval_route_correct": 0,
        "transition_total": 0,
        "transition_correct": 0,
        "revision_total": 0,
        "revision_latest_value_correct": 0,
        "unauthorized_tool_calls": 0,
        "authorized_tool_calls": 0,
        "scenario_total": len(dataset["scenarios"]),
        "scenario_completed": 0,
    }

    for scenario in dataset["scenarios"]:
        turns = {turn["message"]: turn for turn in scenario["turns"]}
        tool = _Tool()
        worker = StoryMakerWorker(
            repository=DataRepository(),
            conversation_model=_FixtureModel(turns),
            brief_builder=_BriefBuilder(),
            generation_tool=tool,
            checkpointer=InMemorySaver(),
        )
        previous_tool_calls = 0
        previous_node_visits = 0
        for index, turn in enumerate(scenario["turns"], start=1):
            outcome = asyncio.run(
                worker.handle(
                    WorkerRequest(
                        thread_id=scenario["id"],
                        input_id=scenario["id"],
                        message=turn["message"],
                        message_id=f"{scenario['id']}-{index}",
                    )
                )
            )
            state = worker.get_state(scenario["id"])
            new_nodes = state["visited_nodes"][previous_node_visits:]
            previous_node_visits = len(state["visited_nodes"])

            metrics["transition_total"] += 1
            metrics["transition_correct"] += int(outcome.stage == turn["expected_stage"])
            if "check_required_fields" in new_nodes:
                metrics["required_check_total"] += 1
                metrics["required_check_correct"] += int(
                    missing_required_fields(outcome.facts) == turn["expected_missing"]
                )

            understanding = turn.get("understanding")
            if understanding:
                for patch in understanding["fact_patches"]:
                    metrics["turn_patch_total"] += 1
                    current = outcome.facts[patch["field"]]
                    expected_status = {
                        "replace": "provided",
                        "append": "provided",
                        "mark_absent": "explicitly-absent",
                        "clear": "unknown",
                    }[patch["operation"]]
                    values_match = (
                        current["values"] == patch["values"]
                        if patch["operation"] != "append"
                        else set(patch["values"]).issubset(current["values"])
                    )
                    metrics["turn_patch_correct"] += int(
                        current["status"] == expected_status and values_match
                    )

            if "plan_next_questions" in new_nodes:
                metrics["question_total"] += 1
                requested = set(outcome.requested_fields)
                missing = set(turn["expected_missing"])
                metrics["question_relevant"] += int(bool(requested.intersection(missing)))
                provided = set(provided_facts(outcome.facts))
                metrics["question_repeated_provided"] += len(requested.intersection(provided))

            if "build_summary" in new_nodes:
                metrics["summary_total"] += 1
                summary = StorySummary.model_validate(outcome.current_summary)
                metrics["summary_grounded"] += int(
                    summary.confirmed_facts == provided_facts(outcome.facts)
                )

            if turn.get("approval_decision"):
                metrics["approval_total"] += 1
                decision = turn["approval_decision"]["decision"]
                expected_stage = {
                    "approve": "submitted",
                    "revise": "awaiting-approval",
                    "reject": "collecting",
                    "ambiguous": "awaiting-approval",
                }[decision]
                metrics["approval_route_correct"] += int(outcome.stage == expected_stage)

            if turn.get("expected_facts"):
                for field, values in turn["expected_facts"].items():
                    metrics["revision_total"] += 1
                    metrics["revision_latest_value_correct"] += int(
                        outcome.facts[field]["values"] == values
                    )

            expected_tool_calls = turn["expected_tool_calls"]
            actual_tool_calls = len(tool.calls)
            if expected_tool_calls == previous_tool_calls:
                metrics["unauthorized_tool_calls"] += max(
                    0, actual_tool_calls - previous_tool_calls
                )
            else:
                metrics["authorized_tool_calls"] += actual_tool_calls - previous_tool_calls
            previous_tool_calls = actual_tool_calls
            assert actual_tool_calls == expected_tool_calls
            if "expected_summary_version" in turn:
                assert outcome.summary_version == turn["expected_summary_version"]

        metrics["scenario_completed"] += 1

    assert metrics == {
        "turn_patch_total": 21,
        "turn_patch_correct": 21,
        "required_check_total": 6,
        "required_check_correct": 6,
        "question_total": 1,
        "question_relevant": 1,
        "question_repeated_provided": 0,
        "summary_total": 5,
        "summary_grounded": 5,
        "approval_total": 6,
        "approval_route_correct": 6,
        "transition_total": 12,
        "transition_correct": 12,
        "revision_total": 1,
        "revision_latest_value_correct": 1,
        "unauthorized_tool_calls": 0,
        "authorized_tool_calls": 3,
        "scenario_total": 4,
        "scenario_completed": 4,
    }
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
