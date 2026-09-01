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
    explicitly_absent_fields,
    missing_required_fields,
    optional_group_for,
    provided_facts,
    skipped_optional_fields,
)
from funding_story_ai.worker import StoryMakerWorker, WorkerRequest

FIXTURE = Path(__file__).parent / "fixtures" / "optional-collection-flow-evaluation.json"


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
        purpose,
        candidate_fields,
        requested_group,
        requested_detail,
        question_history,
        turn_understanding,
    ):
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
            rationale="현재 수집 단계의 후보만 질문합니다.",
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
        return QuestionPlan(
            purpose=purpose,
            requested_fields=candidate_fields[:3],
            requested_group=requested_group,
            requested_detail=requested_detail,
            question=requested_detail,
            rationale="질문을 제약에 맞게 교정합니다.",
        )

    def build_summary(self, *, messages, facts, optional_collection):
        confirmed = provided_facts(facts)
        return StorySummary(
            headline="현재 결정 내용",
            confirmed_facts=confirmed,
            explicitly_absent_fields=explicitly_absent_fields(facts),
            skipped_fields=skipped_optional_fields(optional_collection),
            summary_text=" / ".join(
                value for values in confirmed.values() for value in values
            ),
            confirmation_question="이 내용으로 스토리를 생성할까요?",
        )

    def classify_approval(self, *, message, summary, messages):
        return ApprovalDecision.model_validate(
            self.turns[message["content"]]["approval_decision"]
        )


def test_optional_collection_orchestration_metrics() -> None:
    dataset = json.loads(FIXTURE.read_text(encoding="utf-8"))
    metrics = {
        "scenario_total": len(dataset["scenarios"]),
        "scenario_completed": 0,
        "transition_total": 0,
        "transition_correct": 0,
        "fact_patch_total": 0,
        "fact_patch_correct": 0,
        "required_check_total": 0,
        "required_check_correct": 0,
        "optional_offer_total": 0,
        "optional_offer_correct": 0,
        "optional_question_total": 0,
        "optional_question_same_group": 0,
        "optional_question_max_three": 0,
        "summary_total": 0,
        "summary_grounded": 0,
        "silent_optional_skip": 0,
        "approval_total": 0,
        "approval_route_correct": 0,
        "generation_ready_total": 0,
        "unauthorized_generation_ready": 0,
    }

    for scenario in dataset["scenarios"]:
        turns = {turn["message"]: turn for turn in scenario["turns"]}
        worker = StoryMakerWorker(
            conversation_model=_FixtureModel(turns),
            checkpointer=InMemorySaver(),
        )
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
            metrics["transition_correct"] += int(
                outcome.stage == turn["expected_stage"]
                and outcome.collection_phase == turn["expected_phase"]
            )
            if "check_required_fields" in new_nodes:
                metrics["required_check_total"] += 1
                metrics["required_check_correct"] += int(
                    missing_required_fields(outcome.facts) == turn["expected_missing"]
                )

            understanding = turn.get("understanding", {})
            for patch in understanding.get("fact_patches", []):
                metrics["fact_patch_total"] += 1
                current = outcome.facts[patch["field"]]
                expected_status = {
                    "replace": "provided",
                    "append": "provided",
                    "mark_absent": "explicitly-absent",
                    "clear": "unknown",
                }[patch["operation"]]
                values_match = (
                    current["values"] == patch.get("values", [])
                    if patch["operation"] != "append"
                    else set(patch["values"]).issubset(current["values"])
                )
                metrics["fact_patch_correct"] += int(
                    current["status"] == expected_status and values_match
                )

            if "offer_optional_information" in new_nodes:
                metrics["optional_offer_total"] += 1
                metrics["optional_offer_correct"] += int(
                    outcome.optional_collection["offered"]
                    and bool(outcome.remaining_optional_fields)
                )

            if "prepare_optional_question" in new_nodes:
                metrics["optional_question_total"] += 1
                groups = {optional_group_for(field) for field in outcome.requested_fields}
                metrics["optional_question_same_group"] += int(
                    groups == {outcome.active_optional_group}
                )
                metrics["optional_question_max_three"] += int(
                    1 <= len(outcome.requested_fields) <= 3
                )

            if "build_summary" in new_nodes:
                metrics["summary_total"] += 1
                summary = StorySummary.model_validate(outcome.current_summary)
                metrics["summary_grounded"] += int(
                    summary.confirmed_facts == provided_facts(outcome.facts)
                    and set(summary.skipped_fields)
                    == {
                        field
                        for field, status in outcome.optional_collection["field_states"].items()
                        if status == "skipped"
                    }
                )
                metrics["silent_optional_skip"] += int(
                    any(
                        status not in {"resolved", "skipped"}
                        for status in outcome.optional_collection["field_states"].values()
                    )
                )

            if turn.get("approval_decision"):
                metrics["approval_total"] += 1
                expected_stage = {
                    "approve": "generation-ready",
                    "revise": "awaiting-approval",
                    "reject": "collecting",
                    "ambiguous": "awaiting-approval",
                }[turn["approval_decision"]["decision"]]
                metrics["approval_route_correct"] += int(outcome.stage == expected_stage)

            if outcome.stage == "generation-ready":
                metrics["generation_ready_total"] += 1
                metrics["unauthorized_generation_ready"] += int(
                    not turn["generation_ready"]
                )
            assert (outcome.stage == "generation-ready") == turn["generation_ready"]
            if "expected_summary_version" in turn:
                assert outcome.summary_version == turn["expected_summary_version"]
            for field, values in turn.get("expected_facts", {}).items():
                assert outcome.facts[field]["values"] == values

        metrics["scenario_completed"] += 1

    assert metrics == {
        "scenario_total": 4,
        "scenario_completed": 4,
        "transition_total": 18,
        "transition_correct": 18,
        "fact_patch_total": 35,
        "fact_patch_correct": 35,
        "required_check_total": 12,
        "required_check_correct": 12,
        "optional_offer_total": 3,
        "optional_offer_correct": 3,
        "optional_question_total": 1,
        "optional_question_same_group": 1,
        "optional_question_max_three": 1,
        "summary_total": 5,
        "summary_grounded": 5,
        "silent_optional_skip": 0,
        "approval_total": 6,
        "approval_route_correct": 6,
        "generation_ready_total": 3,
        "unauthorized_generation_ready": 0,
    }
