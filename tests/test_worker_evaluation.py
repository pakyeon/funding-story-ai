from pathlib import Path

from funding_story_ai.config import RuntimeSettings
from funding_story_ai.conversation import (
    FACT_FIELDS,
    REQUIRED_FACT_FIELDS,
    ApprovalDecision,
    CollectionDirective,
    FactPatch,
    QuestionPlan,
    StorySummary,
    TurnUnderstanding,
    explicitly_absent_fields,
    provided_facts,
    skipped_optional_fields,
)
from funding_story_ai.worker_evaluation import FlowCase, FlowDataset, FlowTurn, evaluate_flow

DATASET = (
    Path(__file__).parents[1]
    / "evals"
    / "datasets"
    / "story-worker-flow-evaluation-v1.json"
)


class _FlowModel:
    def __init__(self, adapter) -> None:
        pass

    def understand_turn(self, *, message, messages, facts, image_path):
        content = message["content"]
        if "생략" in content:
            return TurnUnderstanding(
                intent="provide_information",
                collection_directive=CollectionDirective(action="skip_all_optional"),
            )
        return TurnUnderstanding(
            intent="provide_information",
            fact_patches=[
                FactPatch(field=field, operation="replace", values=[f"{field} 값"])
                for field in FACT_FIELDS
            ],
        )

    def plan_questions(
        self,
        *,
        purpose,
        candidate_fields,
        requested_group,
        requested_detail,
        **kwargs,
    ):
        return QuestionPlan(
            purpose=purpose,
            requested_fields=candidate_fields[:3],
            requested_group=requested_group,
            requested_detail=requested_detail,
            question=requested_detail,
            rationale="회귀 테스트",
        )

    repair_question_plan = plan_questions

    def build_summary(self, *, facts, optional_collection, **kwargs):
        return StorySummary(
            headline="입력 요약",
            confirmed_facts=provided_facts(facts),
            explicitly_absent_fields=explicitly_absent_fields(facts),
            skipped_fields=skipped_optional_fields(optional_collection),
            summary_text="확정된 입력을 요약했습니다.",
            confirmation_question="이 내용으로 스토리를 생성할까요?",
        )

    def classify_approval(self, **kwargs):
        return ApprovalDecision(decision="approve", reason="회귀 테스트")


class _QuestionThenApprovalModel:
    """Provide one real question before approval to exercise question-history accounting."""

    def __init__(self, adapter) -> None:
        pass

    def understand_turn(self, *, message, messages, facts, image_path):
        content = message["content"]
        if "전체 생략" in content:
            return TurnUnderstanding(
                intent="provide_information",
                collection_directive=CollectionDirective(action="skip_all_optional"),
            )
        if "초기 제품명" in content:
            return TurnUnderstanding(
                intent="provide_information",
                fact_patches=[
                    FactPatch(field="product_name", operation="replace", values=["클린포지 R1"])
                ],
            )
        return TurnUnderstanding(
            intent="provide_information",
            fact_patches=[
                FactPatch(field=field, operation="replace", values=[f"{field} 값"])
                for field in REQUIRED_FACT_FIELDS
            ],
        )

    def plan_questions(
        self,
        *,
        purpose,
        candidate_fields,
        requested_group,
        requested_detail,
        **kwargs,
    ):
        return QuestionPlan(
            purpose=purpose,
            requested_fields=candidate_fields[:3],
            requested_group=requested_group,
            requested_detail=requested_detail,
            question=f"{requested_detail} 알려주세요.",
            rationale="질문 이력 회귀 테스트",
        )

    repair_question_plan = plan_questions

    def build_summary(self, *, facts, optional_collection, **kwargs):
        return StorySummary(
            headline="입력 요약",
            confirmed_facts=provided_facts(facts),
            explicitly_absent_fields=explicitly_absent_fields(facts),
            skipped_fields=skipped_optional_fields(optional_collection),
            summary_text="확정된 입력을 요약했습니다.",
            confirmation_question="이 내용으로 스토리를 생성할까요?",
        )

    def classify_approval(self, **kwargs):
        return ApprovalDecision(decision="approve", reason="회귀 테스트")


def test_flow_evaluation_checkpoint_resumes_completed_sessions(monkeypatch, tmp_path) -> None:
    dataset = FlowDataset.model_validate_json(DATASET.read_text(encoding="utf-8"))
    dataset = dataset.model_copy(update={"cases": dataset.cases[:50]})
    monkeypatch.setattr("funding_story_ai.worker_evaluation.GeminiConversationModel", _FlowModel)
    checkpoint = tmp_path / "flow.jsonl"
    settings = RuntimeSettings(project_id="test-project")

    first = evaluate_flow(
        dataset,
        settings,
        2,
        checkpoint_path=checkpoint,
        requests_per_minute=60,
    )
    assert first["metrics"]["case_count"] == 50
    assert len(checkpoint.read_text(encoding="utf-8").splitlines()) == 51

    resumed = evaluate_flow(
        dataset,
        settings,
        2,
        checkpoint_path=checkpoint,
        resume=True,
        requests_per_minute=60,
    )
    assert resumed["metrics"] == first["metrics"]


def test_flow_evaluation_does_not_count_approval_request_as_a_new_question(monkeypatch) -> None:
    dataset = FlowDataset(
        schema_version="story-worker-flow-evaluation-v1",
        cases=[
            FlowCase(
                id=f"question-before-approval-{index:03d}",
                scenario="question-before-approval",
                turns=[
                    FlowTurn(
                        message="초기 제품명만 제공",
                        expected_stage="collecting",
                        expected_phase="required",
                    ),
                    FlowTurn(
                        message="필수 정보 모두 제공",
                        expected_stage="collecting",
                        expected_phase="optional-offer",
                    ),
                    FlowTurn(
                        message="선택 정보 전체 생략",
                        expected_stage="awaiting-approval",
                        expected_phase="approval",
                    ),
                    FlowTurn(
                        message="이 내용으로 생성해줘",
                        expected_stage="generation-ready",
                        expected_phase="generation-ready",
                    ),
                ],
            )
            for index in range(50)
        ],
    )
    monkeypatch.setattr(
        "funding_story_ai.worker_evaluation.GeminiConversationModel",
        _QuestionThenApprovalModel,
    )

    result = evaluate_flow(
        dataset,
        RuntimeSettings(project_id="test-project"),
        1,
        requests_per_minute=60,
    )

    assert result["metrics"]["question_output_count"] == 50
    assert result["metrics"]["abnormal_repeated_question_count"] == 0
