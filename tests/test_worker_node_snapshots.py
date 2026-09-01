import json
from pathlib import Path

from funding_story_ai.conversation import (
    OPTIONAL_FACT_FIELDS,
    ApprovalDecision,
    ConversationNodes,
    OptionalCollection,
    QuestionPlan,
    StorySummary,
    TurnUnderstanding,
    apply_fact_patches,
    initial_facts,
    initial_optional_collection,
    missing_required_fields,
    question_plan_error,
    validate_summary_grounding,
)

SNAPSHOTS = Path(__file__).parent / "fixtures" / "story-worker-evaluated-node-snapshots.json"


def _snapshots() -> dict:
    return json.loads(SNAPSHOTS.read_text(encoding="utf-8"))["snapshots"]


def test_evaluated_field_and_question_outputs_satisfy_graph_contracts() -> None:
    snapshots = _snapshots()
    understanding = TurnUnderstanding.model_validate(snapshots["field"]["output"])
    message = {
        "id": "snapshot-message",
        "role": "user",
        "content": snapshots["field"]["message"],
    }
    updated = apply_fact_patches(
        {
            "incoming_message": message,
            "messages": [message],
            "facts": initial_facts(),
            "turn_understanding": understanding.model_dump(mode="json"),
        }
    )

    assert missing_required_fields(updated["facts"]) == []

    question = QuestionPlan.model_validate(snapshots["question"]["output"])
    question_state = {
        **snapshots["question"]["state"],
        "facts_revision": 1,
        "collection_revision": 0,
        "question_history": [],
    }
    assert question_plan_error(question_state, question) is None


def test_evaluated_summary_and_approval_outputs_pass_grounding_and_guard() -> None:
    snapshots = _snapshots()
    facts = initial_facts()
    for field, values in snapshots["summary"]["facts"].items():
        facts[field] = {
            "status": "provided",
            "values": values,
            "source_message_ids": ["snapshot-message"],
            "updated_at_turn": 1,
        }
    collection = OptionalCollection.model_validate(initial_optional_collection(facts))
    collection = collection.model_copy(
        update={
            "offered": True,
            "field_states": {field: "skipped" for field in OPTIONAL_FACT_FIELDS},
        }
    )
    summary = StorySummary.model_validate(snapshots["summary"]["output"])
    collection_data = collection.model_dump(mode="json")

    validate_summary_grounding(summary, facts, collection_data)

    decision = ApprovalDecision.model_validate(snapshots["approval"]["output"])
    result = ConversationNodes.approval_guard(
        {
            "approval_decision": decision.model_dump(mode="json"),
            "approval_pending": True,
            "current_summary": summary.model_dump(mode="json"),
            "summary_version": 1,
            "facts_revision": 1,
            "summary_facts_revision": 1,
            "collection_revision": 1,
            "summary_collection_revision": 1,
            "optional_collection": collection_data,
        }
    )

    assert result["workflow_stage"] == "generation-ready"
