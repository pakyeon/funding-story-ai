from funding_story_ai.intake import build_intake_graph


def test_intake_starts_with_product_request() -> None:
    result = build_intake_graph().invoke({})
    assert result["stage"] == "initial"
    assert result["requested_fields"] == ["initial_message", "product_image"]


def test_agent_question_is_propagated_without_profile_routing() -> None:
    result = build_intake_graph().invoke(
        {
            "initial_message": "제품 스토리를 만들어 줘",
            "agent_ready_to_confirm": False,
            "agent_requested_fields": ["key_strengths", "target_supporters"],
            "agent_question": "핵심 강점과 주요 사용자는 누구인가요?",
        }
    )
    assert result["stage"] == "follow-up"
    assert result["requested_fields"] == ["key_strengths", "target_supporters"]
    assert result["question"] == "핵심 강점과 주요 사용자는 누구인가요?"


def test_agent_ready_state_routes_to_confirmation() -> None:
    result = build_intake_graph().invoke(
        {
            "initial_message": "제품·강점·타깃이 포함된 설명",
            "agent_ready_to_confirm": True,
        }
    )
    assert result["stage"] == "confirmation"


def test_generation_starts_only_after_confirmation_or_explicit_skip() -> None:
    confirmed = build_intake_graph().invoke(
        {"initial_message": "제품 설명", "confirmed": True}
    )
    assert confirmed["stage"] == "ready-to-generate"
    assert confirmed["generation_start_trigger"] == "explicit-confirmation"

    skipped = build_intake_graph().invoke(
        {"initial_message": "제품 설명", "skip_remaining_questions": True}
    )
    assert skipped["stage"] == "ready-to-generate"
    assert skipped["generation_start_trigger"] == "explicit-skip"
