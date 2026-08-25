from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

IntakeStage = Literal["initial", "follow-up", "confirmation", "ready-to-generate"]


class StoryIntakeState(TypedDict, total=False):
    """The LLM decides what to ask; the graph only guards state transitions."""

    initial_message: str
    agent_ready_to_confirm: bool
    agent_question: str | None
    agent_requested_fields: list[str]
    skip_remaining_questions: bool
    confirmed: bool
    generation_start_trigger: Literal["explicit-confirmation", "explicit-skip"]
    stage: IntakeStage
    requested_fields: list[str]
    question: str | None


def route_intake(
    state: StoryIntakeState,
) -> Command[Literal["ask_initial", "ask_follow_up", "confirm", "ready"]]:
    if not str(state.get("initial_message", "")).strip():
        return Command(goto="ask_initial")
    if state.get("skip_remaining_questions", False) or state.get("confirmed", False):
        return Command(goto="ready")
    if not state.get("agent_ready_to_confirm", False):
        return Command(goto="ask_follow_up")
    return Command(goto="confirm")


def ask_initial(_: StoryIntakeState) -> StoryIntakeState:
    return {
        "stage": "initial",
        "requested_fields": ["initial_message", "product_image"],
        "question": "제품과 만들고 싶은 펀딩 스토리를 설명해 주세요.",
    }


def ask_follow_up(state: StoryIntakeState) -> StoryIntakeState:
    return {
        "stage": "follow-up",
        "requested_fields": list(state.get("agent_requested_fields", [])),
        "question": state.get("agent_question"),
    }


def confirm(_: StoryIntakeState) -> StoryIntakeState:
    return {
        "stage": "confirmation",
        "requested_fields": ["confirmed"],
        "question": None,
    }


def ready(state: StoryIntakeState) -> StoryIntakeState:
    trigger: Literal["explicit-confirmation", "explicit-skip"] = (
        "explicit-skip"
        if state.get("skip_remaining_questions", False)
        else "explicit-confirmation"
    )
    return {
        "stage": "ready-to-generate",
        "requested_fields": [],
        "question": None,
        "generation_start_trigger": trigger,
    }


def build_intake_graph():
    builder = StateGraph(StoryIntakeState)
    builder.add_node("route", route_intake)
    builder.add_node("ask_initial", ask_initial)
    builder.add_node("ask_follow_up", ask_follow_up)
    builder.add_node("confirm", confirm)
    builder.add_node("ready", ready)
    builder.add_edge(START, "route")
    for node in ("ask_initial", "ask_follow_up", "confirm", "ready"):
        builder.add_edge(node, END)
    return builder.compile()
