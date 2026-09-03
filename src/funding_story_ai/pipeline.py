from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from .adapter import GeminiAdapter, GenerationResult
from .data_repository import DataRepository, DataValidationError
from .prompting import build_story_prompt
from .selector import TemplateSelection, TemplateSelector
from .validation import StoryValidator, StoryWarning


class StoryPipelineError(RuntimeError):
    pass


class StoryTemplateSelector(Protocol):
    def select(
        self,
        brief: dict[str, Any],
        templates: list[dict[str, Any]],
    ) -> TemplateSelection: ...


class StoryPipelineState(TypedDict, total=False):
    brief: dict[str, Any]
    media_facts: dict[str, Any] | None
    requested_template_id: str | None
    template: dict[str, Any]
    template_version: str
    selection: TemplateSelection
    prompt: str
    content: dict[str, Any]
    generation: GenerationResult
    warnings: list[StoryWarning]
    retry_count: int
    schema_error: str | None
    result: dict[str, Any]
    status: Literal["generating", "validating", "complete"]


class StoryPipeline:
    def __init__(
        self,
        *,
        repository: DataRepository,
        adapter: GeminiAdapter,
        selector: StoryTemplateSelector | None = None,
        validator: StoryValidator | None = None,
        max_correction_attempts: int = 0,
    ) -> None:
        self.repository = repository
        self.adapter = adapter
        self.selector = selector or TemplateSelector()
        self.validator = validator or StoryValidator()
        self.max_correction_attempts = max_correction_attempts
        self.graph = self._build_graph()

    def _select_template(self, state: StoryPipelineState) -> StoryPipelineState:
        requested = state.get("requested_template_id")
        if requested:
            base_template = self.repository.get_template(requested)
            selection = TemplateSelection(
                template_id=requested,
                score=0,
                scores={requested: 0},
                reasons=("explicit template request",),
            )
        else:
            templates = self.repository.load_templates()
            selection = self.selector.select(state["brief"], templates)
            base_template = self.repository.get_template(selection.template_id)
        template = self.repository.compose_template(
            template_id=base_template["id"],
            brief=state["brief"],
            media_facts=state.get("media_facts"),
        )
        return {
            "template": template,
            "template_version": self.repository.get_template_version(base_template["id"]),
            "selection": selection,
            "retry_count": 0,
        }

    def _build_prompt(self, state: StoryPipelineState) -> StoryPipelineState:
        retry = state.get("retry_count", 0)
        warning_messages = [warning.message for warning in state.get("warnings", [])]
        if state.get("schema_error"):
            warning_messages.append(str(state["schema_error"]))
        return {
            "prompt": build_story_prompt(
                brief=state["brief"],
                template=state["template"],
                previous_content=state.get("content") if retry else None,
                validation_messages=warning_messages if retry else None,
            ),
            "status": "generating",
        }

    def _generate(self, state: StoryPipelineState) -> StoryPipelineState:
        generation = self.adapter.generate_json(
            prompt=state["prompt"],
            response_schema=self.repository.story_generation_content_schema(),
        )
        return {
            "generation": generation,
            "content": generation.data,
            "status": "validating",
        }

    def _validate(self, state: StoryPipelineState) -> StoryPipelineState:
        try:
            self.repository.validate_story_generation_content(state["content"])
        except DataValidationError as exc:
            return {
                "schema_error": str(exc),
                "warnings": [StoryWarning("schema-validation-error", str(exc))],
            }
        warnings = self.validator.validate(
            content=state["content"],
            brief=state["brief"],
            template=state["template"],
        )
        return {"warnings": warnings, "schema_error": None}

    def _route_after_validation(self, state: StoryPipelineState) -> str:
        if (
            state.get("warnings")
            and state.get("retry_count", 0) < self.max_correction_attempts
        ):
            return "prepare_retry"
        return "finalize"

    @staticmethod
    def _prepare_retry(state: StoryPipelineState) -> StoryPipelineState:
        return {"retry_count": state.get("retry_count", 0) + 1}

    def _finalize(self, state: StoryPipelineState) -> StoryPipelineState:
        if state.get("schema_error"):
            raise StoryPipelineError(
                "Gemini output failed the content schema after the correction attempt: "
                f"{state['schema_error']}"
            )
        generation = state["generation"]
        content = state["content"]
        warnings = state.get("warnings", [])
        result = {
            "schema_version": "story-result-v1",
            "language": state["brief"]["language"],
            "template_id": state["template"]["id"],
            "template_version": state["template_version"],
            "model": generation.model,
            "title_candidates": content["title_candidates"],
            "sections": content["sections"],
            "warnings": [warning.to_dict() for warning in warnings],
            "automated_validation_passed": not warnings,
            "review_required": True,
        }
        self.repository.validate_story_result(result)
        return {"result": result, "status": "complete"}

    def _build_graph(self):
        builder = StateGraph(StoryPipelineState)
        builder.add_node("select_template", self._select_template)
        builder.add_node("build_prompt", self._build_prompt)
        builder.add_node("generate_story", self._generate)
        builder.add_node("validate_story", self._validate)
        builder.add_node("prepare_retry", self._prepare_retry)
        builder.add_node("finalize", self._finalize)
        builder.add_edge(START, "select_template")
        builder.add_edge("select_template", "build_prompt")
        builder.add_edge("build_prompt", "generate_story")
        builder.add_edge("generate_story", "validate_story")
        builder.add_conditional_edges(
            "validate_story",
            self._route_after_validation,
            {"prepare_retry": "prepare_retry", "finalize": "finalize"},
        )
        builder.add_edge("prepare_retry", "build_prompt")
        builder.add_edge("finalize", END)
        return builder.compile()

    def invoke(
        self,
        brief: dict[str, Any],
        *,
        template_id: str | None = None,
        media_facts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.repository.validate_story_brief(brief)
        state = self.graph.invoke(
            {
                "brief": brief,
                "requested_template_id": template_id,
                "media_facts": media_facts,
            }
        )
        return state["result"]
