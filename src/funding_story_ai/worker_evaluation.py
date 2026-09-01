from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .adapter import GeminiAdapter
from .config import RuntimeSettings
from .conversation import (
    FACT_FIELDS,
    OPTIONAL_FACT_FIELDS,
    ConversationNodes,
    FactField,
    FactValue,
    OptionalCollection,
    QuestionPlan,
    QuestionPurpose,
    StorySummary,
    TurnUnderstanding,
    initial_facts,
    question_plan_error,
    validate_summary_grounding,
)
from .field_evaluation import RequestRateLimiter, _model_error_category
from .worker import GeminiConversationModel, StoryMakerWorker, WorkerRequest


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class CollectionCase(StrictModel):
    id: str
    message: str
    expected_directive: dict[str, Any]
    expected_intent: str
    prior_facts: dict[FactField, FactValue] = Field(default_factory=dict)
    messages: list[EvaluationMessage] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class CollectionDataset(StrictModel):
    schema_version: str
    cases: list[CollectionCase]

    @model_validator(mode="after")
    def validate_size(self) -> CollectionDataset:
        if self.schema_version != "story-worker-collection-evaluation-v1":
            raise ValueError("unsupported collection dataset")
        _validate_case_count(self.cases)
        return self


class QuestionCase(StrictModel):
    id: str
    purpose: QuestionPurpose
    candidate_fields: list[FactField]
    requested_group: str | None
    requested_detail: str
    facts: dict[FactField, FactValue] = Field(default_factory=dict)
    messages: list[EvaluationMessage] = Field(default_factory=list)
    question_history: list[dict[str, Any]] = Field(default_factory=list)
    expected_requested_fields: list[FactField]
    tags: list[str] = Field(default_factory=list)


class QuestionDataset(StrictModel):
    schema_version: str
    cases: list[QuestionCase]

    @model_validator(mode="after")
    def validate_size(self) -> QuestionDataset:
        if self.schema_version != "story-worker-question-evaluation-v1":
            raise ValueError("unsupported question dataset")
        _validate_case_count(self.cases)
        return self


class SummaryCase(StrictModel):
    id: str
    facts: dict[FactField, FactValue]
    field_states: dict[FactField, str]
    messages: list[EvaluationMessage] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class SummaryDataset(StrictModel):
    schema_version: str
    cases: list[SummaryCase]

    @model_validator(mode="after")
    def validate_size(self) -> SummaryDataset:
        if self.schema_version != "story-worker-summary-evaluation-v1":
            raise ValueError("unsupported summary dataset")
        _validate_case_count(self.cases)
        return self


class ApprovalCase(StrictModel):
    id: str
    message: str
    expected_decision: Literal["approve", "revise", "reject", "ambiguous"]
    tags: list[str] = Field(default_factory=list)


class ApprovalDataset(StrictModel):
    schema_version: str
    cases: list[ApprovalCase]

    @model_validator(mode="after")
    def validate_size(self) -> ApprovalDataset:
        if self.schema_version != "story-worker-approval-evaluation-v1":
            raise ValueError("unsupported approval dataset")
        _validate_case_count(self.cases)
        return self


class FlowTurn(StrictModel):
    message: str
    expected_stage: str
    expected_phase: str


class FlowCase(StrictModel):
    id: str
    scenario: str
    turns: list[FlowTurn]


class FlowDataset(StrictModel):
    schema_version: str
    cases: list[FlowCase]

    @model_validator(mode="after")
    def validate_size(self) -> FlowDataset:
        if self.schema_version != "story-worker-flow-evaluation-v1":
            raise ValueError("unsupported flow dataset")
        _validate_case_count(self.cases)
        return self


def _validate_case_count(cases: list[Any]) -> None:
    if not 50 <= len(cases) <= 200:
        raise ValueError("evaluation datasets must contain 50 to 200 cases")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case ids must be unique")


def _load(path: Path, model: type[StrictModel]) -> StrictModel:
    return model.model_validate_json(path.read_text(encoding="utf-8"))


def _messages(values: list[EvaluationMessage], case_id: str) -> list[dict[str, str]]:
    return [
        {"id": f"{case_id}-message-{index}", "role": value.role, "content": value.content}
        for index, value in enumerate(values, start=1)
    ]


def _facts(values: dict[FactField, FactValue]) -> dict[str, dict[str, Any]]:
    result = initial_facts()
    for field, value in values.items():
        result[field] = value.model_dump(mode="json")
    return result


class ModelPool:
    def __init__(
        self,
        settings: RuntimeSettings,
        *,
        before_call: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings
        self.before_call = before_call
        self.local = threading.local()

    def get(self) -> GeminiConversationModel:
        model = getattr(self.local, "model", None)
        if model is None:
            model = GeminiConversationModel(
                GeminiAdapter(
                    self.settings,
                    allow_fallback=False,
                    before_call=self.before_call,
                )
            )
            self.local.model = model
        return model


def _run_parallel(
    cases: list[Any],
    evaluate: Callable[[Any], dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(evaluate, case): case.id for case in cases}
    completed: dict[str, dict[str, Any]] = {}
    try:
        for future in as_completed(futures):
            result = future.result()
            completed[result["id"]] = result
            done = len(completed)
            if done % 5 == 0 or done == len(cases) or result.get("error") is not None:
                errors = sum(item.get("error") is not None for item in completed.values())
                print(f"progress {done}/{len(cases)} errors={errors}", flush=True)
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return [completed[case.id] for case in cases]


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def evaluate_collection(
    dataset: CollectionDataset,
    pool: ModelPool,
    workers: int,
) -> dict[str, Any]:
    def evaluate(case: CollectionCase) -> dict[str, Any]:
        messages = _messages(case.messages, case.id)
        current = {"id": f"{case.id}-current", "role": "user", "content": case.message}
        messages.append(current)
        try:
            model = pool.get()
            output = model.understand_turn(
                message=current,
                messages=messages,
                facts=_facts(case.prior_facts),
                image_path=None,
            )
            directive = output.collection_directive.model_dump(mode="json")
            expected = case.expected_directive
            action_ok = directive["action"] == expected.get("action", "none")
            groups_ok = set(directive["groups"]) == set(expected.get("groups", []))
            fields_ok = set(directive["fields"]) == set(expected.get("fields", []))
            clarification_ok = directive["requires_clarification"] == expected.get(
                "requires_clarification", False
            )
            intent_ok = output.intent == case.expected_intent
            exact = action_ok and groups_ok and fields_ok and clarification_ok
            return {
                "id": case.id,
                "model": model.last_model,
                "error": None,
                "expected": expected,
                "predicted": directive,
                "intent": output.intent,
                "intent_correct": intent_ok,
                "action_correct": action_ok,
                "groups_correct": groups_ok,
                "fields_correct": fields_ok,
                "clarification_correct": clarification_ok,
                "directive_exact": exact,
                "tags": case.tags,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "id": case.id,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "tags": case.tags,
            }

    cases = _run_parallel(dataset.cases, evaluate, workers)
    total = len(cases)
    success = [case for case in cases if case.get("error") is None]
    metrics = {
        "case_count": total,
        "model_response_success_rate": _rate(len(success), total),
        "intent_accuracy": _rate(sum(case["intent_correct"] for case in success), total),
        "action_accuracy": _rate(sum(case["action_correct"] for case in success), total),
        "group_selection_accuracy": _rate(sum(case["groups_correct"] for case in success), total),
        "field_selection_accuracy": _rate(sum(case["fields_correct"] for case in success), total),
        "clarification_accuracy": _rate(
            sum(case["clarification_correct"] for case in success), total
        ),
        "directive_exact_rate": _rate(sum(case["directive_exact"] for case in success), total),
    }
    return _result("collection", dataset.schema_version, metrics, cases)


def evaluate_question(
    dataset: QuestionDataset,
    pool: ModelPool,
    workers: int,
) -> dict[str, Any]:
    internal_names = re.compile("|".join(re.escape(field) for field in FACT_FIELDS))

    def evaluate(case: QuestionCase) -> dict[str, Any]:
        messages = _messages(case.messages, case.id)
        understanding = TurnUnderstanding(intent="provide_information")
        try:
            model = pool.get()
            output = model.plan_questions(
                messages=messages,
                facts=_facts(case.facts),
                purpose=case.purpose,
                candidate_fields=list(case.candidate_fields),
                requested_group=case.requested_group,
                requested_detail=case.requested_detail,
                question_history=case.question_history,
                turn_understanding=understanding,
            )
            raw_output = output
            state = {
                "current_question_plan": raw_output.model_dump(mode="json"),
                "question_plan_error": None,
                "messages": messages,
                "facts": _facts(case.facts),
                "question_purpose": case.purpose,
                "question_candidate_fields": list(case.candidate_fields),
                "question_group": case.requested_group,
                "question_requested_detail": case.requested_detail,
            }
            raw_error = question_plan_error(state, raw_output)
            repaired = raw_error is not None
            if repaired:
                state["question_plan_error"] = raw_error
                repaired_state = ConversationNodes(model).repair_question_plan(state)
                output = QuestionPlan.model_validate(repaired_state["current_question_plan"])
            purpose_ok = output.purpose == case.purpose
            group_ok = output.requested_group == case.requested_group
            detail_ok = output.requested_detail == case.requested_detail
            fields_ok = set(output.requested_fields) == set(case.expected_requested_fields)
            within_candidates = set(output.requested_fields).issubset(set(case.candidate_fields))
            max_three = len(output.requested_fields) <= 3
            no_internal_names = not internal_names.search(output.question)
            question_form = bool(output.question.strip()) and (
                "?" in output.question
                or re.search(
                    r"(알려주세요|알려 주[세요]|말씀해 주세요|작성해 주세요|입력해 주세요|"
                    r"확인해 주세요|선택해 주세요|들려주세요|편하게 들려주세요|"
                    r"괜찮습니다|드릴게요|부탁드립니다|진행할까요|인가요|할까요)",
                    output.question,
                )
                is not None
            )
            exact = purpose_ok and group_ok and detail_ok and fields_ok
            return {
                "id": case.id,
                "model": model.last_model,
                "error": None,
                "expected_fields": case.expected_requested_fields,
                "raw_predicted": raw_output.model_dump(mode="json"),
                "predicted": output.model_dump(mode="json"),
                "raw_plan_valid": raw_error is None,
                "repaired": repaired,
                "purpose_correct": purpose_ok,
                "group_correct": group_ok,
                "detail_correct": detail_ok,
                "requested_fields_exact": fields_ok,
                "within_candidates": within_candidates,
                "max_three": max_three,
                "no_internal_field_names": no_internal_names,
                "question_form": question_form,
                "plan_exact": exact,
                "tags": case.tags,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "id": case.id,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "tags": case.tags,
            }

    cases = _run_parallel(dataset.cases, evaluate, workers)
    total = len(cases)
    success = [case for case in cases if case.get("error") is None]
    keys = [
        "purpose_correct",
        "group_correct",
        "detail_correct",
        "requested_fields_exact",
        "within_candidates",
        "max_three",
        "no_internal_field_names",
        "question_form",
        "plan_exact",
    ]
    metrics = {"case_count": total, "model_response_success_rate": _rate(len(success), total)}
    metrics.update(
        {f"{key}_rate": _rate(sum(case[key] for case in success), total) for key in keys}
    )
    metrics["raw_plan_valid_rate"] = _rate(
        sum(case["raw_plan_valid"] for case in success), total
    )
    metrics["repair_rate"] = _rate(sum(case["repaired"] for case in success), total)
    return _result("question", dataset.schema_version, metrics, cases)


def evaluate_summary(
    dataset: SummaryDataset,
    pool: ModelPool,
    workers: int,
) -> dict[str, Any]:
    def evaluate(case: SummaryCase) -> dict[str, Any]:
        facts = _facts(case.facts)
        collection = OptionalCollection(
            offered=True,
            field_states={field: case.field_states[field] for field in OPTIONAL_FACT_FIELDS},
        ).model_dump(mode="json")
        try:
            model = pool.get()
            output = model.build_summary(
                messages=_messages(case.messages, case.id),
                facts=facts,
                optional_collection=collection,
            )
            try:
                validate_summary_grounding(output, facts, collection)
                grounding_ok = True
                grounding_error = None
            except Exception as exc:  # noqa: BLE001
                grounding_ok = False
                grounding_error = str(exc)
            text = output.summary_text
            absent_expected = bool(output.explicitly_absent_fields)
            skipped_expected = bool(output.skipped_fields)
            absent_labeled = (not absent_expected) or any(
                token in text for token in ("없", "미제공", "미보유")
            )
            skipped_labeled = (not skipped_expected) or any(
                token in text for token in ("생략", "제외", "건너")
            )
            explicit_confirmation = "스토리" in output.confirmation_question and any(
                token in output.confirmation_question for token in ("생성", "작성", "진행")
            )
            return {
                "id": case.id,
                "model": model.last_model,
                "error": None,
                "predicted": output.model_dump(mode="json"),
                "grounding_valid": grounding_ok,
                "grounding_error": grounding_error,
                "absent_state_labeled": absent_labeled,
                "skipped_state_labeled": skipped_labeled,
                "explicit_confirmation_question": explicit_confirmation,
                "tags": case.tags,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "id": case.id,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "tags": case.tags,
            }

    cases = _run_parallel(dataset.cases, evaluate, workers)
    total = len(cases)
    success = [case for case in cases if case.get("error") is None]
    metrics = {
        "case_count": total,
        "model_response_success_rate": _rate(len(success), total),
        "grounding_valid_rate": _rate(sum(case["grounding_valid"] for case in success), total),
        "absent_state_label_rate": _rate(
            sum(case["absent_state_labeled"] for case in success), total
        ),
        "skipped_state_label_rate": _rate(
            sum(case["skipped_state_labeled"] for case in success), total
        ),
        "explicit_confirmation_question_rate": _rate(
            sum(case["explicit_confirmation_question"] for case in success), total
        ),
        "hallucination_or_contract_failure_rate": _rate(
            sum(not case["grounding_valid"] for case in success) + (total - len(success)), total
        ),
    }
    return _result("summary", dataset.schema_version, metrics, cases)


def _approval_summary() -> StorySummary:
    return StorySummary(
        headline="스토리 생성 정보 요약",
        confirmed_facts={
            "product_name": ["오빗클린 V3"],
            "product_type": ["로봇청소기"],
            "category": ["테크·가전"],
            "key_strengths": ["얇은 본체"],
            "target_supporters": ["맞벌이 가구"],
        },
        explicitly_absent_fields=[],
        skipped_fields=list(OPTIONAL_FACT_FIELDS),
        summary_text="오빗클린 V3 로봇청소기 정보를 확인했습니다. 선택 정보는 생략했습니다.",
        confirmation_question="이 내용으로 스토리를 생성할까요?",
    )


def evaluate_approval(
    dataset: ApprovalDataset,
    pool: ModelPool,
    workers: int,
) -> dict[str, Any]:
    summary = _approval_summary()

    def evaluate(case: ApprovalCase) -> dict[str, Any]:
        message = {"id": f"{case.id}-current", "role": "user", "content": case.message}
        try:
            model = pool.get()
            output = model.classify_approval(
                message=message,
                summary=summary,
                messages=[
                    {
                        "id": f"{case.id}-summary",
                        "role": "assistant",
                        "content": summary.summary_text,
                    },
                    message,
                ],
            )
            return {
                "id": case.id,
                "model": model.last_model,
                "error": None,
                "expected": case.expected_decision,
                "predicted": output.decision,
                "correct": output.decision == case.expected_decision,
                "dangerous_false_approval": output.decision == "approve"
                and case.expected_decision != "approve",
                "tags": case.tags,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "id": case.id,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "tags": case.tags,
            }

    cases = _run_parallel(dataset.cases, evaluate, workers)
    total = len(cases)
    success = [case for case in cases if case.get("error") is None]
    confusion = Counter((case["expected"], case["predicted"]) for case in success)
    approve_cases = [case for case in success if case["expected"] == "approve"]
    predicted_approve = [case for case in success if case["predicted"] == "approve"]
    true_approve = sum(
        case["expected"] == "approve" and case["predicted"] == "approve" for case in success
    )
    metrics = {
        "case_count": total,
        "model_response_success_rate": _rate(len(success), total),
        "decision_accuracy": _rate(sum(case["correct"] for case in success), total),
        "approve_precision": _rate(true_approve, len(predicted_approve)),
        "approve_recall": _rate(true_approve, len(approve_cases)),
        "dangerous_false_approval_count": sum(case["dangerous_false_approval"] for case in success),
        "confusion": {
            f"{expected}->{predicted}": count
            for (expected, predicted), count in sorted(confusion.items())
        },
    }
    return _result("approval", dataset.schema_version, metrics, cases)


def evaluate_flow(
    dataset: FlowDataset,
    settings: RuntimeSettings,
    workers: int,
    *,
    checkpoint_path: Path | None = None,
    resume: bool = False,
    requests_per_minute: int = 30,
) -> dict[str, Any]:
    local = threading.local()
    limiter = RequestRateLimiter(requests_per_minute)

    def get_worker() -> StoryMakerWorker:
        worker = getattr(local, "worker", None)
        if worker is None:
            worker = StoryMakerWorker(
                conversation_model=GeminiConversationModel(
                    GeminiAdapter(
                        settings,
                        allow_fallback=False,
                        before_call=limiter.wait,
                    )
                )
            )
            local.worker = worker
        return worker

    def evaluate(case: FlowCase) -> dict[str, Any]:
        worker = get_worker()
        turns: list[dict[str, Any]] = []
        no_approval_ready = False
        seen_question_progress: Counter[tuple[str, str]] = Counter()
        abnormal_repeated_questions = 0
        try:
            for index, turn in enumerate(case.turns, start=1):
                started_at = time.perf_counter()
                history_length_before = len(
                    worker.get_state(case.id).get("question_history", [])
                )
                outcome = asyncio.run(
                    worker.handle(
                        WorkerRequest(
                            thread_id=case.id,
                            input_id=case.id,
                            message=turn.message,
                            message_id=f"{case.id}-turn-{index}",
                        )
                    )
                )
                latency_ms = (time.perf_counter() - started_at) * 1000
                stage_ok = outcome.stage == turn.expected_stage
                phase_ok = outcome.collection_phase == turn.expected_phase
                if outcome.temporary_error:
                    raise RuntimeError("temporary model failure; retry this evaluation session")
                if outcome.stage == "generation-ready" and (
                    outcome.generation_start_trigger != "explicit-confirmation"
                ):
                    no_approval_ready = True
                state = worker.get_state(case.id)
                question_history = state.get("question_history", [])
                new_question = len(question_history) > history_length_before
                if new_question and question_history:
                    latest_question = question_history[-1]
                    key = (
                        str(latest_question.get("signature", "")),
                        str(latest_question.get("progress_fingerprint", "")),
                    )
                    abnormal_repeated_questions += int(seen_question_progress[key] > 1)
                    seen_question_progress[key] += 1
                turns.append(
                    {
                        "turn": index,
                        "message": turn.message,
                        "expected_stage": turn.expected_stage,
                        "actual_stage": outcome.stage,
                        "expected_phase": turn.expected_phase,
                        "actual_phase": outcome.collection_phase,
                        "stage_correct": stage_ok,
                        "phase_correct": phase_ok,
                        "requested_fields": list(outcome.requested_fields),
                        "reply": outcome.reply,
                        "latency_ms": latency_ms,
                        "question_signature": (
                            question_history[-1].get("signature")
                            if new_question and question_history
                            else None
                        ),
                        "progress_fingerprint": (
                            question_history[-1].get("progress_fingerprint")
                            if new_question and question_history
                            else None
                        ),
                    }
                )
            final = turns[-1]
            return {
                "id": case.id,
                "scenario": case.scenario,
                "error": None,
                "turns": turns,
                "all_transitions_correct": all(
                    turn["stage_correct"] and turn["phase_correct"] for turn in turns
                ),
                "final_state_correct": final["actual_stage"] == final["expected_stage"],
                "generation_ready": final["actual_stage"] == "generation-ready",
                "ready_without_explicit_approval": no_approval_ready,
                "abnormal_repeated_question_count": abnormal_repeated_questions,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "id": case.id,
                "scenario": case.scenario,
                "error": {
                    "type": type(exc).__name__,
                    "category": _model_error_category(exc),
                    "message": str(exc),
                },
                "turns": turns,
            }

    checkpoint_header = {
        "type": "header",
        "implementation_revision": "story-worker-flow-mvp-v9",
        "dataset_schema_version": dataset.schema_version,
        "case_count": len(dataset.cases),
        "model": settings.primary_model,
        "location": settings.location,
        "request_timeout_ms": settings.request_timeout_ms,
        "primary_access_attempts": settings.primary_access_attempts,
        "thinking_level": settings.thinking_level,
        "max_output_tokens": settings.max_output_tokens,
        "requests_per_minute": requests_per_minute,
    }
    completed: dict[str, dict[str, Any]] = {}
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
            if not lines or json.loads(lines[0]) != checkpoint_header:
                raise ValueError("flow evaluation checkpoint does not match this run")
            for line in lines[1:]:
                record = json.loads(line)
                result = record.get("result")
                if result is not None and result.get("error") is None:
                    completed[result["id"]] = result
        else:
            checkpoint_path.write_text(
                json.dumps(checkpoint_header, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    pending = [case for case in dataset.cases if case.id not in completed]
    total = len(dataset.cases)
    done = len(completed)
    if done:
        print(f"resumed {done}/{total} successful flow sessions", flush=True)
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(evaluate, case): case.id for case in pending}
    try:
        for future in as_completed(futures):
            result = future.result()
            completed[result["id"]] = result
            done += 1
            if checkpoint_path is not None:
                with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
                    checkpoint.write(
                        json.dumps(
                            {"type": "case", "result": result}, ensure_ascii=False
                        )
                        + "\n"
                    )
            if done % 5 == 0 or done == total or result.get("error") is not None:
                error_count = sum(
                    value.get("error") is not None for value in completed.values()
                )
                print(f"progress {done}/{total} errors={error_count}", flush=True)
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    cases = [completed[case.id] for case in dataset.cases]
    total = len(cases)
    success = [case for case in cases if case.get("error") is None]
    turn_total = sum(len(case["turns"]) for case in cases)
    correct_turns = sum(
        turn["stage_correct"] and turn["phase_correct"]
        for case in success
        for turn in case["turns"]
    )
    question_total = sum(
        bool(turn.get("question_signature")) for case in success for turn in case["turns"]
    )
    abnormal_repeat_total = sum(
        case["abnormal_repeated_question_count"] for case in success
    )
    latencies = sorted(
        float(turn["latency_ms"]) for case in success for turn in case["turns"]
    )

    def percentile(value: float) -> float:
        if not latencies:
            return 0.0
        return latencies[max(0, math.ceil(len(latencies) * value) - 1)]

    metrics = {
        "case_count": total,
        "model_session_success_rate": _rate(len(success), total),
        "transition_accuracy": _rate(correct_turns, turn_total),
        "all_transitions_correct_session_rate": _rate(
            sum(case["all_transitions_correct"] for case in success), total
        ),
        "final_state_accuracy": _rate(sum(case["final_state_correct"] for case in success), total),
        "session_completion_rate": _rate(
            sum(
                case["final_state_correct"] and case["generation_ready"]
                for case in success
            ),
            total,
        ),
        "generation_ready_rate": _rate(sum(case["generation_ready"] for case in success), total),
        "ready_without_explicit_approval_count": sum(
            case["ready_without_explicit_approval"] for case in success
        ),
        "question_output_count": question_total,
        "abnormal_repeated_question_count": abnormal_repeat_total,
        "abnormal_repeated_question_rate": _rate(abnormal_repeat_total, question_total),
        "average_turns": _rate(turn_total, total),
        "max_turns": max((len(case["turns"]) for case in cases), default=0),
        "turn_latency_p50_ms": percentile(0.50),
        "turn_latency_p95_ms": percentile(0.95),
        "error_categories": dict(
            Counter(
                case["error"].get("category", "unknown")
                for case in cases
                if case.get("error") is not None
            )
        ),
    }
    return _result("flow", dataset.schema_version, metrics, cases)


def _result(
    suite: str,
    dataset_schema_version: str,
    metrics: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "story-worker-live-evaluation-result-v1",
        "suite": suite,
        "dataset_schema_version": dataset_schema_version,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "cases": cases,
    }


def _runtime(
    settings: RuntimeSettings, workers: int, requests_per_minute: int
) -> dict[str, Any]:
    return {
        "primary_model": settings.primary_model,
        "fallback_enabled": False,
        "location": settings.location,
        "request_timeout_ms": settings.request_timeout_ms,
        "primary_access_attempts": settings.primary_access_attempts,
        "thinking_level": settings.thinking_level,
        "max_output_tokens": settings.max_output_tokens,
        "workers": workers,
        "requests_per_minute": requests_per_minute,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate live Gemini story-maker-worker nodes")
    parser.add_argument(
        "suite",
        choices=["collection", "question", "summary", "approval", "flow"],
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--request-timeout-ms",
        type=int,
        default=60000,
        help="Per-request timeout for live evaluation calls",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Same-model attempts per request; keep at 1 for bounded evaluation",
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--requests-per-minute", type=int, default=30)
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N validated cases while preserving the full dataset on disk",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if args.request_timeout_ms < 1000:
        raise ValueError("request-timeout-ms must be at least 1000")
    if not 1 <= args.attempts <= 5:
        raise ValueError("attempts must be between 1 and 5")
    if args.requests_per_minute <= 0:
        raise ValueError("requests-per-minute must be positive")
    if args.resume and args.checkpoint is None:
        raise ValueError("--resume requires --checkpoint")

    defaults = {
        "collection": ("story-worker-collection-evaluation-v1.json", CollectionDataset),
        "question": ("story-worker-question-evaluation-v1.json", QuestionDataset),
        "summary": ("story-worker-summary-evaluation-v1.json", SummaryDataset),
        "approval": ("story-worker-approval-evaluation-v1.json", ApprovalDataset),
        "flow": ("story-worker-flow-evaluation-v1.json", FlowDataset),
    }
    filename, dataset_model = defaults[args.suite]
    dataset_path = args.dataset or Path("evals/datasets") / filename
    dataset = _load(dataset_path, dataset_model)
    dataset_total = len(dataset.cases)
    if args.limit is not None:
        if not 50 <= args.limit <= dataset_total:
            raise ValueError(f"limit must be between 50 and {dataset_total}")
        dataset = dataset.model_copy(update={"cases": dataset.cases[: args.limit]})
    if args.validate_only:
        print(
            json.dumps(
                {"suite": args.suite, "case_count": len(dataset.cases), "valid": True},
                ensure_ascii=False,
            )
        )
        return

    settings = replace(
        RuntimeSettings.from_env(),
        request_timeout_ms=args.request_timeout_ms,
        primary_access_attempts=args.attempts,
    )
    limiter = RequestRateLimiter(args.requests_per_minute)
    pool = ModelPool(settings, before_call=limiter.wait)
    evaluators: dict[str, Callable[..., dict[str, Any]]] = {
        "collection": evaluate_collection,
        "question": evaluate_question,
        "summary": evaluate_summary,
        "approval": evaluate_approval,
    }
    if args.suite == "flow":
        result = evaluate_flow(
            dataset,
            settings,
            args.workers,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
            requests_per_minute=args.requests_per_minute,
        )
    else:
        result = evaluators[args.suite](dataset, pool, args.workers)
    result["runtime"] = _runtime(settings, args.workers, args.requests_per_minute)
    result["evaluation_scope"] = {
        "dataset_case_count": dataset_total,
        "evaluated_case_count": len(dataset.cases),
        "limited": args.limit is not None,
    }
    output = args.output or Path("artifacts/evaluations") / f"story-worker-{args.suite}-live.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
