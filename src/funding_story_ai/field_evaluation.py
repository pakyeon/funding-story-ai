from __future__ import annotations

import argparse
import json
import math
import threading
import time
import unicodedata
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .adapter import GeminiAdapter
from .config import RuntimeSettings
from .conversation import (
    FACT_FIELDS,
    FactField,
    FactPatch,
    FactValue,
    TurnUnderstanding,
    apply_fact_patches,
    initial_facts,
)
from .worker import GeminiConversationModel

FIELD_EVALUATION_IMPLEMENTATION_REVISION = "story-worker-field-mvp-v2"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvaluationMessage(StrictModel):
    role: str
    content: str = Field(min_length=1)


class FieldEvaluationCase(StrictModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    history: list[EvaluationMessage] = Field(default_factory=list)
    prior_facts: dict[FactField, FactValue] = Field(default_factory=dict)
    message: str = Field(min_length=1)
    expected: TurnUnderstanding
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def unique_tags(cls, tags: list[str]) -> list[str]:
        return list(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))


class FieldEvaluationDataset(StrictModel):
    schema_version: str
    cases: list[FieldEvaluationCase]

    @model_validator(mode="after")
    def validate_dataset(self) -> FieldEvaluationDataset:
        if self.schema_version != "story-worker-field-evaluation-v1":
            raise ValueError("unsupported field evaluation schema_version")
        if not 50 <= len(self.cases) <= 200:
            raise ValueError("field evaluation dataset must contain 50 to 200 cases")
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("field evaluation case ids must be unique")
        covered_fields = {
            patch.field for case in self.cases for patch in case.expected.fact_patches
        }
        missing_fields = set(FACT_FIELDS) - covered_fields
        if missing_fields:
            raise ValueError(
                f"field evaluation dataset does not cover fields: {sorted(missing_fields)}"
            )
        field_case_counts = Counter(
            patch.field for case in self.cases for patch in case.expected.fact_patches
        )
        undercovered_fields = {
            field: count for field, count in field_case_counts.items() if count < 4
        }
        if undercovered_fields:
            raise ValueError(
                "field evaluation dataset requires at least four positive patches per field: "
                f"{undercovered_fields}"
            )
        covered_operations = {
            patch.operation for case in self.cases for patch in case.expected.fact_patches
        }
        missing_operations = {"replace", "append", "mark_absent", "clear"} - covered_operations
        if missing_operations:
            raise ValueError(
                f"field evaluation dataset does not cover operations: {sorted(missing_operations)}"
            )
        operation_counts = Counter(
            patch.operation for case in self.cases for patch in case.expected.fact_patches
        )
        undercovered_operations = {
            operation: operation_counts[operation]
            for operation in ("append", "clear", "mark_absent")
            if operation_counts[operation] < 5
        }
        if undercovered_operations:
            raise ValueError(
                "field evaluation dataset requires at least five non-replace operations: "
                f"{undercovered_operations}"
            )
        for case in self.cases:
            expected_fields = [patch.field for patch in case.expected.fact_patches]
            if len(expected_fields) != len(set(expected_fields)):
                raise ValueError(f"case {case.id} contains duplicate expected patch fields")
        return self


class UnderstandingModel(Protocol):
    def understand_turn(
        self,
        *,
        message: dict[str, str],
        messages: list[dict[str, str]],
        facts: dict[str, dict[str, Any]],
        image_path: str | None,
    ) -> TurnUnderstanding: ...


class RequestRateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.interval_seconds = 60.0 / requests_per_minute
        self.lock = threading.Lock()
        self.next_request_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            scheduled = max(now, self.next_request_at)
            self.next_request_at = scheduled + self.interval_seconds
        delay = scheduled - now
        if delay > 0:
            time.sleep(delay)


class _PrecomputedUnderstandingModel:
    def __init__(
        self,
        outputs: dict[
            str,
            tuple[
                TurnUnderstanding | None,
                str | None,
                Exception | None,
                float,
                int,
            ],
        ],
    ) -> None:
        self.outputs = outputs
        self.last_model: str | None = None
        self.last_latency_ms = 0.0
        self.last_call_count = 0

    def understand_turn(
        self,
        *,
        message: dict[str, str],
        messages: list[dict[str, str]],
        facts: dict[str, dict[str, Any]],
        image_path: str | None,
    ) -> TurnUnderstanding:
        case_id = message["id"].removesuffix("-current")
        output, model, error, latency_ms, call_count = self.outputs[case_id]
        self.last_model = model
        self.last_latency_ms = latency_ms
        self.last_call_count = call_count
        if error is not None:
            raise RuntimeError(f"{type(error).__name__}: {error}") from error
        assert output is not None
        return output


def load_field_evaluation_dataset(path: Path) -> FieldEvaluationDataset:
    return FieldEvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return " ".join(normalized.split())


def _normalized_values(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(_normalize(value) for value in values)


def _patch_map(patches: Sequence[FactPatch]) -> tuple[dict[str, FactPatch], list[str]]:
    mapped: dict[str, FactPatch] = {}
    duplicates: list[str] = []
    for patch in patches:
        if patch.field in mapped:
            duplicates.append(patch.field)
        mapped[patch.field] = patch
    return mapped, duplicates


def _facts_for_case(case: FieldEvaluationCase) -> dict[str, dict[str, Any]]:
    facts = initial_facts()
    for field, value in case.prior_facts.items():
        facts[field] = value.model_dump(mode="json")
    return facts


def _apply(
    *,
    case: FieldEvaluationCase,
    facts: dict[str, dict[str, Any]],
    understanding: TurnUnderstanding,
) -> dict[str, dict[str, Any]]:
    message_id = f"{case.id}-current"
    messages = [
        {
            "id": f"{case.id}-history-{index}",
            "role": item.role,
            "content": item.content,
        }
        for index, item in enumerate(case.history, start=1)
    ]
    messages.append({"id": message_id, "role": "user", "content": case.message})
    state = {
        "incoming_message": messages[-1],
        "messages": messages,
        "facts": facts,
        "fact_history": [],
        "facts_revision": 0,
        "turn_understanding": understanding.model_dump(mode="json"),
    }
    return dict(apply_fact_patches(state)["facts"])


def _field_state(value: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    parsed = FactValue.model_validate(value)
    return parsed.status, _normalized_values(parsed.values)


def _model_error_category(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}".casefold()
    if "429" in text or "resource_exhausted" in text:
        return "rate-limit"
    if "504" in text or "deadline_exceeded" in text:
        return "gateway-timeout"
    if "timeout" in text:
        return "timeout"
    if any(token in text for token in ("transporterror", "connectionpool", "dns", "resolve")):
        return "network"
    if any(token in text for token in ("json", "schema", "validationerror", "parsed")):
        return "structured-output"
    return "model-error"


def evaluate_field_cases(
    dataset: FieldEvaluationDataset,
    model: UnderstandingModel,
) -> dict[str, Any]:
    counts = {
        "case_total": len(dataset.cases),
        "model_response_success": 0,
        "model_response_error": 0,
        "intent_correct": 0,
        "clarification_correct": 0,
        "field_true_positive": 0,
        "field_false_positive": 0,
        "field_false_negative": 0,
        "matched_field_total": 0,
        "operation_correct": 0,
        "strict_value_correct": 0,
        "field_state_total": len(dataset.cases) * len(FACT_FIELDS),
        "field_status_correct": 0,
        "field_value_correct": 0,
        "case_state_exact": 0,
        "revision_field_total": 0,
        "latest_value_correct": 0,
        "stale_value_retained": 0,
        "duplicate_predicted_patches": 0,
        "successful_field_true_positive": 0,
        "successful_field_false_positive": 0,
        "successful_field_false_negative": 0,
        "successful_case_state_exact": 0,
    }
    error_categories: dict[str, int] = {}
    per_field_counts = {
        field: {
            "expected": 0,
            "predicted": 0,
            "true_positive": 0,
            "false_positive": 0,
            "false_negative": 0,
            "matched": 0,
            "operation_correct": 0,
            "strict_value_correct": 0,
            "status_correct": 0,
            "value_correct": 0,
        }
        for field in FACT_FIELDS
    }
    case_results: list[dict[str, Any]] = []

    for case in dataset.cases:
        facts = _facts_for_case(case)
        current_message = {
            "id": f"{case.id}-current",
            "role": "user",
            "content": case.message,
        }
        messages = [
            {
                "id": f"{case.id}-history-{index}",
                "role": message.role,
                "content": message.content,
            }
            for index, message in enumerate(case.history, start=1)
        ]
        messages.append(current_message)
        model_error: dict[str, str] | None = None
        try:
            predicted = model.understand_turn(
                message=current_message,
                messages=messages,
                facts=facts,
                image_path=None,
            )
            counts["model_response_success"] += 1
        except Exception as exc:  # noqa: BLE001 - model failures are evaluation outcomes
            predicted = TurnUnderstanding(intent="unclear")
            category = _model_error_category(exc)
            model_error = {
                "type": type(exc).__name__,
                "category": category,
                "message": str(exc),
            }
            counts["model_response_error"] += 1
            error_categories[category] = error_categories.get(category, 0) + 1
        actual_model = getattr(model, "last_model", None)
        latency_ms = float(getattr(model, "last_latency_ms", 0.0))
        llm_call_count = int(getattr(model, "last_call_count", 0))
        expected = case.expected
        predicted_patches, duplicate_fields = _patch_map(predicted.fact_patches)
        expected_patches, _ = _patch_map(expected.fact_patches)
        predicted_fields = set(predicted_patches)
        expected_fields = set(expected_patches)
        matched_fields = predicted_fields.intersection(expected_fields)
        false_positive = predicted_fields - expected_fields
        false_negative = expected_fields - predicted_fields

        for field in FACT_FIELDS:
            field_counts = per_field_counts[field]
            field_counts["expected"] += int(field in expected_fields)
            field_counts["predicted"] += int(field in predicted_fields)
            field_counts["true_positive"] += int(field in matched_fields)
            field_counts["false_positive"] += int(field in false_positive)
            field_counts["false_negative"] += int(field in false_negative)

        intent_correct = model_error is None and predicted.intent == expected.intent
        clarification_correct = (
            model_error is None
            and predicted.requires_clarification == expected.requires_clarification
        )
        counts["intent_correct"] += int(intent_correct)
        counts["clarification_correct"] += int(clarification_correct)
        counts["field_true_positive"] += len(matched_fields)
        counts["field_false_positive"] += len(false_positive)
        counts["field_false_negative"] += len(false_negative)
        counts["matched_field_total"] += len(matched_fields)
        counts["duplicate_predicted_patches"] += len(duplicate_fields)
        if model_error is None:
            counts["successful_field_true_positive"] += len(matched_fields)
            counts["successful_field_false_positive"] += len(false_positive)
            counts["successful_field_false_negative"] += len(false_negative)

        case_operation_correct = 0
        case_value_correct = 0
        for field in matched_fields:
            predicted_patch = predicted_patches[field]
            expected_patch = expected_patches[field]
            operation_matches = predicted_patch.operation == expected_patch.operation
            values_match = _normalized_values(predicted_patch.values) == _normalized_values(
                expected_patch.values
            )
            counts["operation_correct"] += int(operation_matches)
            counts["strict_value_correct"] += int(values_match)
            per_field_counts[field]["matched"] += 1
            per_field_counts[field]["operation_correct"] += int(operation_matches)
            per_field_counts[field]["strict_value_correct"] += int(values_match)
            case_operation_correct += int(operation_matches)
            case_value_correct += int(values_match)

        predicted_facts = _apply(case=case, facts=facts, understanding=predicted)
        expected_facts = _apply(case=case, facts=facts, understanding=expected)
        state_exact = True
        status_matches = 0
        value_matches = 0
        for field in FACT_FIELDS:
            predicted_state = _field_state(predicted_facts[field])
            expected_state = _field_state(expected_facts[field])
            status_matches += int(predicted_state[0] == expected_state[0])
            value_matches += int(predicted_state[1] == expected_state[1])
            per_field_counts[field]["status_correct"] += int(
                predicted_state[0] == expected_state[0]
            )
            per_field_counts[field]["value_correct"] += int(predicted_state[1] == expected_state[1])
            state_exact = state_exact and predicted_state == expected_state
        if model_error is not None:
            for field in FACT_FIELDS:
                per_field_counts[field]["status_correct"] -= int(
                    _field_state(predicted_facts[field])[0]
                    == _field_state(expected_facts[field])[0]
                )
                per_field_counts[field]["value_correct"] -= int(
                    _field_state(predicted_facts[field])[1]
                    == _field_state(expected_facts[field])[1]
                )
            status_matches = 0
            value_matches = 0
            state_exact = False
        counts["field_status_correct"] += status_matches
        counts["field_value_correct"] += value_matches
        counts["case_state_exact"] += int(state_exact)
        if model_error is None:
            counts["successful_case_state_exact"] += int(state_exact)

        revision_fields: list[str] = []
        stale_fields: list[str] = []
        for field, expected_patch in expected_patches.items():
            prior = FactValue.model_validate(facts[field])
            if prior.status != "provided" or expected_patch.operation not in {
                "replace",
                "clear",
                "mark_absent",
            }:
                continue
            revision_fields.append(field)
            counts["revision_field_total"] += 1
            predicted_state = _field_state(predicted_facts[field])
            expected_state = _field_state(expected_facts[field])
            counts["latest_value_correct"] += int(predicted_state == expected_state)
            stale_values = set(_normalized_values(prior.values)) - set(expected_state[1])
            retained = bool(stale_values.intersection(predicted_state[1]))
            counts["stale_value_retained"] += int(retained)
            if retained:
                stale_fields.append(field)

        case_results.append(
            {
                "id": case.id,
                "model": actual_model,
                "model_error": model_error,
                "latency_ms": latency_ms,
                "llm_call_count": llm_call_count,
                "tags": case.tags,
                "expected": expected.model_dump(mode="json"),
                "predicted": predicted.model_dump(mode="json"),
                "intent_correct": intent_correct,
                "clarification_correct": clarification_correct,
                "matched_fields": sorted(matched_fields),
                "false_positive_fields": sorted(false_positive),
                "false_negative_fields": sorted(false_negative),
                "operation_correct": case_operation_correct,
                "strict_value_correct": case_value_correct,
                "state_exact": state_exact,
                "status_field_matches": status_matches,
                "value_field_matches": value_matches,
                "revision_fields": revision_fields,
                "stale_value_fields": stale_fields,
                "duplicate_predicted_patch_fields": duplicate_fields,
                "manual_review_required": (
                    bool(matched_fields) and case_value_correct != len(matched_fields)
                ),
            }
        )

    true_positive = counts["field_true_positive"]
    false_positive = counts["field_false_positive"]
    false_negative = counts["field_false_negative"]
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched = counts["matched_field_total"]
    state_total = counts["field_state_total"]
    revisions = counts["revision_field_total"]
    successful_true_positive = counts["successful_field_true_positive"]
    successful_false_positive = counts["successful_field_false_positive"]
    successful_false_negative = counts["successful_field_false_negative"]
    successful_case_count = counts["model_response_success"]

    metrics = {
        "case_count": counts["case_total"],
        "model_response_success_rate": (counts["model_response_success"] / counts["case_total"]),
        "intent_accuracy": counts["intent_correct"] / counts["case_total"],
        "clarification_accuracy": counts["clarification_correct"] / counts["case_total"],
        "field_selection_precision": precision,
        "field_selection_recall": recall,
        "field_selection_f1": f1,
        "operation_accuracy": counts["operation_correct"] / matched if matched else 1.0,
        "strict_value_accuracy": counts["strict_value_correct"] / matched if matched else 1.0,
        "field_status_accuracy": counts["field_status_correct"] / state_total,
        "field_value_accuracy": counts["field_value_correct"] / state_total,
        "case_state_exact_rate": counts["case_state_exact"] / counts["case_total"],
        "unsupported_field_fill_rate": false_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0,
        "latest_value_accuracy": counts["latest_value_correct"] / revisions if revisions else 1.0,
        "stale_value_retention_rate": counts["stale_value_retained"] / revisions
        if revisions
        else 0.0,
        "duplicate_predicted_patch_count": counts["duplicate_predicted_patches"],
        "manual_review_case_count": sum(
            int(result["manual_review_required"]) for result in case_results
        ),
        "successful_response_field_recall": (
            successful_true_positive
            / (successful_true_positive + successful_false_negative)
            if successful_true_positive + successful_false_negative
            else 1.0
        ),
        "successful_response_unsupported_fill_rate": (
            successful_false_positive
            / (successful_true_positive + successful_false_positive)
            if successful_true_positive + successful_false_positive
            else 0.0
        ),
        "successful_response_case_state_exact_rate": (
            counts["successful_case_state_exact"] / successful_case_count
            if successful_case_count
            else 0.0
        ),
    }
    per_field_metrics: dict[str, dict[str, Any]] = {}
    for field, field_counts in per_field_counts.items():
        field_tp = field_counts["true_positive"]
        field_fp = field_counts["false_positive"]
        field_fn = field_counts["false_negative"]
        field_precision = field_tp / (field_tp + field_fp) if field_tp + field_fp else 1.0
        field_recall = field_tp / (field_tp + field_fn) if field_tp + field_fn else 1.0
        field_f1 = (
            2 * field_precision * field_recall / (field_precision + field_recall)
            if field_precision + field_recall
            else 0.0
        )
        field_matched = field_counts["matched"]
        per_field_metrics[field] = {
            **field_counts,
            "selection_precision": field_precision,
            "selection_recall": field_recall,
            "selection_f1": field_f1,
            "operation_accuracy": (
                field_counts["operation_correct"] / field_matched if field_matched else 1.0
            ),
            "strict_value_accuracy": (
                field_counts["strict_value_correct"] / field_matched if field_matched else 1.0
            ),
            "final_status_accuracy": field_counts["status_correct"] / len(dataset.cases),
            "final_value_accuracy": field_counts["value_correct"] / len(dataset.cases),
        }
    return {
        "schema_version": "story-worker-field-evaluation-result-v1",
        "dataset_schema_version": dataset.schema_version,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "per_field_metrics": per_field_metrics,
        "counts": counts,
        "model_error_categories": error_categories,
        "cases": case_results,
    }


def _build_live_model() -> tuple[GeminiConversationModel, RuntimeSettings]:
    settings = RuntimeSettings.from_env()
    return GeminiConversationModel(GeminiAdapter(settings, allow_fallback=False)), settings


def _precompute_live_understandings(
    dataset: FieldEvaluationDataset,
    settings: RuntimeSettings,
    *,
    workers: int,
    checkpoint_path: Path | None,
    resume: bool,
    requests_per_minute: int,
) -> _PrecomputedUnderstandingModel:
    local = threading.local()
    limiter = RequestRateLimiter(requests_per_minute)

    def model() -> UnderstandingModel:
        current = getattr(local, "model", None)
        if current is None:
            adapter = GeminiAdapter(
                settings,
                allow_fallback=False,
                before_call=limiter.wait,
            )
            current = GeminiConversationModel(adapter)
            local.model = current
        return current

    def predict(
        case: FieldEvaluationCase,
    ) -> tuple[
        str,
        TurnUnderstanding | None,
        str | None,
        Exception | None,
        float,
        int,
    ]:
        facts = _facts_for_case(case)
        current_message = {
            "id": f"{case.id}-current",
            "role": "user",
            "content": case.message,
        }
        messages = [
            {
                "id": f"{case.id}-history-{index}",
                "role": message.role,
                "content": message.content,
            }
            for index, message in enumerate(case.history, start=1)
        ]
        messages.append(current_message)
        current_model = model()
        started_at = time.perf_counter()
        try:
            output = current_model.understand_turn(
                message=current_message,
                messages=messages,
                facts=facts,
                image_path=None,
            )
            error = None
        except Exception as exc:  # noqa: BLE001
            output = None
            error = exc
        latency_ms = (time.perf_counter() - started_at) * 1000
        return (
            case.id,
            output,
            getattr(current_model, "last_model", None),
            error,
            latency_ms,
            int(getattr(current_model, "last_call_count", 1)),
        )

    completed: dict[
        str,
        tuple[TurnUnderstanding | None, str | None, Exception | None, float, int],
    ] = {}
    checkpoint_header = {
        "type": "header",
        "implementation_revision": FIELD_EVALUATION_IMPLEMENTATION_REVISION,
        "strategy": "combined",
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
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
            if not lines:
                raise ValueError("evaluation checkpoint is empty")
            header = json.loads(lines[0])
            if header != checkpoint_header:
                raise ValueError("evaluation checkpoint does not match this run")
            for line in lines[1:]:
                record = json.loads(line)
                if record.get("error") is not None or record.get("output") is None:
                    continue
                completed[record["id"]] = (
                    TurnUnderstanding.model_validate(record["output"]),
                    record.get("model"),
                    None,
                    float(record.get("latency_ms", 0.0)),
                    int(record.get("llm_call_count", 0)),
                )
        else:
            checkpoint_path.write_text(
                json.dumps(checkpoint_header, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    pending = [case for case in dataset.cases if case.id not in completed]
    total = len(dataset.cases)
    done = len(completed)
    if done:
        print(f"resumed {done}/{total} successful cases", flush=True)
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(predict, case): case.id for case in pending}
    try:
        for future in as_completed(futures):
            prediction = future.result()
            case_id, output, actual_model, error, latency_ms, call_count = prediction
            completed[case_id] = (output, actual_model, error, latency_ms, call_count)
            done += 1
            if checkpoint_path is not None:
                record = {
                    "type": "case",
                    "id": case_id,
                    "model": actual_model,
                    "output": output.model_dump(mode="json") if output is not None else None,
                    "error": (
                        {"type": type(error).__name__, "message": str(error)}
                        if error is not None
                        else None
                    ),
                    "latency_ms": latency_ms,
                    "llm_call_count": call_count,
                }
                with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
                    checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
            if done % 5 == 0 or done == total or error is not None:
                error_count = sum(item[2] is not None for item in completed.values())
                print(
                    f"progress {done}/{total} errors={error_count}",
                    flush=True,
                )
    except KeyboardInterrupt:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return _PrecomputedUnderstandingModel(completed)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Gemini fact field construction for story-maker-worker"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/datasets/story-worker-field-evaluation-v1.json"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--request-timeout-ms", type=int, default=60000)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--requests-per-minute", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Append per-case results to this JSONL file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful cases from --checkpoint and retry the rest",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate dataset size and schema without calling Gemini",
    )
    args = parser.parse_args()
    dataset = load_field_evaluation_dataset(args.dataset)
    if args.validate_only:
        field_counts = Counter(
            patch.field for case in dataset.cases for patch in case.expected.fact_patches
        )
        operation_counts = Counter(
            patch.operation for case in dataset.cases for patch in case.expected.fact_patches
        )
        print(
            json.dumps(
                {
                    "schema_version": dataset.schema_version,
                    "case_count": len(dataset.cases),
                    "field_positive_counts": dict(field_counts),
                    "operation_counts": dict(operation_counts),
                    "valid": True,
                },
                ensure_ascii=False,
            )
        )
        return

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

    _, base_settings = _build_live_model()
    settings = replace(
        base_settings,
        request_timeout_ms=args.request_timeout_ms,
        primary_access_attempts=args.attempts,
    )
    model = _precompute_live_understandings(
        dataset,
        settings,
        workers=args.workers,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        requests_per_minute=args.requests_per_minute,
    )
    result = evaluate_field_cases(dataset, model)
    successful_cases = [
        case for case in result["cases"] if case.get("model_error") is None
    ]
    latencies = [float(case["latency_ms"]) for case in successful_cases]
    calls = [int(case["llm_call_count"]) for case in successful_cases]
    result["metrics"].update(
        {
            "average_llm_calls": sum(calls) / len(calls) if calls else 0.0,
            "latency_p50_ms": _percentile(latencies, 0.5),
            "latency_p95_ms": _percentile(latencies, 0.95),
        }
    )
    result["runtime"] = {
        "strategy": "combined",
        "primary_model": settings.primary_model,
        "fallback_model": settings.fallback_model,
        "fallback_enabled": False,
        "location": settings.location,
        "request_timeout_ms": settings.request_timeout_ms,
        "primary_access_attempts": settings.primary_access_attempts,
        "requests_per_minute": args.requests_per_minute,
        "thinking_level": settings.thinking_level,
        "max_output_tokens": settings.max_output_tokens,
        "workers": args.workers,
    }
    output = args.output or Path(
        "artifacts/evaluations/story-worker-field-evaluation-live.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
