from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import re
import threading
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv

from funding_story_ai.adapter import GeminiAdapter
from funding_story_ai.config import RuntimeSettings
from funding_story_ai.semantic_normalization import (
    SEMANTIC_RESPONSE_SCHEMA,
    semantic_prompt,
    semantic_repair_prompt,
    validated_semantic_propositions,
)

EVAL_ROOT = Path(__file__).parent
BASE_VALIDATOR_PATH = EVAL_ROOT / "validate_semantic_normalization_dataset.py"
BASE_SPEC = importlib.util.spec_from_file_location(
    "semantic_dataset_validator", BASE_VALIDATOR_PATH
)
assert BASE_SPEC is not None and BASE_SPEC.loader is not None
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)

DEV_ROOT = EVAL_ROOT / "datasets" / "robot-vacuum-semantic-normalization-v1"
HOLDOUT_ROOT = EVAL_ROOT / "datasets" / "robot-vacuum-semantic-normalization-holdout-v1"
ADVERSARIAL_ROOT = (
    EVAL_ROOT / "datasets" / "robot-vacuum-semantic-normalization-adversarial-v1"
)

DETERMINISTIC_GROUPS = {
    "product": "product_identity_outcome",
    "problem": "problem_environment",
    "evidence": "evidence_performance",
}


class JsonAdapter(Protocol):
    def generate_json(
        self, *, prompt: str, response_schema: dict[str, Any]
    ) -> Any: ...


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(value: str) -> str:
    return " ".join(value.strip().rstrip(".!?。 ").split())


def _grounded_segments(statement: str) -> set[str]:
    segments = {_normalize(statement)}
    for segment in re.split(r"\s+그리고\s+|[;\n]+|(?<=[.!?。])\s+", statement):
        normalized = _normalize(segment)
        if len(normalized) >= 10:
            segments.add(normalized)
    segments.update(_normalize(match) for match in base.QUOTED_FACT.findall(statement))
    return segments


def _proposition_id(case_id: str, fact_id: str, text: str, group: str) -> str:
    payload = "\0".join((case_id, fact_id, _normalize(text), group)).encode()
    return "p_" + hashlib.sha256(payload).hexdigest()[:12]


def _adapter_result(decision: str) -> dict[str, Any]:
    return {
        "decision": decision,
        "failure_code": base.FAILURE_CODE_BY_DECISION[decision],
        "owns": ["availability", "support_level", "collection_state"],
    }


def _semantic_model_view(model_view: dict[str, Any]) -> dict[str, Any]:
    return {
        "facts": [
            fact for fact in model_view["facts"] if fact["entity_kind"] != "distractor"
        ]
    }


def _prompt(model_view: dict[str, Any]) -> str:
    return semantic_prompt(_semantic_model_view(model_view))


def _repair_prompt(
    model_view: dict[str, Any], invalid_response: dict[str, Any], validation_error: str
) -> str:
    return semantic_repair_prompt(
        _semantic_model_view(model_view), invalid_response, validation_error
    )


def _semantic_output(
    *, case: dict[str, Any], response: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    facts = case["model_view"]["facts"]
    classified_facts = [fact for fact in facts if fact["entity_kind"] != "distractor"]
    ignored = [fact["fact_id"] for fact in facts if fact["entity_kind"] == "distractor"]
    propositions = validated_semantic_propositions(
        facts=classified_facts, response=response
    )
    return propositions, ignored


def _join_media_facts(
    case: dict[str, Any], propositions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    projection = case["input"]["approved_entity_projection"]
    facts = {fact["fact_id"]: fact for fact in projection["facts"]}
    states = {
        state["fact_id"]: state for state in case["input"]["worker_projection"]["fact_states"]
    }
    proposition_ids: dict[str, list[str]] = {}
    for proposition in propositions:
        proposition_ids.setdefault(proposition["fact_id"], []).append(
            proposition["proposition_id"]
        )
    result = []
    for fact_id, ids in proposition_ids.items():
        fact = facts[fact_id]
        state = states[fact_id]
        result.append(
            {
                "fact_id": fact_id,
                "proposition_ids": ids,
                "source_refs": fact["source_refs"],
                "evidence_refs": fact["evidence_refs"],
                "asset_refs": fact["asset_refs"],
                "availability": state["availability"],
                "support_level": state["support_level"],
                "collection_state": state["collection_state"],
            }
        )
    return result


def run_case_detailed(
    case: dict[str, Any], adapter: JsonAdapter
) -> tuple[dict[str, Any], str | None, dict[str, Any]]:
    unresolved = base._unresolved_links(case)
    decision = base._expected_decision(case, unresolved)
    if decision != "accepted":
        return {
            "adapter": _adapter_result(decision),
            "atomic_propositions": [],
            "media_facts": [],
            "ignored_fact_ids": [],
        }, None, {"semantic_model_call_count": 0, "repair_used": False}
    model_call_count = 0
    try:
        model_call_count += 1
        result = adapter.generate_json(
            prompt=_prompt(case["model_view"]),
            response_schema=SEMANTIC_RESPONSE_SCHEMA,
        )
    except Exception as exc:
        return {
            "adapter": _adapter_result(decision),
            "atomic_propositions": [],
            "media_facts": [],
            "ignored_fact_ids": [],
        }, f"{type(exc).__name__}: {exc}", {
            "semantic_model_call_count": model_call_count,
            "repair_used": False,
        }
    repair_used = False
    try:
        propositions, ignored = _semantic_output(case=case, response=result.data)
    except ValueError as validation_error:
        repair_used = True
        try:
            model_call_count += 1
            repaired = adapter.generate_json(
                prompt=_repair_prompt(
                    case["model_view"], result.data, str(validation_error)
                ),
                response_schema=SEMANTIC_RESPONSE_SCHEMA,
            )
            propositions, ignored = _semantic_output(case=case, response=repaired.data)
        except Exception as repair_error:
            return {
                "adapter": _adapter_result(decision),
                "atomic_propositions": [],
                "media_facts": [],
                "ignored_fact_ids": [],
            }, (
                f"initial_validation={type(validation_error).__name__}: {validation_error}; "
                f"repair={type(repair_error).__name__}: {repair_error}"
            ), {
                "semantic_model_call_count": model_call_count,
                "repair_used": True,
            }
    try:
        return {
            "adapter": _adapter_result(decision),
            "atomic_propositions": propositions,
            "media_facts": _join_media_facts(case, propositions),
            "ignored_fact_ids": ignored,
        }, None, {
            "semantic_model_call_count": model_call_count,
            "repair_used": repair_used,
        }
    except Exception as exc:
        return {
            "adapter": _adapter_result(decision),
            "atomic_propositions": [],
            "media_facts": [],
            "ignored_fact_ids": [],
        }, f"{type(exc).__name__}: {exc}", {
            "semantic_model_call_count": model_call_count,
            "repair_used": repair_used,
        }


def run_case(case: dict[str, Any], adapter: JsonAdapter) -> tuple[dict[str, Any], str | None]:
    output, error, _ = run_case_detailed(case, adapter)
    return output, error


def run_cases(
    cases: list[dict[str, Any]], adapter: JsonAdapter
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions = []
    errors = []
    for case in cases:
        output, error = run_case(case, adapter)
        predictions.append({"case_id": case["case_id"], "output": output})
        if error is not None:
            errors.append({"case_id": case["case_id"], "error": error})
    return (
        {
            "schema_version": "semantic-normalization-predictions-v1",
            "predictions": predictions,
        },
        {
            "schema_version": "semantic-normalization-run-errors-v1",
            "case_count": len(cases),
            "error_count": len(errors),
            "errors": errors,
        },
    )


def run_live_cases(
    cases: list[dict[str, Any]],
    settings: RuntimeSettings,
    *,
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    local = threading.local()

    def invoke(
        case: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str | None, dict[str, Any]]:
        adapter = getattr(local, "adapter", None)
        if adapter is None:
            adapter = GeminiAdapter(settings)
            local.adapter = adapter
        output, error, diagnostics = run_case_detailed(case, adapter)
        return case["case_id"], output, error, diagnostics

    rows: dict[str, tuple[dict[str, Any], str | None, dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(invoke, case): case["case_id"] for case in cases}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            case_id, output, error, diagnostics = future.result()
            rows[case_id] = output, error, diagnostics
            completed += 1
            status = "repaired" if diagnostics["repair_used"] and error is None else "ok"
            if error is not None:
                status = "error"
            print(f"[{completed}/{len(cases)}] {case_id}: {status}", flush=True)

    predictions = {
        "schema_version": "semantic-normalization-predictions-v1",
        "predictions": [
            {"case_id": case["case_id"], "output": rows[case["case_id"]][0]}
            for case in cases
        ],
    }
    errors = [
        {"case_id": case["case_id"], "error": rows[case["case_id"]][1]}
        for case in cases
        if rows[case["case_id"]][1] is not None
    ]
    diagnostics = [
        {"case_id": case["case_id"], **rows[case["case_id"]][2]} for case in cases
    ]
    return predictions, {
        "schema_version": "semantic-normalization-run-errors-v1",
        "case_count": len(cases),
        "error_count": len(errors),
        "errors": errors,
        "semantic_model_call_count": sum(
            item["semantic_model_call_count"] for item in diagnostics
        ),
        "repair_case_count": sum(item["repair_used"] for item in diagnostics),
        "repair_case_ids": [item["case_id"] for item in diagnostics if item["repair_used"]],
    }


def _dataset_path(split: str) -> Path:
    return {
        "normal": DEV_ROOT / "normal-cases.json",
        "defensive": DEV_ROOT / "defensive-cases.json",
        "holdout": HOLDOUT_ROOT / "holdout-cases.json",
        "adversarial": ADVERSARIAL_ROOT / "adversarial-cases.json",
    }[split]


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run live Gemini semantic normalization")
    parser.add_argument(
        "--split",
        choices=("normal", "defensive", "holdout", "adversarial"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--errors-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    settings = RuntimeSettings.from_env()
    predictions, errors = run_live_cases(
        _load(_dataset_path(args.split)), settings, workers=args.workers
    )
    args.output.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.errors_output.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(errors, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
