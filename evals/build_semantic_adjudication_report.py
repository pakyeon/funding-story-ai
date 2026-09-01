from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
DATASET_PATH = (
    ROOT / "datasets" / "robot-vacuum-semantic-normalization-v1" / "normal-cases.json"
)
ANNOTATION_ROOT = ROOT / "adjudication" / "semantic-normalization-v1"
ANNOTATOR_A_PATH = ANNOTATION_ROOT / "annotator-a.json"
ANNOTATOR_B_PATH = ANNOTATION_ROOT / "annotator-b.json"
DEFAULT_OUTPUT = ANNOTATION_ROOT / "adjudication-report.json"
HUMAN_SIGNOFF_PATH = ANNOTATION_ROOT / "human-signoff.json"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().rstrip(".!?。 ").split())


def _annotation_index(document: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for case in document["cases"]:
        for annotation in case["annotations"]:
            key = case["case_id"], annotation["fact_id"]
            if key in index:
                raise ValueError(f"duplicate annotation: {key}")
            index[key] = annotation
    return index


def _gold_annotation(case: dict[str, Any], fact_id: str) -> dict[str, Any]:
    propositions = [
        {
            "text": proposition["text"],
            "capability_group": proposition["capability_group"],
        }
        for proposition in case["expected"]["atomic_propositions"]
        if proposition["fact_id"] == fact_id
    ]
    return {
        "decision": "ignore" if fact_id in case["expected"]["ignored_fact_ids"] else "classify",
        "propositions": propositions,
    }


def _exact_signature(annotation: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    return annotation["decision"], tuple(
        sorted(
            (
                _normalize_text(proposition["text"]),
                proposition["capability_group"],
            )
            for proposition in annotation["propositions"]
        )
    )


def _label_signature(annotation: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    return annotation["decision"], tuple(
        sorted({proposition["capability_group"] for proposition in annotation["propositions"]})
    )


def build_report() -> dict[str, Any]:
    cases = _load(DATASET_PATH)[:16]
    annotator_a = _load(ANNOTATOR_A_PATH)
    annotator_b = _load(ANNOTATOR_B_PATH)
    a_index = _annotation_index(annotator_a)
    b_index = _annotation_index(annotator_b)
    human_signoff = _load(HUMAN_SIGNOFF_PATH) if HUMAN_SIGNOFF_PATH.exists() else None

    fact_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    expected_keys: set[tuple[str, str]] = set()
    for case in cases:
        for fact in case["model_view"]["facts"]:
            key = case["case_id"], fact["fact_id"]
            expected_keys.add(key)
            fact_rows.append((case, fact))
    if set(a_index) != expected_keys or set(b_index) != expected_keys:
        raise ValueError("annotator coverage does not match the frozen 16-case packet")

    exact_agreement = 0
    label_agreement = 0
    decision_agreement = 0
    a_gold_exact = 0
    b_gold_exact = 0
    disagreements: list[dict[str, Any]] = []
    for case, fact in fact_rows:
        key = case["case_id"], fact["fact_id"]
        a = a_index[key]
        b = b_index[key]
        gold = _gold_annotation(case, fact["fact_id"])
        exact_same = _exact_signature(a) == _exact_signature(b)
        labels_same = _label_signature(a) == _label_signature(b)
        decisions_same = a["decision"] == b["decision"]
        exact_agreement += exact_same
        label_agreement += labels_same
        decision_agreement += decisions_same
        a_gold_exact += _exact_signature(a) == _exact_signature(gold)
        b_gold_exact += _exact_signature(b) == _exact_signature(gold)
        if not exact_same:
            disagreement_types: list[str] = []
            if not decisions_same:
                disagreement_types.append("decision")
            if not labels_same:
                disagreement_types.append("capability_label_set")
            if len(a["propositions"]) != len(b["propositions"]):
                disagreement_types.append("proposition_count")
            if {
                _normalize_text(item["text"]) for item in a["propositions"]
            } != {_normalize_text(item["text"]) for item in b["propositions"]}:
                disagreement_types.append("clause_boundary_or_text")
            disagreements.append(
                {
                    "case_id": case["case_id"],
                    "fact_id": fact["fact_id"],
                    "statement": fact["statement"],
                    "types": disagreement_types,
                    "annotator_a": {
                        "decision": a["decision"],
                        "propositions": a["propositions"],
                    },
                    "annotator_b": {
                        "decision": b["decision"],
                        "propositions": b["propositions"],
                    },
                    "provisional_gold": gold,
                }
            )

    count = len(fact_rows)
    return {
        "schema_version": "semantic-normalization-adjudication-report-v1",
        "scope": {
            "case_ids": [case["case_id"] for case in cases],
            "case_count": len(cases),
            "fact_count": count,
            "annotation_type": "independent_ai_double_annotation",
            "human_annotation_claimed": False,
        },
        "annotators": [
            {
                "name": annotator_a["annotator"],
                "model": annotator_a["model"],
                "reasoning_effort": annotator_a["reasoning_effort"],
            },
            {
                "name": annotator_b["annotator"],
                "model": annotator_b["model"],
                "reasoning_effort": annotator_b["reasoning_effort"],
            },
        ],
        "agreement": {
            "decision_exact_count": decision_agreement,
            "decision_exact_rate": decision_agreement / count,
            "capability_label_set_exact_count": label_agreement,
            "capability_label_set_exact_rate": label_agreement / count,
            "full_annotation_exact_count": exact_agreement,
            "full_annotation_exact_rate": exact_agreement / count,
            "annotator_a_to_provisional_gold_exact_count": a_gold_exact,
            "annotator_a_to_provisional_gold_exact_rate": a_gold_exact / count,
            "annotator_b_to_provisional_gold_exact_count": b_gold_exact,
            "annotator_b_to_provisional_gold_exact_rate": b_gold_exact / count,
        },
        "provisional_adjudication": {
            "status": (
                "human_signoff_complete" if human_signoff else "human_signoff_required"
            ),
            "human_signoff": human_signoff,
            "gold_change_count": 1,
            "gold_changes": [
                {
                    "case_id": "rv_semantic_normal_006",
                    "fact_id": "f_58c2c007dde6",
                    "before": "evidence_performance",
                    "after": "configuration_maintenance",
                    "reason": (
                        "두 독립 주석자가 모두 후기의 출처보다 브러시 관리라는 주장 내용을 "
                        "분류 대상으로 판단했다."
                    ),
                }
            ],
            "retained_policies": [
                (
                    "같은 능력군·상태·근거를 공유하는 한 문장 안의 병렬 동작은 하나의 "
                    "proposition으로 유지한다."
                ),
                "'그리고'로 연결된 독립 서술이나 서로 다른 능력군·상태·근거를 갖는 절만 분리한다.",
                (
                    "방 선택·금지 영역·지도 기반 재청소 지정은 물리적 주행 능력이 아니라 "
                    "사용자 설정인 control_personalization으로 분류한다."
                ),
            ],
            "review_items": [
                "gold 1건의 evidence_performance → configuration_maintenance 수정 승인",
                "동일 능력군 병렬 동작을 합쳐 두는 clause boundary 정책 승인",
                "방 선택·금지 영역·지도 기반 지정의 control_personalization 분류 승인",
            ],
        },
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the semantic double-annotation report")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["agreement"], ensure_ascii=False, indent=2))
    print(f"disagreements={report['disagreement_count']}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
