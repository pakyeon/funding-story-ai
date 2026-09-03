from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "evals" / "datasets"

OPTIONAL_GROUPS = {
    "story_persuasion": ["problem_context", "trust_elements", "maker_team_intro"],
    "funding_configuration": ["rewards", "funding_end", "shipping_start"],
    "policy": ["refund_policy", "as_policy"],
    "project_explanation": ["funding_plan", "risk_response"],
}
REQUIRED_FIELDS = [
    "product_name",
    "product_type",
    "category",
    "key_strengths",
    "target_supporters",
]
ALL_FIELDS = [
    *REQUIRED_FIELDS,
    *(field for fields in OPTIONAL_GROUPS.values() for field in fields),
]

PRODUCTS = [
    ("오빗클린 V3", "로봇청소기", "테크·가전", "얇은 본체", "맞벌이 가구"),
    ("브리즈팟", "탁상용 공기청정기", "테크·가전", "저소음 정화", "원룸 거주자"),
    ("슬립온", "수면 안대", "홈·리빙", "빛 차단 구조", "교대 근무자"),
    ("트레일라이트", "초경량 텐트", "스포츠·아웃도어", "1kg 미만 무게", "솔로 캠퍼"),
    ("데일리결", "기초 화장품", "뷰티", "무향 보습", "민감성 피부 사용자"),
    ("펫세이프", "반려동물 급식기", "반려동물", "정량 급여", "직장인 반려인"),
    ("리드메이트", "전자책 단말기", "도서·전자책", "눈부심 저감", "장시간 독서가"),
    ("키친큐브", "밀폐용기", "푸드", "모듈형 적층", "소형 주방 사용자"),
    ("워크핏", "노트북 거치대", "테크·가전", "높이 조절", "재택근무자"),
    ("뮤트키", "저소음 키보드", "테크·가전", "저소음 스위치", "공유 사무실 사용자"),
]


def write(name: str, payload: dict[str, Any]) -> None:
    path = DATASET_ROOT / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def patch(field: str, value: str | list[str], operation: str = "replace") -> dict[str, Any]:
    values = value if isinstance(value, list) else [value]
    if operation in {"clear", "mark_absent"}:
        values = []
    return {"field": field, "operation": operation, "values": values}


def understanding(
    *,
    intent: str = "provide_information",
    patches: list[dict[str, Any]] | None = None,
    directive: dict[str, Any] | None = None,
    requires_clarification: bool = False,
    clarification_question: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {"intent": intent, "fact_patches": patches or []}
    if directive is not None:
        value["collection_directive"] = directive
    if requires_clarification:
        value.update(
            {
                "requires_clarification": True,
                "clarification_question": clarification_question,
            }
        )
    return value


def fact_value(value: str | list[str] | None, *, status: str = "provided") -> dict[str, Any]:
    values = [] if value is None else (value if isinstance(value, list) else [value])
    return {"status": status, "values": values}


def expand_field_dataset() -> None:
    path = DATASET_ROOT / "story-worker-field-evaluation-v1.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["cases"][:50]
    for index, case in enumerate(cases, start=1):
        case["id"] = f"field-{index:03d}"
    additions: list[dict[str, Any]] = []

    for index, (name, product_type, category, strength, target) in enumerate(PRODUCTS, 51):
        additions.append(
            {
                "id": f"field-{index:03d}",
                "description": f"{product_type} 필수 정보 장문 입력",
                "message": (
                    f"제품명은 {name}이고 {category} 카테고리의 {product_type}야. "
                    f"핵심 강점은 {strength}이고 주요 서포터는 {target}이야."
                ),
                "expected": understanding(
                    patches=[
                        patch("product_name", name),
                        patch("product_type", product_type),
                        patch("category", category),
                        patch("key_strengths", strength),
                        patch("target_supporters", target),
                    ]
                ),
                "tags": ["initial", "required-fields", "long-input", product_type],
            }
        )

    optional_specs = [
        (
            "problem_context",
            "매일 반복되는 바닥 청소 시간이 부담됨",
            "해결하려는 문제는 매일 반복되는 바닥 청소 시간이 부담된다는 점이야.",
        ),
        ("trust_elements", "KC 인증 완료", "신뢰 근거로 KC 인증을 완료했어."),
        (
            "maker_team_intro",
            "생활가전 개발자 3명으로 구성된 팀",
            "메이커는 생활가전 개발자 3명으로 구성된 팀이야.",
        ),
        ("rewards", "얼리버드 1대 39만 원", "리워드는 얼리버드 1대 39만 원이야."),
        ("funding_end", "2026년 11월 30일", "펀딩 종료일은 2026년 11월 30일이야."),
        (
            "shipping_start",
            "2026년 12월 15일부터 순차 발송",
            "리워드는 2026년 12월 15일부터 순차 발송할게.",
        ),
        (
            "refund_policy",
            "제품 하자는 수령 후 14일 이내 환불",
            "환불 정책은 제품 하자가 있으면 수령 후 14일 이내 환불이야.",
        ),
        (
            "as_policy",
            "배송 완료일부터 1년 무상 수리",
            "A/S는 배송 완료일부터 1년간 무상 수리로 제공해.",
        ),
        (
            "funding_plan",
            "초도 생산과 품질 검사",
            "모인 펀딩금은 초도 생산과 품질 검사에 사용할 계획이야.",
        ),
        (
            "risk_response",
            "부품 공급사 이원화로 수급 지연 대응",
            "부품 수급 지연 위험에는 공급사 이원화로 대응할게.",
        ),
    ]
    for offset, (field, value, message) in enumerate(optional_specs, 61):
        additions.append(
            {
                "id": f"field-{offset:03d}",
                "description": f"선택 필드 {field} 자연어 입력",
                "message": message,
                "expected": understanding(patches=[patch(field, value)]),
                "tags": ["initial", "optional-field", field],
            }
        )

    revisions = [
        (
            "product_name",
            "오빗클린 V2",
            "오빗클린 V4",
            "제품명은 오빗클린 V2가 아니라 오빗클린 V4로 바꿀게.",
        ),
        (
            "product_type",
            "로봇청소기",
            "물걸레 로봇청소기",
            "제품 종류는 로봇청소기에서 물걸레 로봇청소기로 바꿀게.",
        ),
        (
            "category",
            "홈·리빙",
            "테크·가전",
            "카테고리는 홈·리빙이 아니라 테크·가전으로 수정할게.",
        ),
        (
            "key_strengths",
            "자동 먼지 비움",
            "8.0cm 초슬림 본체",
            "핵심 강점은 자동 먼지 비움 대신 8.0cm 초슬림 본체로 바꿀게.",
        ),
        (
            "target_supporters",
            "1인 가구",
            "반려동물과 사는 맞벌이 가구",
            "핵심 타깃은 1인 가구에서 반려동물과 사는 맞벌이 가구로 바꿀게.",
        ),
        (
            "rewards",
            "본체 1대 49만 원",
            "본체 1대 45만 원",
            "리워드는 본체 1대 49만 원에서 45만 원으로 변경할게.",
        ),
        (
            "funding_end",
            "2026년 10월 10일",
            "2026년 10월 20일",
            "펀딩 종료일은 2026년 10월 10일에서 20일로 변경할게.",
        ),
        (
            "shipping_start",
            "2026년 11월 1일",
            "2026년 11월 15일",
            "발송 시작일은 2026년 11월 1일에서 15일로 변경할게.",
        ),
        (
            "refund_policy",
            "7일 이내 환불",
            "14일 이내 환불",
            "환불 가능 기간은 7일에서 14일로 늘릴게.",
        ),
        (
            "as_policy",
            "6개월 무상 수리",
            "1년 무상 수리",
            "무상 수리 기간은 6개월에서 1년으로 늘릴게.",
        ),
    ]
    for offset, (field, old, new, message) in enumerate(revisions, 71):
        additions.append(
            {
                "id": f"field-{offset:03d}",
                "description": f"{field} 최신값 교체",
                "history": [{"role": "user", "content": f"이전 입력 값은 {old}이야."}],
                "prior_facts": {field: fact_value(old)},
                "message": message,
                "expected": understanding(intent="revise_information", patches=[patch(field, new)]),
                "tags": ["follow-up", "revision", field],
            }
        )

    absent_clear = [
        ("trust_elements", "인증 자료는 아직 없어.", "mark_absent"),
        ("maker_team_intro", "팀 소개는 이번에는 제공하지 않을게.", "mark_absent"),
        ("refund_policy", "환불 정책은 아직 정하지 못했어.", "clear"),
        ("as_policy", "A/S 기간은 미정으로 돌려줘.", "clear"),
        ("risk_response", "위험 대응 계획은 현재 없어.", "mark_absent"),
        ("funding_plan", "사용 계획은 아직 미정이야.", "clear"),
        ("rewards", "리워드 가격은 미정으로 바꿔줘.", "clear"),
        ("shipping_start", "발송 일정은 아직 없어.", "clear"),
        ("problem_context", "해결 문제는 우선 미정으로 둘게.", "clear"),
        ("rewards", "이번 스토리에는 별도 리워드가 없어.", "mark_absent"),
    ]
    for offset, (field, message, operation) in enumerate(absent_clear, 81):
        if field == "maker_team_intro":
            expected = understanding(directive={"action": "skip_fields", "fields": [field]})
            tags = ["follow-up", "skip_fields", field]
        else:
            expected = understanding(
                intent="revise_information", patches=[patch(field, [], operation)]
            )
            tags = ["follow-up", operation, field]
        additions.append(
            {
                "id": f"field-{offset:03d}",
                "description": f"{field} {operation}",
                "prior_facts": {field: fact_value("이전에 입력한 값")},
                "message": message,
                "expected": expected,
                "tags": tags,
            }
        )

    edge_cases = [
        ("이제 바로 스토리를 생성해줘.", "request_generation"),
        ("스토리 생성은 취소할게.", "cancel"),
        ("지금 확인된 정보가 뭐야?", "ask_question"),
        ("그건 아까 말한 걸로 해줘.", "unclear"),
        ("조금 더 좋은 방향으로 알아서 정리해줘.", "unclear"),
        ("온라인에서 공개할 거야.", "provide_information"),
        ("제품 설명 예시를 하나 보여줘.", "ask_question"),
        ("생성하지 말고 여기서 멈춰.", "cancel"),
        ("앞의 값은 지우고 아직 미정으로 해줘.", "unclear"),
        ("이 정보로 생성할 수 있는지 알려줘.", "ask_question"),
    ]
    for offset, (message, intent) in enumerate(edge_cases, 91):
        clarify = intent == "unclear"
        additions.append(
            {
                "id": f"field-{offset:03d}",
                "description": f"의도 분류 {intent}",
                "message": message,
                "expected": understanding(
                    intent=intent,
                    requires_clarification=clarify,
                    clarification_question=(
                        "어떤 정보를 뜻하는지 구체적으로 알려주세요." if clarify else None
                    ),
                ),
                "tags": ["intent", intent],
            }
        )

    additions.extend(
        [
            {
                "id": "field-coverage-maker",
                "description": "메이커 소개 추가 입력",
                "message": "메이커는 생활가전 제품을 개발해 온 3인 팀이야.",
                "expected": understanding(
                    patches=[patch("maker_team_intro", "생활가전 제품을 개발해 온 3인 팀")]
                ),
                "tags": ["optional-field", "maker_team_intro"],
            },
            {
                "id": "field-coverage-plan",
                "description": "펀딩금 사용 계획 추가 입력",
                "message": "펀딩금은 금형 제작과 초도 생산에 사용할 계획이야.",
                "expected": understanding(patches=[patch("funding_plan", "금형 제작과 초도 생산")]),
                "tags": ["optional-field", "funding_plan"],
            },
        ]
    )
    payload["cases"] = [*cases, *additions]
    for index, case in enumerate(payload["cases"], start=1):
        case["id"] = f"field-{index:03d}"
    write("story-worker-field-evaluation-v1.json", payload)


def build_collection_dataset() -> None:
    variants: dict[str, list[str]] = {
        "continue_recommended": [
            "권장 순서대로 모두 입력할게.",
            "추천하는 순서로 선택 정보를 채우자.",
            "네가 권장하는 흐름으로 전부 진행해줘.",
            "선택 항목을 처음부터 차례대로 물어봐 줘.",
            "모든 추가 정보를 순서대로 입력하겠습니다.",
            "전체 항목을 권장 순서로 진행할래.",
            "빠짐없이 하나씩 질문해줘.",
            "네, 선택 정보 모두 작성하겠습니다.",
            "추천 방식으로 계속하자.",
            "네 가지 그룹을 전부 순서대로 입력할게.",
        ],
        "skip_all_optional": [
            "선택 정보는 전부 생략할게.",
            "추가 항목 10개는 모두 이번 생성에서 빼줘.",
            "선택 정보 전체를 건너뛰겠습니다.",
            "필수 정보만 쓰고 나머지는 전부 생략해.",
            "네 그룹 모두 작성하지 않을게.",
            "추가 질문 없이 선택 항목을 모두 스킵해줘.",
            "선택 정보는 하나도 제공하지 않겠습니다.",
            "남은 선택 항목 전체 생략을 확정할게.",
            "이번 생성에서는 추가 정보를 전부 제외해.",
            "선택 사항 전체를 생략하고 다음으로 가자.",
        ],
        "return_to_optional": [
            "아까 생략한 선택 정보를 다시 입력할게.",
            "선택 정보 입력 단계로 돌아가자.",
            "생략을 취소하고 추가 항목을 작성하겠습니다.",
            "다시 선택 정보부터 보완하고 싶어.",
            "건너뛴 항목을 다시 열어줘.",
            "추가 정보 입력으로 되돌아갈게.",
            "선택 항목을 다시 작성할 수 있게 해줘.",
            "이전 생략을 취소할래.",
            "선택 정보 수집을 재개하자.",
            "아까 스킵한 부분을 다시 채우겠습니다.",
        ],
    }
    cases: list[dict[str, Any]] = []
    case_id = 1

    for action, messages in variants.items():
        for message in messages:
            cases.append(
                {
                    "id": f"collection-{case_id:03d}",
                    "message": message,
                    "expected_directive": {"action": action},
                    "expected_intent": "provide_information",
                    "tags": [action],
                }
            )
            case_id += 1

    group_phrases = {
        "story_persuasion": ["스토리 설득 정보", "문제와 신뢰 근거", "메이커 소개 그룹"],
        "funding_configuration": ["펀딩 구성", "리워드와 일정", "가격과 발송 정보"],
        "policy": ["정책 그룹", "환불과 A/S", "교환·보증 정보"],
        "project_explanation": ["프로젝트 설명", "펀딩 목적과 위험", "자금 계획과 위험 대응"],
    }
    for group, phrases in group_phrases.items():
        for repeat in range(4):
            phrase = phrases[repeat % len(phrases)]
            cases.append(
                {
                    "id": f"collection-{case_id:03d}",
                    "message": f"{phrase}에 해당하는 그룹 전체부터 입력할게.",
                    "expected_directive": {"action": "select_groups", "groups": [group]},
                    "expected_intent": "provide_information",
                    "tags": ["select_groups", group],
                }
            )
            case_id += 1

    field_choices = [
        ("problem_context", "해결하려는 문제만 입력할게."),
        ("trust_elements", "신뢰 근거부터 작성하자."),
        ("maker_team_intro", "메이커 소개 항목을 선택할게."),
        ("rewards", "리워드 구성만 알려줄게."),
        ("funding_end", "펀딩 종료 일정부터 입력할래."),
        ("shipping_start", "발송 시작 일정을 선택해줘."),
        ("refund_policy", "교환·환불 정책만 작성하겠습니다."),
        ("as_policy", "보증과 사후지원 내용을 입력할게."),
        ("funding_plan", "펀딩금 사용 계획부터 작성하자."),
        ("risk_response", "위험과 대응 계획만 보완할게."),
    ]
    for repeat in range(2):
        for field, message in field_choices:
            cases.append(
                {
                    "id": f"collection-{case_id:03d}",
                    "message": message if repeat == 0 else f"이번에는 {message}",
                    "expected_directive": {"action": "select_fields", "fields": [field]},
                    "expected_intent": "provide_information",
                    "tags": ["select_fields", field],
                }
            )
            case_id += 1

    skip_specs = [
        (["problem_context"], "해결하려는 문제 항목은 이번에 생략할게."),
        (["trust_elements"], "신뢰 근거는 빼고 진행하자."),
        (["maker_team_intro"], "메이커 소개는 이번 생성에서 제외해."),
        (["rewards"], "리워드 구성은 생략하겠습니다."),
        (["funding_end", "shipping_start"], "종료일과 발송 일정은 둘 다 생략할게."),
        (["refund_policy", "as_policy"], "환불과 A/S 정책은 이번에 건너뛰자."),
        (["funding_plan"], "펀딩금 사용 계획은 빼줘."),
        (["risk_response"], "위험 대응 계획은 이번에 작성하지 않겠어."),
        (OPTIONAL_GROUPS["story_persuasion"], "스토리 설득 정보 그룹 전체를 생략해."),
        (OPTIONAL_GROUPS["funding_configuration"], "펀딩 구성 그룹은 전부 건너뛰어."),
        (OPTIONAL_GROUPS["policy"], "정책 그룹 전체를 제외할게."),
        (OPTIONAL_GROUPS["project_explanation"], "프로젝트 설명 그룹은 모두 생략하자."),
        (["rewards", "refund_policy"], "리워드와 환불 정책만 이번에 빼줘."),
        (["maker_team_intro", "risk_response"], "팀 소개와 위험 대응 계획은 생략할게."),
        (["trust_elements", "risk_response"], "신뢰 근거와 위험 대응 계획은 건너뛰자."),
    ]
    for fields, message in skip_specs:
        cases.append(
            {
                "id": f"collection-{case_id:03d}",
                "message": message,
                "expected_directive": {"action": "skip_fields", "fields": fields},
                "expected_intent": "provide_information",
                "tags": ["skip_fields"],
            }
        )
        case_id += 1

    ambiguous = [
        "괜찮아.",
        "그냥 진행해줘.",
        "다음으로 가자.",
        "이 정도면 됐어.",
        "알아서 해줘.",
        "더 묻지 않아도 될 것 같아.",
        "가능하면 넘어가자.",
        "일단 계속해.",
        "응, 좋아.",
        "그 부분은 됐고 다음 단계로.",
        "필요한 것만 알아서 처리해줘.",
        "뭐든 괜찮으니 진행해.",
    ]
    for message in ambiguous:
        cases.append(
            {
                "id": f"collection-{case_id:03d}",
                "message": message,
                "expected_directive": {
                    "action": "none",
                    "requires_clarification": True,
                },
                "expected_intent": "unclear",
                "tags": ["ambiguous", "clarification"],
            }
        )
        case_id += 1

    assert 50 <= len(cases) <= 200
    neutral = [
        ("제품의 핵심 강점을 추가로 말할게.", "provide_information"),
        ("스토리를 생성해줘.", "request_generation"),
        ("지금까지 입력한 정보를 보여줘.", "ask_question"),
        ("작성을 취소할게.", "cancel"),
    ]
    for message, intent in neutral:
        cases.append(
            {
                "id": f"collection-{case_id:03d}",
                "message": message,
                "expected_directive": {"action": "none"},
                "expected_intent": intent,
                "tags": ["none", intent],
            }
        )
        case_id += 1
    assert 50 <= len(cases) <= 200
    write(
        "story-worker-collection-evaluation-v1.json",
        {"schema_version": "story-worker-collection-evaluation-v1", "cases": cases},
    )


def build_question_dataset() -> None:
    cases: list[dict[str, Any]] = []
    purposes = [
        (
            "required",
            ["category"],
            None,
            "부족한 필수 제품 정보를 확인합니다.",
        ),
        (
            "required",
            ["key_strengths", "target_supporters"],
            None,
            "부족한 필수 제품 정보를 확인합니다.",
        ),
        (
            "optional-collect",
            OPTIONAL_GROUPS["story_persuasion"],
            "story_persuasion",
            "스토리 설득 정보를 보완합니다.",
        ),
        (
            "optional-collect",
            OPTIONAL_GROUPS["funding_configuration"],
            "funding_configuration",
            "펀딩 구성을 보완합니다.",
        ),
        ("optional-collect", OPTIONAL_GROUPS["policy"], "policy", "정책을 보완합니다."),
        (
            "optional-collect",
            OPTIONAL_GROUPS["project_explanation"],
            "project_explanation",
            "프로젝트 설명을 보완합니다.",
        ),
        ("clarify", ["target_supporters"], None, "'그분들'이 어떤 서포터를 뜻하는지 알려주세요."),
        (
            "clarify",
            ["rewards"],
            None,
            "가격을 바꾼다는 뜻인지 리워드를 추가한다는 뜻인지 알려주세요.",
        ),
        (
            "confirm-skip",
            ["refund_policy", "as_policy"],
            None,
            "정책 두 항목을 모두 생략할지 확인해 주세요.",
        ),
        ("confirm-skip", [], None, "남은 선택 정보 전체를 생략할지 명확히 알려주세요."),
    ]
    for index in range(100):
        purpose, candidates, group, detail = purposes[index % len(purposes)]
        candidate_copy = list(candidates)
        expected_fields = candidate_copy[:3]
        cases.append(
            {
                "id": f"question-{index + 1:03d}",
                "purpose": purpose,
                "candidate_fields": candidate_copy,
                "requested_group": group,
                "requested_detail": detail,
                "facts": {
                    "product_name": fact_value(PRODUCTS[index % len(PRODUCTS)][0]),
                    "product_type": fact_value(PRODUCTS[index % len(PRODUCTS)][1]),
                },
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{PRODUCTS[index % len(PRODUCTS)][0]} 펀딩 정보를 입력하고 있어."
                        ),
                    }
                ],
                "question_history": [],
                "expected_requested_fields": expected_fields,
                "tags": [purpose, group or "no-group"],
            }
        )
    write(
        "story-worker-question-evaluation-v1.json",
        {"schema_version": "story-worker-question-evaluation-v1", "cases": cases},
    )


def build_summary_dataset() -> None:
    cases: list[dict[str, Any]] = []
    for index in range(100):
        name, product_type, category, strength, target = PRODUCTS[index % len(PRODUCTS)]
        facts = {
            "product_name": fact_value(f"{name} {index + 1}"),
            "product_type": fact_value(product_type),
            "category": fact_value(category),
            "key_strengths": fact_value(strength),
            "target_supporters": fact_value(target),
        }
        field_states = {field: "skipped" for field in ALL_FIELDS if field not in REQUIRED_FIELDS}
        mode = index % 5
        if mode >= 1:
            facts["problem_context"] = fact_value(f"반복 작업에 하루 {index % 7 + 1}시간이 소요됨")
            field_states["problem_context"] = "resolved"
        if mode >= 2:
            facts["trust_elements"] = fact_value(f"내구성 시험 {100 + index}회 완료")
            field_states["trust_elements"] = "resolved"
        if mode >= 3:
            facts["refund_policy"] = fact_value(None, status="explicitly-absent")
            field_states["refund_policy"] = "resolved"
        if mode >= 4:
            facts["rewards"] = fact_value(f"얼리버드 {index + 1}개 한정")
            facts["shipping_start"] = fact_value(f"2027년 {index % 12 + 1}월 순차 발송")
            field_states["rewards"] = "resolved"
            field_states["shipping_start"] = "resolved"
        cases.append(
            {
                "id": f"summary-{index + 1:03d}",
                "facts": facts,
                "field_states": field_states,
                "messages": [{"role": "user", "content": "이 정보로 생성 전 내용을 정리해줘."}],
                "tags": [f"mode-{mode}"],
            }
        )
    write(
        "story-worker-summary-evaluation-v1.json",
        {"schema_version": "story-worker-summary-evaluation-v1", "cases": cases},
    )


def build_approval_dataset() -> None:
    phrases = {
        "approve": [
            "네, 이 요약 그대로 스토리를 생성해 주세요.",
            "확인했습니다. 현재 내용으로 생성을 시작해.",
            "수정 없이 이 내용으로 스토리 작성 진행을 승인합니다.",
            "요약 내용이 맞으니 그대로 생성해줘.",
            "지금 확정된 정보로 스토리를 만들어 주세요.",
        ],
        "revise": [
            "타깃을 1인 가구로 바꾼 뒤 다시 보여줘.",
            "핵심 강점에 저소음 기능을 추가할게.",
            "제품명은 오빗클린 V4로 수정해줘.",
            "리워드 가격을 45만 원으로 바꾸겠습니다.",
            "발송 일정은 미정으로 되돌려줘.",
        ],
        "reject": [
            "스토리를 생성하지 않을게.",
            "이번 작성은 여기서 취소해줘.",
            "생성하지 말고 종료하겠습니다.",
            "이 프로젝트의 스토리 작성은 중단해.",
            "승인하지 않습니다. 만들지 마세요.",
        ],
        "ambiguous": [
            "괜찮은 것 같아.",
            "이대로 해도 될까?",
            "조금 고민해볼게.",
            "필수 정보는 다 들어갔어?",
            "가능하면 이 방향으로 가자.",
        ],
    }
    cases: list[dict[str, Any]] = []
    index = 1
    for decision, examples in phrases.items():
        for repeat in range(5):
            for phrase in examples:
                cases.append(
                    {
                        "id": f"approval-{index:03d}",
                        "message": phrase if repeat == 0 else f"{phrase} ({repeat + 1}번째 확인)",
                        "expected_decision": decision,
                        "tags": [decision],
                    }
                )
                index += 1
    assert len(cases) == 100
    write(
        "story-worker-approval-evaluation-v1.json",
        {"schema_version": "story-worker-approval-evaluation-v1", "cases": cases},
    )


def full_required_message(product: tuple[str, str, str, str, str], index: int) -> str:
    name, product_type, category, strength, target = product
    return (
        f"제품명은 {name} {index}이고 {category} 카테고리의 {product_type}야. "
        f"핵심 강점은 {strength}, 주요 서포터는 {target}이야."
    )


def full_optional_message(index: int) -> str:
    return (
        f"해결 문제는 반복 작업 부담이고, 시험 {100 + index}회 완료가 신뢰 근거야. "
        "메이커는 관련 개발자 3명으로 구성됐어. "
        f"리워드는 얼리버드 {index}개, 종료일은 2027년 3월 1일, 발송은 4월부터야. "
        "제품 하자는 14일 이내 환불하고 1년 무상 수리할게. "
        "펀딩금은 초도 생산에 사용하고 초기 사용자 피드백을 받기 위해 와디즈를 선택했어. "
        "부품 지연은 공급사 이원화로 대응할게."
    )


def build_flow_dataset() -> None:
    cases: list[dict[str, Any]] = []
    for index in range(100):
        product = PRODUCTS[index % len(PRODUCTS)]
        scenario = index % 6
        first = full_required_message(product, index + 1)
        if scenario == 0:
            turns = [
                {
                    "message": first,
                    "expected_stage": "collecting",
                    "expected_phase": "optional-offer",
                },
                {
                    "message": "선택 정보는 모두 생략할게.",
                    "expected_stage": "awaiting-approval",
                    "expected_phase": "approval",
                },
                {
                    "message": "이 요약 그대로 스토리를 생성해 주세요.",
                    "expected_stage": "generation-ready",
                    "expected_phase": "generation-ready",
                },
            ]
        elif scenario == 1:
            turns = [
                {
                    "message": f"{first} {full_optional_message(index + 1)}",
                    "expected_stage": "awaiting-approval",
                    "expected_phase": "approval",
                },
                {
                    "message": "확인했습니다. 현재 내용 그대로 생성해 주세요.",
                    "expected_stage": "generation-ready",
                    "expected_phase": "generation-ready",
                },
            ]
        elif scenario == 2:
            turns = [
                {
                    "message": f"제품명은 {product[0]} {index + 1}이고 제품은 {product[1]}야.",
                    "expected_stage": "collecting",
                    "expected_phase": "required",
                },
                {
                    "message": (
                        f"카테고리는 {product[2]}, 강점은 {product[3]}, 타깃은 {product[4]}야."
                    ),
                    "expected_stage": "collecting",
                    "expected_phase": "optional-offer",
                },
                {
                    "message": "권장 순서로 선택 정보를 모두 입력할게.",
                    "expected_stage": "collecting",
                    "expected_phase": "optional-collect",
                },
                {
                    "message": (
                        "문제는 반복 작업 부담이고 시험 300회 완료, 메이커는 개발자 3명이야."
                    ),
                    "expected_stage": "collecting",
                    "expected_phase": "optional-collect",
                },
                {
                    "message": "리워드는 얼리버드 50개, 종료는 3월 1일, 발송은 4월부터야.",
                    "expected_stage": "collecting",
                    "expected_phase": "optional-collect",
                },
                {
                    "message": "제품 하자는 14일 이내 환불하고 1년 무상 수리할게.",
                    "expected_stage": "collecting",
                    "expected_phase": "optional-collect",
                },
                {
                    "message": (
                        "펀딩금은 초도 생산에 쓰고 피드백을 위해 와디즈를 선택했으며 "
                        "공급사 이원화로 대응할게."
                    ),
                    "expected_stage": "awaiting-approval",
                    "expected_phase": "approval",
                },
                {
                    "message": "이 요약 그대로 생성해 주세요.",
                    "expected_stage": "generation-ready",
                    "expected_phase": "generation-ready",
                },
            ]
        elif scenario == 3:
            turns = [
                {
                    "message": first,
                    "expected_stage": "collecting",
                    "expected_phase": "optional-offer",
                },
                {
                    "message": "그냥 진행해줘.",
                    "expected_stage": "collecting",
                    "expected_phase": "optional-offer",
                },
                {
                    "message": "남은 선택 정보 전체를 생략할게.",
                    "expected_stage": "awaiting-approval",
                    "expected_phase": "approval",
                },
                {
                    "message": "수정 없이 이 내용으로 스토리 생성을 승인합니다.",
                    "expected_stage": "generation-ready",
                    "expected_phase": "generation-ready",
                },
            ]
        elif scenario == 4:
            turns = [
                {
                    "message": first,
                    "expected_stage": "collecting",
                    "expected_phase": "optional-offer",
                },
                {
                    "message": "선택 정보 전체를 생략할게.",
                    "expected_stage": "awaiting-approval",
                    "expected_phase": "approval",
                },
                {
                    "message": "타깃을 초기 사용자로 바꾼 뒤 다시 요약해줘.",
                    "expected_stage": "awaiting-approval",
                    "expected_phase": "approval",
                },
                {
                    "message": "수정된 요약 그대로 스토리를 생성해 주세요.",
                    "expected_stage": "generation-ready",
                    "expected_phase": "generation-ready",
                },
            ]
        else:
            turns = [
                {
                    "message": "바로 스토리를 생성해줘.",
                    "expected_stage": "collecting",
                    "expected_phase": "required",
                },
                {
                    "message": "바로 스토리를 생성해줘.",
                    "expected_stage": "collecting",
                    "expected_phase": "required",
                },
                {
                    "message": "바로 스토리를 생성해줘.",
                    "expected_stage": "collecting",
                    "expected_phase": "required",
                },
                {
                    "message": first,
                    "expected_stage": "collecting",
                    "expected_phase": "optional-offer",
                },
                {
                    "message": "선택 정보 전체를 생략할게.",
                    "expected_stage": "awaiting-approval",
                    "expected_phase": "approval",
                },
                {
                    "message": "현재 요약 그대로 스토리 생성을 승인합니다.",
                    "expected_stage": "generation-ready",
                    "expected_phase": "generation-ready",
                },
            ]
        cases.append(
            {
                "id": f"flow-{index + 1:03d}",
                "scenario": f"scenario-{scenario}",
                "turns": turns,
            }
        )
    write(
        "story-worker-flow-evaluation-v1.json",
        {"schema_version": "story-worker-flow-evaluation-v1", "cases": cases},
    )


def main() -> None:
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    expand_field_dataset()
    build_collection_dataset()
    build_question_dataset()
    build_summary_dataset()
    build_approval_dataset()
    build_flow_dataset()


if __name__ == "__main__":
    main()
