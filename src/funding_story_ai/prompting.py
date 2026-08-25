from __future__ import annotations

import json
from typing import Any


def build_story_prompt(
    *,
    brief: dict[str, Any],
    template: dict[str, Any],
    previous_content: dict[str, Any] | None = None,
    validation_messages: list[str] | None = None,
) -> str:
    correction = ""
    if previous_content is not None and validation_messages:
        correction = (
            "\n이전 출력은 아래 검증 문제를 포함했습니다. 템플릿 골격은 유지하고 문제만 "
            "수정해 전체 JSON을 다시 작성하세요.\n"
            f"검증 문제: {json.dumps(validation_messages, ensure_ascii=False)}\n"
            f"이전 출력: {json.dumps(previous_content, ensure_ascii=False)}\n"
        )

    language = {
        "ko": "한국어",
        "en": "영어",
        "ja": "일본어",
        "zh": "중국어",
    }[brief["language"]]
    return f"""당신은 {language} 크라우드펀딩 스토리 초안을 작성합니다.

목표:
- 제공된 템플릿의 섹션 순서·역할·스타일을 채웁니다.
- 브리프에 있는 사실과 명시된 미확인 정보만 사용합니다.
- 템플릿의 문구를 그대로 복사하지 말고 해당 제품에 맞는 새 문장으로 작성합니다.

사실성 규칙:
1. 입력에 없는 가격, 할인율, 인증, 수상, 후기, 판매량, 일정, AS, 환불, 성능 수치와
   비교 우위를 만들지 마세요.
2. unknown 항목이 필요한 섹션은 값을 추정하지 말고 무엇이 미입력인지 명확히 쓰세요.
3. 수치에는 브리프에 제공된 조건을 같은 문단에 표시하세요.
4. 합성 자체 시험을 외부 기관 인증이나 실제 판매 제품 시험으로 표현하지 마세요.
5. 각 섹션의 source_fields에는 그 문장을 뒷받침하는 브리프 ID 또는 경로만 넣으세요.
6. 감성 문구도 입력과 충돌해서는 안 됩니다.
7. 제품군에서 흔하다는 이유로 입력에 없는 동작·관리 방식·소유 관계를 보완하지 마세요.
8. 유사한 표현이라도 기능·효과·사용 범위를 확대하지 마세요.
9. 입력으로 뒷받침되지 않는 세부 동작은 미확인 문구로 채우기보다 문장에서 생략하세요.
10. unknown.* 같은 내부 필드 ID나 source_fields 값을 제목·본문에 노출하지 마세요.
11. `자동`, `완전`, `필요 없음`, `방지`, `해결`처럼 동작 또는 효과를 강화하는 표현은
    브리프에 같은 의미가 명시된 경우에만 사용하세요.

출력 규칙:
- JSON만 반환합니다.
- title_candidates는 1~3개입니다.
- sections는 템플릿 layout과 같은 개수·순서·template_section_id·type을 사용합니다.
- body는 원시 HTML이 아닌 안전한 Markdown으로 작성합니다. 템플릿 content_type에 맞게
  문단, `-` 목록, `1.` 순서 목록, `|` 표, `>` 인용, `**굵게**`, `==강조==`를
  필요한 섹션에서 사용하되 사실 근거가 없는 행·항목·인용은 만들지 마세요.
- 표를 요구하는 섹션에 입력값이 없으면 수치나 계획을 채운 표를 만들지 말고
  미입력 항목을 한 문단으로 표시하세요.
- image_intent.required는 템플릿 image_required와 같아야 합니다.
- 이미지가 필요하지 않으면 purpose와 visual_hint는 빈 문자열, source_fields는 빈 배열로 둡니다.
- 이미지가 필요하면 템플릿 visual_hint를 제품 사실에 맞게 구체화하고
  사용할 source_fields를 적습니다.

템플릿 명세:
{json.dumps(template, ensure_ascii=False)}

제품 브리프:
{json.dumps(brief, ensure_ascii=False)}
{correction}
"""
