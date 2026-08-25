from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(frozen=True, slots=True)
class StoryWarning:
    code: str
    message: str
    section_id: str | None = None
    source_fields: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "section_id": self.section_id,
            "source_fields": list(self.source_fields),
        }


def brief_source_fields(brief: dict[str, Any]) -> set[str]:
    values = {
        "product.name",
        "product.summary",
        "product.category",
        "product.product_type",
        "schedule_policy",
    }
    for group in (
        brief["product"]["facts"],
        brief["audiences"],
        brief["problems"],
        brief["features"],
        brief["claims"],
        brief["evidence"],
        brief["assets"],
        brief["rewards"],
    ):
        values.update(item["id"] for item in group)
    values.update(f"unknown.{item['field']}" for item in brief["unknowns"])
    return values


_NUMBER = re.compile(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?")
_ORDERED_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s+")
_UNKNOWN_OVERREACH = re.compile(
    r"확인\s*중|확정\s*준비\s*중|(?:추후|향후|추가).{0,20}(?:공지|안내|업데이트)|"
    r"정식\s*오픈\s*시|확정되는\s*대로|안내드리겠습니다|"
    r"(?:공지|안내)(?:될|할)\s*예정|확정\s*시.{0,30}확인"
)
_INTERNAL_IDENTIFIER = re.compile(r"\bunknown\.[A-Za-z0-9_.-]+\b")


def _number_key(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None


def _all_string_values(value: object) -> list[str]:
    if isinstance(value, dict):
        return [item for child in value.values() for item in _all_string_values(child)]
    if isinstance(value, list):
        return [item for child in value for item in _all_string_values(child)]
    return [str(value)] if isinstance(value, (str, int, float)) else []


def allowed_numbers(brief: dict[str, Any]) -> set[Decimal]:
    result: set[Decimal] = set()
    for text in _all_string_values(brief):
        for token in _NUMBER.findall(text):
            parsed = _number_key(token)
            if parsed is not None:
                result.add(parsed)
    return result


def claim_number_tokens(prose: str) -> list[str]:
    """Return factual number candidates, excluding ordered-list labels."""

    return _NUMBER.findall(_ORDERED_LIST_MARKER.sub("", prose))


class StoryValidator:
    def validate(
        self,
        *,
        content: dict[str, Any],
        brief: dict[str, Any],
        template: dict[str, Any],
    ) -> list[StoryWarning]:
        warnings: list[StoryWarning] = []
        expected = template["layout"]
        actual = content["sections"]
        expected_ids = [section["id"] for section in expected]
        actual_ids = [section["template_section_id"] for section in actual]

        for section_id in expected_ids:
            if section_id not in actual_ids:
                warnings.append(
                    StoryWarning(
                        "missing-template-section",
                        f"템플릿 섹션이 생성되지 않았습니다: {section_id}",
                        section_id,
                    )
                )
        for section_id in actual_ids:
            if section_id not in expected_ids:
                warnings.append(
                    StoryWarning(
                        "extra-template-section",
                        f"템플릿에 없는 섹션이 생성됐습니다: {section_id}",
                        section_id,
                    )
                )
        if actual_ids != expected_ids:
            warnings.append(
                StoryWarning(
                    "section-order-mismatch",
                    "생성 섹션의 순서가 템플릿 layout과 다릅니다.",
                )
            )

        expected_by_id = {section["id"]: section for section in expected}
        allowed_sources = brief_source_fields(brief)
        grounded_numbers = allowed_numbers(brief)
        for section in actual:
            section_id = section["template_section_id"]
            expected_section = expected_by_id.get(section_id)
            if expected_section is not None:
                if section["type"] != expected_section["type"]:
                    warnings.append(
                        StoryWarning(
                            "section-type-mismatch",
                            f"{section_id} type은 {expected_section['type']}이어야 합니다.",
                            section_id,
                        )
                    )
                if section["image_intent"]["required"] != expected_section["image_required"]:
                    warnings.append(
                        StoryWarning(
                            "image-contract-mismatch",
                            f"{section_id} 이미지 필요 여부가 템플릿과 다릅니다.",
                            section_id,
                        )
                    )

            referenced = set(section["source_fields"]) | set(
                section["image_intent"]["source_fields"]
            )
            unknown_sources = sorted(referenced - allowed_sources)
            if unknown_sources:
                warnings.append(
                    StoryWarning(
                        "unknown-source-field",
                        f"브리프에 없는 source_fields: {', '.join(unknown_sources)}",
                        section_id,
                        tuple(unknown_sources),
                    )
                )

            prose = f"{section['heading']} {section['body']}"
            unlisted = sorted(
                {
                    token
                    for token in claim_number_tokens(prose)
                    if _number_key(token) not in grounded_numbers
                }
            )
            if unlisted:
                warnings.append(
                    StoryWarning(
                        "unlisted-number",
                        f"브리프에서 찾지 못한 수치: {', '.join(unlisted)}",
                        section_id,
                        tuple(section["source_fields"]),
                    )
                )
            if any(source.startswith("unknown.") for source in referenced) and (
                match := _UNKNOWN_OVERREACH.search(prose)
            ):
                warnings.append(
                    StoryWarning(
                        "unsupported-future-commitment",
                        "미확인 정보에 입력되지 않은 현재 진행 상태 또는 미래 약속을 "
                        f"추가했습니다: {match.group(0)}",
                        section_id,
                        tuple(
                            sorted(
                                source
                                for source in referenced
                                if source.startswith("unknown.")
                            )
                        ),
                    )
                )
            if match := _INTERNAL_IDENTIFIER.search(prose):
                warnings.append(
                    StoryWarning(
                        "internal-identifier-leak",
                        f"사용자 본문에 내부 필드 ID를 노출했습니다: {match.group(0)}",
                        section_id,
                        tuple(section["source_fields"]),
                    )
                )
        return warnings
