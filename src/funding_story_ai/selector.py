from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TemplateSelection:
    template_id: str
    score: int
    scores: dict[str, int]
    reasons: tuple[str, ...]


def _normalized(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).lower()


def _brief_text(brief: dict[str, Any]) -> str:
    parts: list[str] = [
        brief["product"]["name"],
        brief["product"]["category"],
        brief["product"]["product_type"],
        brief["product"]["summary"],
    ]
    for group in ("audiences", "problems", "features", "claims", "evidence"):
        for item in brief[group]:
            parts.extend(str(value) for value in item.values() if isinstance(value, str))
    return _normalized(" ".join(parts))


class TemplateSelector:
    """Transparent local fallback selector."""

    def select(
        self,
        brief: dict[str, Any],
        templates: list[dict[str, Any]],
    ) -> TemplateSelection:
        category = brief["product"]["category"]
        eligible = [template for template in templates if template["category"] == category]
        if not eligible:
            raise ValueError(f"No template for category: {category}")

        text = _brief_text(brief)
        scores: dict[str, int] = {}
        reasons_by_id: dict[str, list[str]] = {}
        for template in eligible:
            template_id = template["id"]
            reasons = ["category exact match"]
            score = 10
            matches = [
                keyword
                for keyword in template["content_strategy"]["product_keywords"]
                if _normalized(keyword) in text
            ]
            score += len(matches) * 2
            if matches:
                reasons.append(f"keyword matches: {', '.join(matches)}")

            if template_id == "t01_performance_value_evidence":
                score += min(len(brief["product"]["facts"]), 4)
                score += min(len(brief["evidence"]), 2)
                if not brief["rewards"]:
                    score -= 4
                    reasons.append("reward/value input missing")
            elif template_id == "t02_problem_solution_automation":
                score += min(len(brief["problems"]), 4)
                automation_signals = sum(
                    any(
                        token in _normalized(feature["name"] + feature["description"])
                        for token in ("자동", "도킹", "앱")
                    )
                    for feature in brief["features"]
                )
                score += automation_signals * 2
                reasons.append(f"automation/problem signals: {automation_signals}")
            elif template_id == "t03_lifestyle_social_proof":
                score += min(len(brief["audiences"]), 3)
                social_evidence = sum(
                    evidence["evidence_type"] in {"review", "award"}
                    for evidence in brief["evidence"]
                )
                score += social_evidence * 3
                if social_evidence == 0:
                    score -= 5
                    reasons.append("social proof input missing")

            scores[template_id] = score
            reasons_by_id[template_id] = reasons

        ranked = sorted(scores, key=lambda item: (-scores[item], item))
        winner = ranked[0]
        return TemplateSelection(
            template_id=winner,
            score=scores[winner],
            scores=scores,
            reasons=tuple(reasons_by_id[winner]),
        )
