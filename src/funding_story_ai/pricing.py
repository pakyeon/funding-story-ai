from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

MILLION = Decimal("1000000")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0

    @property
    def billable_output_tokens(self) -> int:
        return self.output_tokens + self.thinking_tokens


@dataclass(frozen=True, slots=True)
class ModelPrice:
    input_usd_per_million: Decimal
    output_usd_per_million: Decimal
    source: str

    def estimate_usd(self, usage: TokenUsage) -> Decimal:
        input_cost = Decimal(usage.prompt_tokens) * self.input_usd_per_million / MILLION
        output_cost = (
            Decimal(usage.billable_output_tokens) * self.output_usd_per_million / MILLION
        )
        return input_cost + output_cost


class PricingCatalog:
    """Explicit model prices used for estimates and the preflight budget guard."""

    _GEMINI_36_STANDARD = ModelPrice(
        input_usd_per_million=Decimal("1.50"),
        output_usd_per_million=Decimal("7.50"),
        source="google-standard-pricing-2026-08-14",
    )
    _GEMINI_37_INTRODUCTORY = ModelPrice(
        input_usd_per_million=Decimal("0.75"),
        output_usd_per_million=Decimal("3.75"),
        source="google-introductory-pricing-2026-08-13-through-2026-12-31",
    )
    _GEMINI_37_POST_INTRODUCTORY = ModelPrice(
        input_usd_per_million=Decimal("1.50"),
        output_usd_per_million=Decimal("7.50"),
        source="google-post-introductory-pricing-effective-2027-01-01",
    )

    def __init__(self, prices: dict[str, ModelPrice] | None = None) -> None:
        self._prices = prices or self._prices_from_env()

    @classmethod
    def _default_gemini_37_price(cls, *, today: date | None = None) -> ModelPrice:
        effective_date = today or date.today()
        if effective_date <= date(2026, 12, 31):
            return cls._GEMINI_37_INTRODUCTORY
        return cls._GEMINI_37_POST_INTRODUCTORY

    @classmethod
    def _prices_from_env(cls) -> dict[str, ModelPrice]:
        input_37 = os.getenv("GEMINI_37_INPUT_USD_PER_MILLION")
        output_37 = os.getenv("GEMINI_37_OUTPUT_USD_PER_MILLION")
        if bool(input_37) != bool(output_37):
            raise ValueError("Both Gemini 3.7 price environment variables must be set together")

        price_37 = (
            ModelPrice(
                input_usd_per_million=Decimal(input_37),
                output_usd_per_million=Decimal(output_37),
                source="configured-gemini-3.7-price",
            )
            if input_37 and output_37
            else cls._default_gemini_37_price()
        )
        return {
            "gemini-3.7-flash": price_37,
            "gemini-3.6-flash": cls._GEMINI_36_STANDARD,
        }

    def get(self, model: str) -> ModelPrice:
        try:
            return self._prices[model]
        except KeyError as exc:
            raise ValueError(f"No price configured for model: {model}") from exc
